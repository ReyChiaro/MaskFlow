"""Standalone QwenImage MaskFlow module benchmark (no evaluate.py dependency)."""

import json
import os
import random
from functools import partial
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms.functional as T
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torchvision.utils import save_image

from data_module.utils import preprocess_images
from trainer.lora_utils import merge_lora
from trainer.parallel.handler import parallel_handler
from utils.stage_timing import StageTimer, maskflow_targets, summarize


def validate_config(cfg):
    for key in ("source", "mask"):
        if not cfg.input.get(key) or not Path(cfg.input[key]).is_file():
            raise ValueError(f"input.{key} must name an existing image")
    if not isinstance(cfg.input.prompt, str):
        raise ValueError("input.prompt must be a string")
    if cfg.runtime.fsdp_strategy not in {"no_shard", "full_shard"}:
        raise ValueError("runtime.fsdp_strategy must be no_shard or full_shard")
    if cfg.runtime.dtype not in {"bfloat16", "float16", "float32"}:
        raise ValueError("runtime.dtype must be bfloat16, float16 or float32")
    for name, minimum in (("warmup", 0), ("repeats", 1), ("batch_size", 1), ("num_inference_steps", 1)):
        value = cfg.runtime[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"runtime.{name} must be an integer >= {minimum}")
    if cfg.pipeline.velocity_visualization_dir is not None:
        raise ValueError("Disable velocity_visualization_dir to profile a single inference trajectory")
    if cfg.preprocessing.max_resolution < cfg.pipeline.divisible_by ** 2:
        raise ValueError("preprocessing.max_resolution is too small")


def make_batch(cfg):
    images = {}
    for key in ("source", "mask"):
        with Image.open(cfg.input[key]) as image:
            images[key] = T.to_tensor(image.convert("RGB"))
    if images["source"].shape != images["mask"].shape:
        raise ValueError("Source and mask must have matching dimensions")
    conditions, _ = preprocess_images(
        images, max_resolution=cfg.preprocessing.max_resolution,
        divisible_by=cfg.pipeline.divisible_by,
    )
    if min(conditions["source"].shape[-2:]) == 0:
        raise ValueError("Preprocessing produced an empty image")
    size = cfg.runtime.batch_size
    return dict(
        prompt=[cfg.input.prompt] * size,
        negative_prompt=[cfg.input.negative_prompt] * size,
        conditions={key: value.unsqueeze(0).repeat(size, 1, 1, 1) for key, value in conditions.items()},
    )


def build_pipeline(cfg, device):
    """Construct, optionally fuse LoRA on CPU, shard, then load weights."""
    dtype = getattr(torch, cfg.runtime.dtype)
    pipe = instantiate(cfg.pipeline, device=device, dtype=dtype)
    if cfg.checkpoint.path:
        pipe.configure_model(
            pipe.transformer,
            partial(merge_lora, lora_path=cfg.checkpoint.path,
                    adapter_name=cfg.checkpoint.adapter_name,
                    lora_scale=cfg.checkpoint.scale, weight_name=cfg.checkpoint.weight_name),
            update_structure=False,
        )
    else:
        logger.warning("No MaskFlow LoRA supplied; benchmarking the base transformer weights.")
    full_shard = cfg.runtime.fsdp_strategy == "full_shard"
    if full_shard:
        pipe.setup_fsdp_modules("full_shard", device, dtype)
    pipe.load_pretrained_weights(broadcast=full_shard)
    for module in (pipe.vae, pipe.text_pipeline.text_encoder, pipe.transformer):
        module.requires_grad_(False).eval()
    return pipe


@torch.inference_mode()
def benchmark(pipe, batch, cfg, device):
    kwargs = dict(
        num_inference_steps=cfg.runtime.num_inference_steps,
        text_cfg_scale=cfg.runtime.text_cfg_scale,
        mask_cfg_scale=cfg.runtime.mask_cfg_scale,
        interaction_cfg_scale=cfg.runtime.interaction_cfg_scale,
        seed=cfg.runtime.seed,
    )
    timer = StageTimer(maskflow_targets(pipe), device=device)
    output = None
    with timer:
        for index in range(cfg.runtime.warmup):
            pipe.eval_step(batch, **kwargs)
            torch.cuda.synchronize(device)
            logger.info(f"Warmup {index + 1}/{cfg.runtime.warmup} complete")
        for index in range(cfg.runtime.repeats):
            # Keep FSDP workers aligned, but exclude the barrier from inference time.
            if dist.is_initialized():
                dist.barrier()
            output = None
            with timer.iteration(index):
                output = pipe.eval_step(batch, **kwargs)
            logger.info(f"Iteration {index + 1}: {timer.iterations[-1]['wall_ms']:.2f} ms")
    return timer, output


def summarize_ranks(reports):
    # FSDP ranks cooperate on each batch; never add their latency together.
    slowest = [
        dict(iteration=row["iteration"], wall_ms=max(r["iterations"][i]["wall_ms"] for r in reports))
        for i, row in enumerate(reports[0]["iterations"])
    ]
    return dict(
        slowest_rank_iterations=slowest,
        slowest_rank_wall=summarize([row["wall_ms"] for row in slowest]),
        per_rank=[dict(rank=r["rank"], stages=r["stages"], inference_wall=r["inference_wall"])
                  for r in reports],
    )


@hydra.main(version_base=None, config_path="configs", config_name="profile_qwenimage_maskflow")
def main(cfg: DictConfig):
    validate_config(cfg)
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA; no CPU timing fallback is used")
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    if world_size > 1 and cfg.runtime.fsdp_strategy != "full_shard":
        raise ValueError("Use full_shard for multi-GPU profiling, or launch a single process")
    torch.cuda.set_device(device)
    random.seed(cfg.runtime.seed)
    np.random.seed(cfg.runtime.seed)
    torch.manual_seed(cfg.runtime.seed)
    try:
        if cfg.runtime.fsdp_strategy == "full_shard":
            parallel_handler.setup_parallel()
        # Broadcast the run directory because Hydra timestamps may differ by rank.
        paths = [str(Path(cfg.output.dir).resolve()) if rank == 0 else None]
        if dist.is_initialized():
            dist.broadcast_object_list(paths, src=0, device=device)
        output_dir = Path(paths[0])
        output_dir.mkdir(parents=True, exist_ok=True)
        batch = make_batch(cfg)
        pipe = build_pipeline(cfg, device)
        timer, result = benchmark(pipe, batch, cfg, device)
        report = timer.report()
        report.update(
            rank=rank, world_size=world_size, device=torch.cuda.get_device_name(device),
            torch_version=torch.__version__, input_shape=list(batch["conditions"]["source"].shape),
            output_dir=str(output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            scope="CUDA current-stream intervals inside patched methods; outer module hooks excluded",
        )
        (output_dir / f"timings_rank_{rank}.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        timer.write_calls(output_dir / f"calls_rank_{rank}.csv")
        reports = [None] * world_size
        if dist.is_initialized():
            dist.all_gather_object(reports, report)
        else:
            reports[0] = report
        if rank == 0:
            summary = summarize_ranks(reports)
            (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            for name, stats in report["stages"].items():
                logger.info(
                    f"Rank 0 {name}: calls={stats['per_call']['count']}, "
                    f"mean/call={stats['per_call']['mean_ms']:.2f} ms, "
                    f"mean/iteration={stats['per_iteration']['mean_ms']:.2f} ms"
                )
            logger.info(f"Slowest-rank mean inference: {summary['slowest_rank_wall']['mean_ms']:.2f} ms")
            if cfg.output.save_image:
                save_image(result["output"].float().cpu(), output_dir / "prediction.png")
            logger.info(f"Profiling results: {output_dir}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
