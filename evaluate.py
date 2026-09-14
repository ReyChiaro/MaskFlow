import os
import random
from collections import Counter
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.distributed as dist

from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Subset
from torchvision.utils import save_image
from tqdm import tqdm

from data_module.dataset import SchemaDataset
from data_module.dataloader import get_dataloader
from data_module.sample_utils import image_name as sample_image_name
from pipelines.base_pipeline import BasePipeline
from trainer.lora_utils import merge_lora


def load_lora_adapters(pipe: BasePipeline, adapter_cfgs: DictConfig | None):
    if adapter_cfgs is None:
        return
    for adapter_type, path_and_cfg in adapter_cfgs.items():
        path = path_and_cfg.path
        cfg = path_and_cfg.cfg
        if not path and path_and_cfg.get("optional", False):
            continue
        if not path:
            raise ValueError(f"adapters.{adapter_type}.path must point to a trained LoRA checkpoint.")
        logger.info(f"Load adapter {adapter_type} from {path}.")
        merge_lora(pipe.transformer, path, cfg.adapter_name, lora_scale=1.0)
    logger.info("All adapters have been loaded into the model.")


@hydra.main(config_path="configs", config_name="eval_qwenimage_maskflow", version_base="v1.2")
def evaluate(cfgs: DictConfig):
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    if world_size > 1:
        dist.init_process_group("nccl")
    try:
        # Hydra's timestamp can resolve differently in separate workers.
        output_dir = [str(cfgs.project.evaluation_dir) if rank == 0 else None]
        if world_size > 1:
            dist.broadcast_object_list(output_dir, src=0, device=device)
        run_evaluation(cfgs, device, rank, world_size, Path(output_dir[0]))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@torch.inference_mode()
def run_evaluation(cfgs: DictConfig, device: torch.device, rank: int, world_size: int, evaluate_dir: Path):
    cfg_contents = "\n" + " Configs ".center(50, "=")
    cfg_contents += "\n" + OmegaConf.to_yaml(cfgs)
    cfg_contents += "\n" + "=" * 50
    logger.info(cfg_contents)

    # -------- Initialize Environment -------- #
    base_seed = cfgs.base_seed

    weight_dtype = torch.float32
    if cfgs.weight_dtype == "bf16":
        weight_dtype = torch.bfloat16
    elif cfgs.weight_dtype == "fp16":
        weight_dtype = torch.float16

    # Use the same seed across different device for one evaluation
    generator = torch.Generator(device).manual_seed(base_seed)

    random.seed(base_seed)
    np.random.seed(base_seed)
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed(base_seed)
    torch.cuda.manual_seed_all(base_seed)

    prediction_dir = evaluate_dir / "predictions"
    mask_dir = evaluate_dir / "mask"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Prediction outputs will be saved to {prediction_dir}.")
    logger.info(f"Masks will be saved to {mask_dir}.")

    # -------- Initialize Dataset -------- #
    evalset: SchemaDataset = instantiate(cfgs.evalset)
    # Preserve original filenames, but reject collisions before saving results.
    names = Counter(sample_image_name(sample) for sample in evalset.samples)
    duplicates = [name for name, count in names.items() if count > 1]
    if duplicates:
        raise ValueError(f"Evaluation output names must be unique; duplicates: {duplicates[:5]}")

    # No padding or dropping: every sample belongs to exactly one worker.
    local_evalset = Subset(evalset, range(rank, len(evalset), world_size))
    eval_loader, _ = get_dataloader(
        local_evalset,
        batch_size_per_process=cfgs.batch_size_per_process,
        num_workers=cfgs.num_workers,
        num_replicas=1,
        global_rank=0,
        global_seed=base_seed,
        drop_last=False,
        is_train=False,
    )
    logger.info(f"Rank {rank}/{world_size}: {len(local_evalset)}/{len(evalset)} samples, {len(eval_loader)} batches.")
    if len(local_evalset) == 0:
        logger.info(f"Rank {rank} has no samples to evaluate.")
        return

    # -------- Pipeline and LoRA loading -------- #
    pipe: BasePipeline = instantiate(cfgs.pipeline, device=device, generator=generator, dtype=weight_dtype)
    load_lora_adapters(pipe, cfgs.adapters)
    pipe.transformer.requires_grad_(False).eval()
    pipe.vae.eval()
    pipe.text_pipeline.text_encoder.eval()

    # -------- Evaluation Preparation -------- #
    text_cfg_scale = cfgs.text_cfg_scale
    cfg_kwargs = {}
    if cfgs.get("mask_cfg_scale", 1.0) != 1.0:
        cfg_kwargs["mask_cfg_scale"] = cfgs.mask_cfg_scale
    if cfgs.get("interaction_cfg_scale") is not None:
        cfg_kwargs["interaction_cfg_scale"] = cfgs.interaction_cfg_scale
    num_inference_steps = cfgs.num_inference_steps
    eval_with_position_prompt = cfgs.eval_with_position_prompt

    for step, batch in tqdm(enumerate(eval_loader), desc=f"Eval rank {rank}", total=len(eval_loader)):
        if eval_with_position_prompt:
            batch["prompt"] = [p for p in batch["edit_instruction"]]
        output: dict[str, torch.Tensor] = pipe.eval_step(
            batch,
            num_inference_steps=num_inference_steps,
            text_cfg_scale=text_cfg_scale,
            **cfg_kwargs,
        )
        for i, image_name in enumerate(batch["image_name"]):
            pred_path = prediction_dir / f"{image_name}.png"
            mask_path = mask_dir / f"{image_name}.png"
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            save_image(output["output"][i], pred_path)
            if "mask" in output:
                save_image(output["mask"][i], mask_path)
        logger.info(f"Rank {rank}: Eval [{step + 1}/{len(eval_loader)}] saved.")
    logger.info(f"Rank {rank}: evaluation finished, saved to {evaluate_dir}.")


if __name__ == "__main__":
    evaluate()
