import os
import random
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.distributed as dist

from hydra.utils import instantiate
from loguru import logger
from omegaconf import OmegaConf
from torchvision.utils import save_image
from tqdm import tqdm

from data_module.dataset import SchemaDataset
from data_module.dataloader import get_dataloader
from pipelines.base_pipeline import BasePipeline
from trainer.lora_utils import merge_lora


def load_lora_adapters(pipe: BasePipeline, adapter_cfgs: OmegaConf):
    cfgs = adapter_cfgs.to_container(adapter_cfgs, resolve=True)
    for adapter_type, path_and_cfg in cfgs.items():
        path = path_and_cfg.path
        cfg = path_and_cfg.cfg
        logger.info(f"Load adapter {adapter_type} from {path_and_cfg}.")
        merge_lora(pipe.transformer, path, cfg.adapter_name, lora_scale=1.0)
    logger.info(f"All adapters has been loaded into model.")


@hydra.main(config_path="configs", config_name="eval_maskflow", version_base="v1.2")
def evaluate(cfgs: OmegaConf):
    cfg_contents = "\n" + " Configs ".center(50, "=")
    cfg_contents += "\n" + OmegaConf.to_yaml(cfgs)
    cfg_contents += "\n" + "=" * 50
    logger.info(cfg_contents)

    # -------- Initialize Environment -------- #
    dist.init_process_group("nccl")

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    base_seed = cfgs.base_seed
    local_seed = base_seed + rank

    weight_dtype = torch.float32
    if cfgs.weight_dtype == "bf16":
        weight_dtype = torch.bfloat16
    elif cfgs.weight_dtype == "fp16":
        weight_dtype = torch.float16
    device = torch.device(f"cuda:{local_rank}")

    # Use the same seed across different device for one evaluation
    generator = torch.Generator(device).manual_seed(base_seed)

    random.seed(base_seed)
    np.random.seed(base_seed)
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed(base_seed)
    torch.cuda.manual_seed_all(base_seed)

    evaluate_dir = Path(cfgs.project.evaluation_dir)
    prediction_dir = evaluate_dir / "predictions"
    mask_dir = evaluate_dir / "mask"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Prediction outputs will be saved to {prediction_dir}.")
    logger.info(f"Masks will be saved to {mask_dir}.")

    # -------- Initialize Dataset -------- #
    evalset: SchemaDataset = instantiate(cfgs.evalset)
    eval_loader, eval_sampler = get_dataloader(
        evalset,
        batch_size_per_process=cfgs.batch_size_per_process,
        num_workers=cfgs.num_workers,
        num_replicas=world_size,
        global_rank=rank,
        global_seed=base_seed,
        drop_last=False,
        is_train=False,
    )
    logger.info(f"Eval Dataloader and Sampler initialized, length: {len(eval_loader)}.")

    # -------- Pipeline and LoRA loading -------- #
    pipe: BasePipeline = instantiate(cfgs.pipeline, device=device, generator=generator, dtype=weight_dtype)
    load_lora_adapters(pipe, cfgs.adapters)
    pipe.transformer.requires_grad_(False)

    # -------- Evaluation Preparation -------- #
    text_cfg_scale = cfgs.text_cfg_scale
    num_inference_steps = cfgs.num_inference_steps
    eval_with_position_prompt = cfgs.eval_with_position_prompt

    eval_sampler.set_epoch(0)
    for step, batch in tqdm(enumerate(eval_loader), desc="Eval", total=len(eval_loader)):
        if eval_with_position_prompt:
            batch["prompt"] = [p for p in batch["edit_instruction"]]
        output: dict[str, torch.Tensor] = pipe.eval_step(
            batch,
            num_inference_steps=num_inference_steps,
            text_cfg_scale=text_cfg_scale,
        )
        for i in range(cfgs.batch_size_per_process):
            pred_path = prediction_dir / f'{batch["image_name"][i]}.jpg'
            mask_path = mask_dir / f'{batch["image_name"][i]}.png'
            save_image(output["output"][i], pred_path)
            save_image(output["mask"][i], mask_path)
        logger.info(f"Eval [{step + 1}/{len(eval_loader)}] saved.")
    logger.info(f"Evaluation finished, saved to {evaluate_dir}.")


if __name__ == "__main__":
    evaluate()
