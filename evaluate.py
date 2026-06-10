import os
import time

import hydra
import torch
from hydra.utils import instantiate
from loguru import logger
from omegaconf import OmegaConf
from torchvision.utils import save_image
from tqdm import tqdm

from data_module import MaskEditDataset
from pipelines import BasePipeline


def get_value(cfg: OmegaConf, key: str, default, value_type):
    value = getattr(cfg, key, default)
    return default if value == default else value_type(value)


def _to_batched_sample(sample: dict) -> dict:
    return {
        "prompt": [sample["prompt"]],
        "negative_prompt": [sample.get("negative_prompt", "")],
        "conditions": [condition.unsqueeze(0) for condition in sample["conditions"]],
        "target": sample["target"].unsqueeze(0),
    }


def _mask_output_name(sample: dict) -> str:
    mask_path = sample["conditions"][1]
    return os.path.splitext(os.path.basename(mask_path))[0]


@hydra.main(config_path="configs", config_name="inference", version_base="v1.2")
def evaluate(cfgs: OmegaConf):
    r"""
    Run LoRA evaluation over a JSONL file without building a DataLoader.

    Common CLI overrides:
        +data_file=outputs/data_cache/demo.jsonl
        +image_root=.
        +output_dir=outputs/evaluations/demo
        +lora_model=outputs/checkpoints/step-xxx/adapter
        +lora_adapter_name=default
    """
    data_file = get_value(cfgs, "data_file", "", str)
    image_root = get_value(cfgs, "image_root", "", str)
    output_dir = get_value(cfgs, "output_dir", "outputs/evaluations", str)
    data_load_ratio = get_value(cfgs, "data_load_ratio", 1.0, float)

    max_resolution = get_value(cfgs, "max_resolution", 1024 * 1024, int)
    divisible_by = get_value(cfgs, "divisible_by", 16, int)
    enable_prompt_truncation = get_value(cfgs, "enable_prompt_truncation", False, bool)
    replace_prompt_placeholder_with = get_value(
        cfgs,
        "replace_prompt_placeholder_with",
        "the masked area in the image 2",
        str,
    )

    rank = get_value(cfgs, "rank", 0, int)
    seed = get_value(cfgs, "seed", int(time.time()), int)
    num_inference_steps = get_value(cfgs, "num_inference_steps", 50, int)
    cfg_scale = get_value(cfgs, "cfg_scale", 0.0, float)
    lora_model = get_value(cfgs, "lora_model", None, str)
    lora_adapter_name = get_value(cfgs, "lora_adapter_name", "default", str)

    if not data_file:
        raise ValueError("`data_file` must be provided, e.g. data_file=outputs/data_cache/demo.jsonl")

    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    generator = torch.Generator(device).manual_seed(seed)

    dataset = MaskEditDataset(
        image_root=image_root,
        data_file=data_file,
        data_load_ratio=data_load_ratio,
        max_resolution=max_resolution,
        divisible_by=divisible_by,
        enable_prompt_truncation=enable_prompt_truncation,
        replace_prompt_placeholder_with=replace_prompt_placeholder_with,
    )

    pipe: BasePipeline = instantiate(cfgs.pipeline, generator=generator, device=device, dtype=dtype)
    if lora_model is not None and os.path.exists(lora_model):
        pipe.transformer.load_lora_adapter(lora_model, prefix=None, adapter_name=lora_adapter_name)
        pipe.transformer.set_adapter(lora_adapter_name)
        pipe.transformer.requires_grad_(False)
        logger.info(f"Loaded LoRA adapter from {lora_model}.")
    elif lora_model is not None:
        raise FileNotFoundError(f"LoRA model path does not exist: {lora_model}")

    logger.info(f"Evaluate {len(dataset)} samples, save to {output_dir}.")
    for index in tqdm(range(len(dataset)), desc="Evaluating"):
        raw_sample = dataset.samples[index]
        batch = _to_batched_sample(dataset[index])

        with torch.inference_mode():
            output = pipe.eval_step(batch, index, num_inference_steps, cfg_scale)

        if isinstance(output, (tuple, list)):
            output = output[-1]

        save_path = os.path.join(output_dir, f"{_mask_output_name(raw_sample)}.jpg")
        save_image(output[0].float().clamp(0, 1), save_path)
        logger.info(f"Saved {save_path}.")

    logger.info(f"Evaluation finished, saved to {output_dir}.")


if __name__ == "__main__":
    evaluate()
