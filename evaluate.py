import json
import random
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch

from hydra.utils import instantiate
from loguru import logger
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from pipelines.base_pipeline import BasePipeline


def _select(cfgs: OmegaConf, key: str, default: Any = None) -> Any:
    return OmegaConf.select(cfgs, key, default=default)


def _seed_everything(seed: int, device: torch.device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _load_lora_adapter(pipe: BasePipeline, cfgs: OmegaConf):
    r"""
    Load LoRA weights for evaluation.

    Args: cfgs can include following options
        +is_fsdp_checkpoint (bool)
        +lora_safetensors_dir (str): If a DCP/FSDP checkpoint is given,
            it will be converted into safetensors for convenient reuse.
    """
    resume_from = _select(cfgs, "resume_from")
    if not resume_from or not Path(resume_from).exists():
        if resume_from:
            logger.warning(f"LoRA checkpoint not found: {resume_from}.")
        return

    adapter_name = cfgs.adapter.adapter_name
    safetensors_dir = resume_from

    if _select(cfgs, "is_fsdp_checkpoint", False):
        import peft
        import torch.distributed.checkpoint as DCP
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            get_model_state_dict,
            set_model_state_dict,
        )

        lora_configs = peft.LoraConfig(
            r=cfgs.adapter.r,
            lora_alpha=cfgs.adapter.lora_alpha,
            lora_dropout=cfgs.adapter.lora_dropout,
            bias="none",
            target_modules=list(cfgs.adapter.target_modules),
        )
        pipe.transformer.add_adapter(lora_configs, adapter_name=adapter_name)
        pipe.transformer.set_adapter(adapter_name)

        for name, param in pipe.transformer.named_parameters():
            if adapter_name in name and "lora_" in name:
                param.requires_grad_(True)

        transformer_states = get_model_state_dict(
            pipe.transformer,
            options=StateDictOptions(full_state_dict=False, ignore_frozen_params=True),
        )
        DCP.load({"model": transformer_states}, checkpoint_id=str(resume_from))
        set_model_state_dict(
            pipe.transformer,
            transformer_states,
            options=StateDictOptions(full_state_dict=False, ignore_frozen_params=True, strict=False),
        )

        safetensors_dir = Path(_select(cfgs, "lora_safetensors_dir", f"{resume_from}/lora_adapter"))
        safetensors_dir.mkdir(exist_ok=True, parents=True)
        pipe.transformer.save_lora_adapter(
            safetensors_dir,
            adapter_name=adapter_name,
            safe_serialization=True,
        )
        pipe.transformer.delete_adapters(adapter_name)
        logger.info(f"Converted DCP checkpoint {resume_from} to LoRA safetensors at {safetensors_dir}.")

    pipe.transformer.load_lora_adapter(
        safetensors_dir,
        prefix=None,
        adapter_name=adapter_name,
        use_safetensors=True,
    )
    pipe.transformer.set_adapter(adapter_name)
    logger.info(f"Loaded LoRA adapter '{adapter_name}' from {safetensors_dir}.")


def _normalize_outputs(output: Any) -> dict[str, torch.Tensor]:
    if not isinstance(output, dict):
        raise TypeError(f"eval_step must return dict[str, Tensor], got {type(output)}.")

    outputs = {}
    for name, value in output.items():
        if not isinstance(value, torch.Tensor):
            logger.warning(f"Skip non-tensor eval output '{name}' with type {type(value)}.")
            continue
        outputs[str(name)] = value
    return outputs


def _to_chw_image(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach().float().cpu()
    if tensor.ndim == 4 and tensor.shape[1] == 1:
        tensor = tensor[:, 0]
    if tensor.ndim == 4:
        tensor = tensor[:, 0]
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)
    return tensor[:3].clamp(0, 1)


def _safe_dir_name(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_")


def _save_eval_batch(
    batch: dict[str, Any],
    outputs: dict[str, torch.Tensor],
    evaluate_dir: Path,
    batch_start: int,
):
    prompt = batch.get("prompt", [])
    negative_prompt = batch.get("negative_prompt", [])
    image_name = batch.get("image_name", None)

    if isinstance(prompt, str):
        prompt = [prompt]
    if isinstance(negative_prompt, str):
        negative_prompt = [negative_prompt]

    first_output = next(iter(outputs.values()))
    batch_size = first_output.shape[0]
    output_names = list(outputs.keys())

    for name in output_names:
        (evaluate_dir / _safe_dir_name(name)).mkdir(exist_ok=True, parents=True)

    for batch_idx in range(batch_size):
        save_name = f"eval_{batch_start + batch_idx:06d}"
        if image_name is not None:
            save_name = image_name[batch_idx] if isinstance(image_name, (tuple, list)) else image_name

        for name, tensor in outputs.items():
            save_tensor = _to_chw_image(tensor[batch_idx])
            save_image(save_tensor, evaluate_dir / _safe_dir_name(name) / f"{save_name}.png")

        prompt_value = prompt[batch_idx] if batch_idx < len(prompt) else ""
        neg_prompt_value = negative_prompt[batch_idx] if batch_idx < len(negative_prompt) else ""
        with open(evaluate_dir / "prompt.jsonl", "a") as f:
            f.write(
                json.dumps(
                    {
                        "image_name": save_name,
                        "prompt": prompt_value,
                        "negative_prompt": neg_prompt_value,
                        "outputs": output_names,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


@hydra.main(config_path="configs", config_name="evaluation", version_base="v1.2")
def evaluate(cfgs: OmegaConf):
    cfg_contents = "\n" + " Configs ".center(50, "=")
    cfg_contents += "\n" + OmegaConf.to_yaml(cfgs)
    cfg_contents += "\n" + "=" * 50
    logger.info(cfg_contents)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    seed = int(_select(cfgs, "base_seed", 42))
    generator = torch.Generator(device).manual_seed(seed)
    _seed_everything(seed, device)

    evaluate_dir = Path(cfgs.project.evaluation_dir)
    evaluate_dir.mkdir(exist_ok=True, parents=True)

    pipe: BasePipeline = instantiate(cfgs.pipe_configs, device=device, generator=generator, dtype=dtype)
    pipe.transformer.requires_grad_(False)
    _load_lora_adapter(pipe, cfgs)
    pipe.transformer.requires_grad_(False)

    evalset_configs = _select(cfgs, "eval_data_configs", None)
    if evalset_configs is None:
        evalset_configs = cfgs.evalset
    dataset = instantiate(evalset_configs)
    batch_size = int(_select(cfgs, "batch_size_per_process", 1))
    num_workers = int(_select(cfgs, "data_loader_workers", 0))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

    if not hasattr(pipe, "eval_step"):
        raise AttributeError(f"{type(pipe).__name__} does not implement eval_step.")

    num_inference_steps = int(_select(cfgs, "num_inference_steps", 50))
    cfg_scale = float(_select(cfgs, "cfg_scale", 1.0))
    mask_cfg_scale = float(_select(cfgs, "mask_cfg_scale", 1.0))

    batch_start = 0
    for step, batch in tqdm(enumerate(dataloader), desc="Eval", total=len(dataloader)):
        if cfgs.eval_with_position_prompt:
            batch["prompt"] = [p for p in batch["edit_instruction"]]
        eval_kwargs = {}
        if hasattr(pipe, "mask_cfg_null_type"):
            eval_kwargs["mask_cfg_scale"] = mask_cfg_scale
        output: dict[str, torch.Tensor] = pipe.eval_step(
            batch,
            num_inference_steps=num_inference_steps,
            cfg_scale=cfg_scale,
            **eval_kwargs,
        )
        outputs = _normalize_outputs(output)
        if not outputs:
            logger.warning(f"Eval [{step + 1}/{len(dataloader)}] returned no tensor outputs.")
            continue

        _save_eval_batch(batch, outputs, evaluate_dir, batch_start)
        batch_start += next(iter(outputs.values())).shape[0]
        logger.info(f"Eval [{step + 1}/{len(dataloader)}] saved.")

    logger.info(f"Evaluation finished, saved to {evaluate_dir}.")


if __name__ == "__main__":
    evaluate()
