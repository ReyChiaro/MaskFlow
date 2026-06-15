import json
import random
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
import torch.nn.functional as F

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


def _normalize_outputs(output: Any) -> tuple[list[str], list[torch.Tensor]]:
    if isinstance(output, torch.Tensor):
        return ["output"], [output]

    if isinstance(output, dict):
        names, tensors = [], []
        for name, value in output.items():
            if isinstance(value, torch.Tensor):
                names.append(str(name))
                tensors.append(value)
        return names, tensors

    if isinstance(output, (tuple, list)):
        tensors = [value for value in output if isinstance(value, torch.Tensor)]
        names = [f"output_{idx}" for idx in range(len(tensors))]
        if tensors:
            names[-1] = "output"
        return names, tensors

    raise TypeError(f"Unsupported eval_step output type: {type(output)}.")


def _batched_conditions(conditions: Any) -> list[torch.Tensor]:
    if conditions is None:
        return []
    if isinstance(conditions, dict):
        return [value for value in conditions.values() if isinstance(value, torch.Tensor)]
    if isinstance(conditions, torch.Tensor):
        return [conditions]
    if isinstance(conditions, (tuple, list)):
        return [value for value in conditions if isinstance(value, torch.Tensor)]
    return []


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


def _pad_to_height(tensor: torch.Tensor, height: int) -> torch.Tensor:
    pad_bottom = max(height - tensor.shape[-2], 0)
    return F.pad(tensor, pad=(0, 0, 0, pad_bottom), mode="constant", value=0)


def _save_eval_batch(
    batch: dict[str, Any],
    output_names: list[str],
    outputs: list[torch.Tensor],
    evaluate_dir: Path,
    batch_start: int,
):
    prompt = batch.get("prompt", [])
    negative_prompt = batch.get("negative_prompt", [])
    image_name = batch.get("image_name", None)
    target = batch.get("target", None)
    conditions = _batched_conditions(batch.get("conditions", None))

    if isinstance(prompt, str):
        prompt = [prompt]
    if isinstance(negative_prompt, str):
        negative_prompt = [negative_prompt]

    batch_size = outputs[-1].shape[0] if outputs else len(prompt)

    for batch_idx in range(batch_size):
        save_name = f"eval_{batch_start + batch_idx:06d}"
        if image_name is not None:
            save_name = image_name[batch_idx] if isinstance(image_name, (tuple, list)) else image_name

        final_output = _to_chw_image(outputs[-1][batch_idx])
        save_image(final_output, evaluate_dir / f"{save_name}.jpg")

        concat_tensors = []
        concat_tensors.extend(_to_chw_image(condition[batch_idx]) for condition in conditions)
        if isinstance(target, torch.Tensor):
            concat_tensors.append(_to_chw_image(target[batch_idx]))
        concat_tensors.extend(_to_chw_image(output[batch_idx]) for output in outputs)

        if concat_tensors:
            max_h = max(tensor.shape[-2] for tensor in concat_tensors)
            concat_tensors = [_pad_to_height(tensor, max_h) for tensor in concat_tensors]
            save_image(torch.cat(concat_tensors, dim=-1), evaluate_dir / f"c_{save_name}.jpg")

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

    batch_start = 0
    for step, batch in tqdm(enumerate(dataloader), desc="Eval", total=len(dataloader)):
        output = pipe.eval_step(
            batch,
            num_inference_steps=num_inference_steps,
            cfg_scale=cfg_scale,
        )
        output_names, outputs = _normalize_outputs(output)
        if not outputs:
            logger.warning(f"Eval [{step + 1}/{len(dataloader)}] returned no tensor outputs.")
            continue

        _save_eval_batch(batch, output_names, outputs, evaluate_dir, batch_start)
        batch_start += outputs[-1].shape[0]
        logger.info(f"Eval [{step + 1}/{len(dataloader)}] saved.")

    logger.info(f"Evaluation finished, saved to {evaluate_dir}.")


if __name__ == "__main__":
    evaluate()
