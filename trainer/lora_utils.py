import json

import torch
from diffusers.loaders.lora_base import LORA_ADAPTER_METADATA_KEY, LORA_WEIGHT_NAME_SAFE
from diffusers.loaders.peft import PeftAdapterMixin
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from safetensors.torch import save_file
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict


def add_trainable_lora(
    transformer: torch.nn.Module,
    cfgs: OmegaConf,
    dtype: torch.dtype,
) -> list[torch.nn.Parameter]:
    """Insert adapters on the model's current device (meta during construction)."""
    lora_config = LoraConfig(
        r=cfgs.r,
        lora_alpha=cfgs.lora_alpha,
        lora_dropout=cfgs.lora_dropout,
        bias="none",
        target_modules=cfgs.target_modules if isinstance(cfgs.target_modules, str) else list(cfgs.target_modules),
    )

    transformer.requires_grad_(False)
    transformer.add_adapter(lora_config, adapter_name=cfgs.adapter_name)
    transformer.set_adapter(cfgs.adapter_name)

    params = []
    for name, param in transformer.named_parameters():
        if cfgs.adapter_name in name and "lora_" in name:
            param.data = param.to(dtype=dtype).data
            if param.grad is not None:
                param.grad = param.grad.to(dtype=dtype)
            param.requires_grad_(True)
            params.append(param)

    logger.info(f"Added trainable LoRA adapter '{cfgs.adapter_name}'.")
    return params


def merge_lora(
    transformer: PeftAdapterMixin,
    lora_path: str,
    adapter_name: str,
    lora_scale: float = 1.0,
    weight_name: str | None = None,
) -> None:
    """Merge lora weights into the model."""
    load_kwargs = {"adapter_name": adapter_name, "prefix": None, "weight_name": weight_name}
    transformer.load_lora_adapter(lora_path, **load_kwargs)
    transformer.set_adapter(adapter_name)
    transformer.fuse_lora(
        lora_scale=lora_scale,
        safe_fusing=True,
        adapter_names=[adapter_name],
    )
    transformer.unload_lora()
    logger.info(f"Merged LoRA '{lora_path}' into transformer with scale {lora_scale}.")


def load_inference_loras(
    transformer: PeftAdapterMixin,
    checkpoint_cfgs: DictConfig,
    lora_scale: float = 1.0,
) -> None:
    """Apply the SFT LoRA and, when requested, activate its DMD residual LoRA."""
    sft_path = checkpoint_cfgs.get("sft_path")
    dmd_path = checkpoint_cfgs.get("dmd_path")

    if dmd_path and not sft_path:
        raise ValueError("checkpoint.sft_path is required when checkpoint.dmd_path is set.")

    if sft_path:
        merge_lora(
            transformer,
            sft_path,
            checkpoint_cfgs.sft_adapter_name,
            lora_scale=lora_scale,
            weight_name=checkpoint_cfgs.sft_weight_name,
        )

    if dmd_path:
        dmd_adapter_name = checkpoint_cfgs.get("dmd_adapter_name", "dmd")
        load_kwargs = {"prefix": None, "adapter_name": dmd_adapter_name}
        if checkpoint_cfgs.get("dmd_weight_name"):
            load_kwargs["weight_name"] = checkpoint_cfgs.dmd_weight_name
        transformer.load_lora_adapter(dmd_path, **load_kwargs)
        transformer.set_adapter(dmd_adapter_name)
        logger.info(f"Loaded DMD LoRA '{dmd_path}' as adapter '{dmd_adapter_name}'.")

    if not sft_path and not dmd_path:
        logger.warning("No MaskFlow LoRA checkpoint was configured; using the base transformer.")


def save_lora_adapter(
    transformer: torch.nn.Module,
    cfgs: OmegaConf,
    adapter_dir,
    is_main_process: bool,
):
    full_state_dict = get_model_state_dict(
        transformer,
        options=StateDictOptions(
            full_state_dict=True,
            cpu_offload=True,
            ignore_frozen_params=True,
        ),
    )

    if not is_main_process:
        return

    adapter_dir.mkdir(exist_ok=True, parents=True)
    lora_state_dict = get_peft_model_state_dict(
        transformer,
        state_dict=full_state_dict,
        adapter_name=cfgs.adapter_name,
    )
    if not lora_state_dict:
        raise RuntimeError(f"No LoRA weights found for adapter '{cfgs.adapter_name}'.")

    metadata = {"format": "pt"}
    lora_adapter_metadata = transformer.peft_config[cfgs.adapter_name].to_dict()
    for key, value in lora_adapter_metadata.items():
        if isinstance(value, set):
            lora_adapter_metadata[key] = list(value)
    metadata[LORA_ADAPTER_METADATA_KEY] = json.dumps(lora_adapter_metadata, indent=2, sort_keys=True)

    save_path = adapter_dir / LORA_WEIGHT_NAME_SAFE
    save_file(lora_state_dict, save_path, metadata=metadata)
    logger.info(f"LoRA safetensors saved to {save_path}.")
