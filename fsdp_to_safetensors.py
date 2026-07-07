import peft
import hydra
import torch.distributed.checkpoint as DCP

from hydra.utils import instantiate
from omegaconf import OmegaConf
from typing import Any
from pathlib import Path
from loguru import logger
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
)

from pipelines.base_pipeline import BasePipeline
from pipelines.qwenimage.qwenimage_mask_flow import QwenImageMaskFlow


def load_lora_adapter(
    pipe: BasePipeline,
    resume_from: str,
    adapter_name: str,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: list[str],
    safetensors_dir: str,
):
    r"""
    Load LoRA weights for evaluation.

    Args: cfgs can include following options
        +is_fsdp_checkpoint (bool)
        +lora_safetensors_dir (str): If a DCP/FSDP checkpoint is given,
            it will be converted into safetensors for convenient reuse.
    """
    if not resume_from or not Path(resume_from).exists():
        if resume_from:
            logger.warning(f"LoRA checkpoint not found: {resume_from}.")
        return

    lora_configs = peft.LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=list(target_modules),
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
    safetensors_dir = Path(safetensors_dir)

    safetensors_dir.mkdir(exist_ok=True, parents=True)
    pipe.transformer.save_lora_adapter(
        safetensors_dir,
        adapter_name=adapter_name,
        safe_serialization=True,
    )
    pipe.transformer.delete_adapters(adapter_name)
    logger.info(f"Converted DCP checkpoint {resume_from} to LoRA safetensors at {safetensors_dir}.")


@hydra.main(config_path="configs", config_name="inference", version_base="v1.2")
def main(cfgs: OmegaConf):
    pipe: QwenImageMaskFlow = instantiate(cfgs.pipeline)
    load_lora_adapter(
        pipe,
        resume_from=cfgs.resume_from,
        adapter_name=cfgs.adapter.adapter_name,
        lora_rank=cfgs.adapter.r,
        lora_alpha=cfgs.adapter.lora_alpha,
        lora_dropout=cfgs.adapter.lora_dropout,
        target_modules=cfgs.adapter.target_modules,
        safetensors_dir=cfgs.safetensors_dir,
    )


if __name__ == "__main__":
    main()
