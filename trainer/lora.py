import dataclasses
import json

from loguru import logger
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from omegaconf import OmegaConf
from pathlib import Path
from safetensors.torch import save_file
from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions

from diffusers.loaders.lora_base import LORA_ADAPTER_METADATA_KEY, LORA_WEIGHT_NAME_SAFE

from trainer.base_trainer import BaseTrainer
from trainer.prompt_sampler.prompt_sampler import PromptSampler
from trainer.parallel.utils import wait_for_everyone


@dataclasses.dataclass
class LoraTrainer(BaseTrainer):

    lora_configs: OmegaConf | None = None
    adapter_state_dict_dir: str = "adapter"

    def _init_trainable(self):
        # Adapter configs is list of target_modules
        cfgs = self.lora_configs

        lora_configs = LoraConfig(
            r=cfgs.r,
            lora_alpha=cfgs.lora_alpha,
            lora_dropout=cfgs.lora_dropout,
            bias="none",
            target_modules=list(cfgs.target_modules),
        )

        self.pipe.transformer.requires_grad_(False)
        self.pipe.transformer.add_adapter(lora_configs, adapter_name=cfgs.adapter_name)
        self.pipe.transformer.set_adapter(cfgs.adapter_name)

        for n, p in self.pipe.transformer.named_parameters():
            if cfgs.adapter_name in n and "lora_" in n:
                p.data = p.to(self.device, dtype=self._train_dtype).data
                if p.grad is not None:
                    p.grad = p.grad.to(self.device, dtype=self._train_dtype)
                p.requires_grad_(True)

        logger.info(f"Add LoRA adapter to transformer.")

    def save_lora_adapter_checkpoint(self, checkpoint_dir: Path):
        adapter_name = self.lora_configs.adapter_name
        adapter_dir = checkpoint_dir / self.adapter_state_dict_dir

        transformer = self.unwrap_model(self.pipe.transformer)
        full_state_dict = get_model_state_dict(
            transformer,
            options=StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,
                ignore_frozen_params=True,
            ),
        )

        if not self.is_main_process:
            return

        adapter_dir.mkdir(exist_ok=True, parents=True)
        lora_state_dict = get_peft_model_state_dict(
            transformer,
            state_dict=full_state_dict,
            adapter_name=adapter_name,
        )
        if not lora_state_dict:
            raise RuntimeError(f"No LoRA weights found for adapter '{adapter_name}'.")

        metadata = {"format": "pt"}
        lora_adapter_metadata = transformer.peft_config[adapter_name].to_dict()
        for key, value in lora_adapter_metadata.items():
            if isinstance(value, set):
                lora_adapter_metadata[key] = list(value)
        metadata[LORA_ADAPTER_METADATA_KEY] = json.dumps(lora_adapter_metadata, indent=2, sort_keys=True)

        save_path = adapter_dir / LORA_WEIGHT_NAME_SAFE
        save_file(lora_state_dict, save_path, metadata=metadata)
        logger.info(f"LoRA safetensors saved to {save_path}.")

    def preprocess_train_batch(self, batch, step: int, cfg_dropout: float | None = None):
        if cfg_dropout is not None:
            if "edit_instruction" in batch:
                batch["prompt"] = ["" if self.rng.random() < cfg_dropout else p for p in batch["edit_instruction"]]
            else:
                batch["prompt"] = ["" if self.rng.random() < cfg_dropout else p for p in batch["prompt"]]
        return batch

    def preprocess_eval_batch(self, batch, step: int, cfg_dropout: float | None = None):
        if "edit_instruction" in batch:
            batch["prompt"] = [p for p in batch["edit_instruction"]]
        return batch

    def on_train_end(self, global_step: int):
        if global_step <= 0:
            logger.warning("Skip LoRA safetensors export because no training step was completed.")
            return

        checkpoint_dir = Path(self.checkpoint_dir) / f"step-{global_step}"
        wait_for_everyone()
        self.save_lora_adapter_checkpoint(checkpoint_dir)
        wait_for_everyone()


@dataclasses.dataclass
class MaskFlowTrainer(LoraTrainer):

    prompt_sampler_cfgs: OmegaConf = None

    def __post_init__(self):
        super().__post_init__()
        self.prompt_sampler = PromptSampler(**self.prompt_sampler_cfgs)

    def preprocess_train_batch(self, batch, step: int, cfg_dropout: float | None = None):
        runtime_prompt = self.prompt_sampler.sample_batch(
            tuple(zip(batch["edit_instruction"], batch["prompt"])),
            step,
        )
        batch["prompt"] = runtime_prompt
        if cfg_dropout is not None:
            batch["prompt"] = ["" if self.rng.random() < cfg_dropout else p for p in runtime_prompt]
        return batch

    def preprocess_eval_batch(self, batch, step: int, cfg_dropout: float | None = None):
        return super().preprocess_eval_batch(batch, step, cfg_dropout)
