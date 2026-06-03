import os
import torch
import dataclasses

from diffusers.loaders.peft import PeftAdapterMixin
from pathlib import Path
from safetensors.torch import save_file, load_file
from loguru import logger
from peft import LoraConfig
from omegaconf import OmegaConf

from trainer import BaseTrainer


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

    def save_checkpoints(self, global_step: int):
        r"""
        - Training states
        - Data sampler
        - Model: checkpoints of *trainable* parameters of trasnformer by default.
        """
        if not self.is_main_process:
            return
        if not (global_step == 1 or (global_step % self.save_steps == 0) or global_step == self.max_training_steps):
            return

        checkpoint_dir = Path(self.checkpoint_dir) / f"step-{global_step}"
        checkpoint_dir.mkdir(exist_ok=True, parents=True)

        # Train state
        train_path = Path(checkpoint_dir) / self.training_state_dict_file
        train_state = self.train_state_dict()
        torch.save(train_state, train_path)

        # Data sampler
        if self.train_sampler is not None:
            ds_path = Path(checkpoint_dir) / self.data_sampler_state_dict_file
            ds_state = self.train_sampler.state_dict()
            torch.save(ds_state, ds_path)

        adapter_name = self.lora_configs.adapter_name
        transformer: PeftAdapterMixin = self.unwrap_model(self.pipe.transformer)
        transformer.save_lora_adapter(checkpoint_dir / self.adapter_state_dict_dir, adapter_name)
        logger.info(f"Checkpoints saved to {checkpoint_dir}.")

    def load_checkpoints(self, checkpoint_path: str, **kwargs):
        # Data sampler
        checkpoint_dir = Path(checkpoint_path)
        if self.train_sampler is not None:
            ds_path = checkpoint_dir / self.data_sampler_state_dict_file
            ds_state = torch.load(ds_path)
            self.train_sampler.load_state_dict(ds_state)

        # Train state
        train_path = checkpoint_dir / self.training_state_dict_file
        train_state = torch.load(train_path)
        self.load_train_state_dict(train_state)

        # Pretrained adapter
        self.pipe.transformer.load_lora_adapter(checkpoint_dir / self.adapter_state_dict_dir)
        logger.info(f"Load checkpoints from {checkpoint_path}.")
