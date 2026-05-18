import os
import torch

from accelerate import Accelerator

from diffusers.loaders.peft import PeftAdapterMixin

from loguru import logger
from peft import LoraConfig
from typing import Union
from safetensors.torch import load_file

from pipelines.pipeline_manager import DenoiserInputs, DenoiserOutputs
from trainer.trainer import Trainer
from utils.summary import gpu_utilization


class LoRATrainer(Trainer):

    def _init_adapter(self, accelerator: Accelerator):
        # Adapter configs is list of target_modules
        cfgs = self.adapter_configs

        lora_configs = LoraConfig(
            r=cfgs.r,
            lora_alpha=cfgs.lora_alpha,
            lora_dropout=cfgs.lora_dropout,
            bias="none",
            target_modules=list(cfgs.target_modules),
        )

        self.pipeline.transformer.requires_grad_(False)
        self.pipeline.transformer.add_adapter(lora_configs, adapter_name=cfgs.adapter_name)
        self.pipeline.transformer.set_adapter(cfgs.adapter_name)

        for n, p in self.pipeline.transformer.named_parameters():
            if cfgs.adapter_name in n and "lora_" in n:
                p = p.to(self.device, dtype=self._train_dtype)
                p.requires_grad_(True)

        logger.info(f"Add LoRA adapter to transformer.")

    def save_checkpoints(self, accelerator: Accelerator, global_step: int):
        cfgs = self.adapter_configs
        align_w = len(f"{self.training_steps_per_process}")
        checkpoint_dir = os.path.join(self.checkpoint_dir, f"step-{global_step:0{align_w}d}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        unwrap_m: PeftAdapterMixin = self.unwrap_model(accelerator, self.pipeline.transformer)
        unwrap_m.save_lora_adapter(
            checkpoint_dir,
            adapter_name=cfgs.adapter_name,
            upcast_before_saving=False,
            safe_serialization=True,
            weight_name=f"{cfgs.adapter_name}.safetensors",
        )

        # Upcast is used in save_lora_adapter, we must convert it back
        unwrap_m.to(dtype=self._train_dtype)
        torch.cuda.empty_cache()
        logger.info(f"Adapter saved to {checkpoint_dir}")

    def load_checkpoints(self, checkpoint_path: str, adapter_name: str = "default"):
        state_dict = load_file(checkpoint_path)
        self.pipeline.transformer.load_lora_adapter(
            state_dict,
            prefix=None,
            adapter_name=adapter_name,
            use_safetensors=True,
        )
        self.pipeline.transformer.set_adapter(adapter_name)

    @torch.inference_mode()
    def generate(self):
        return super().generate()
