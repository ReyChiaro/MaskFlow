import dataclasses

from loguru import logger
from peft import LoraConfig
from omegaconf import OmegaConf

from trainer import BaseTrainer


@dataclasses.dataclass
class LoraTrainer(BaseTrainer):

    lora_configs: OmegaConf | None = None

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
                p = p.to(self.device, dtype=self._train_dtype)
                p.requires_grad_(True)

        logger.info(f"Add LoRA adapter to transformer.")

