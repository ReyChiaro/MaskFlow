import dataclasses

from loguru import logger
from peft import LoraConfig
from omegaconf import OmegaConf

from trainer.base_trainer import BaseTrainer
from trainer.prompt_sampler.prompt_sampler import PromptSampler


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
            print("cfg drop")
            batch["prompt"] = ["" if self.rng.random() < cfg_dropout else p for p in runtime_prompt]
        return batch

    def preprocess_eval_batch(self, batch, step: int, cfg_dropout: float | None = None):
        return super().preprocess_eval_batch(batch, step, cfg_dropout)
