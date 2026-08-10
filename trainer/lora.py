import json
import torch
import dataclasses

from pathlib import Path
from typing import Literal
from omegaconf import OmegaConf
from loguru import logger
from safetensors.torch import save_file
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict

from diffusers.loaders.lora_base import LORA_ADAPTER_METADATA_KEY, LORA_WEIGHT_NAME_SAFE

from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
from trainer.base_trainer import BaseTrainer
from trainer.parallel.utils import wait_for_everyone
from trainer.prompt_sampler.prompt_sampler import PromptSampler


@dataclasses.dataclass
class LoraTrainer(BaseTrainer):

    lora_configs: OmegaConf | None = None
    adapter_state_dict_dir: str = "lora_adapter"

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

    def preprocess_eval_batch(self, batch, step: int):
        r"""LoraTrainer will use prompt with position cues to evlauate."""
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

    # Mirrored from pipeline.cfg_type by the trainer config interpolation.
    cfg_type: Literal["progressive", "condition_weighted"] = "condition_weighted"

    mask_cfg_dropout: float = 0.0
    mask_cfg_scale: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        if self.cfg_type not in {"progressive", "condition_weighted"}:
            raise ValueError(f"Unsupported cfg_type: {self.cfg_type}.")
        if self.text_cfg_dropout + self.mask_cfg_dropout > 1.0:
            raise ValueError("text_cfg_dropout + mask_cfg_dropout must be <= 1 for mutually exclusive CFG dropout.")
        self.prompt_sampler = PromptSampler(**self.prompt_sampler_cfgs)

    def preprocess_train_batch(self, batch, step: int):
        # Sample prompt based on the given probability
        runtime_prompt = self.prompt_sampler.sample_batch(
            tuple(zip(batch["edit_instruction"], batch["prompt"])),
            step,
        )

        # Handle the CFG dropout
        # Text dropout always drops only the prompt. Mask dropout drops the
        # VLM/DiT mask and additionally drops the prompt for progressive CFG.
        #
        # The CFG includes two types
        # Type I: `progressive`
        #   v(M,P) <- v(null,null) + s_M (v(M,null) - v(null,null)) + s_T (v(M,P) - v(M,null))
        # Type II: `condition_weighted`
        #   v_T <- v(M,P) + (s_T - 1) (v(M,P) - v(M,null))
        #   v_M <- v(M,P) + (s_M - 1) (v(M,P) - v(null,P))
        #   v(M,P) <- 0.5 * (rescale(v_T) + rescale(v_M))
        dropout_sample = self.rng.random()
        if dropout_sample < self.text_cfg_dropout:
            batch["prompt"] = ["" for _ in runtime_prompt]
            batch["mask_cfg_dropped"] = False
        elif dropout_sample < self.text_cfg_dropout + self.mask_cfg_dropout:
            if self.cfg_type == "progressive":
                batch["prompt"] = ["" for _ in runtime_prompt]
            else:
                batch["prompt"] = runtime_prompt
            batch["mask_cfg_dropped"] = True
        else:
            batch["prompt"] = runtime_prompt
            batch["mask_cfg_dropped"] = False
        return batch

    def preprocess_eval_batch(self, batch, step: int):
        return batch

    @torch.inference_mode()
    def evaluate(self, global_step: int):
        r"""Method for inference during training"""

        if self.eval_loader is None:
            return
        if not (global_step == 1 or (global_step % self.eval_steps == 0) or global_step == self.max_training_steps):
            return

        save_dir = Path(self.evaluation_dir) / f"step-{global_step}"
        metadata_dir = save_dir / "metadata"
        if self.is_main_process:
            save_dir.mkdir(exist_ok=True, parents=True)
            metadata_dir.mkdir(exist_ok=True, parents=True)
            logger.info(f"Evaluate start, save to {save_dir}.")
        wait_for_everyone()

        rank_metadata_path = metadata_dir / f"rank-{self.global_rank:05d}.jsonl"
        with open(rank_metadata_path, "w", encoding="utf-8") as metadata_file:
            for step, batch in enumerate(self.eval_loader):
                batch = self.preprocess_eval_batch(batch, global_step)

                # The pipeline must support text CFG and mask CFG
                output: dict[str, torch.Tensor] = self.pipe.eval_step(
                    batch=batch,
                    num_inference_steps=self.num_inference_steps,
                    text_cfg_scale=self.text_cfg_scale,
                    mask_cfg_scale=self.mask_cfg_scale,
                )
                self._save_eval_batch(batch, output, save_dir, step, metadata_file)

        wait_for_everyone()
        self._merge_eval_metadata(save_dir, metadata_dir)

        if self.is_main_process:
            logger.info(f"Evaluation finished, saved to {save_dir}.")
        wait_for_everyone()
