import dataclasses
from functools import partial
from pathlib import Path

import torch
from loguru import logger
from omegaconf import OmegaConf

from pipelines.cfg import training_probabilities
from trainer.base_trainer import BaseTrainer
from trainer.lora_utils import add_trainable_lora, save_lora_adapter
from trainer.parallel.utils import wait_for_everyone
from trainer.prompt_sampler.prompt_sampler import PromptSampler


@dataclasses.dataclass
class LoraTrainer(BaseTrainer):
    lora_configs: OmegaConf | None = None
    adapter_state_dict_dir: str = "lora_adapter"

    def _init_trainable(self) -> None:
        """Insert meta adapters and replay their initialization on rank 0 CPU at load time."""
        self.pipe.configure_model(
            self.pipe.transformer,
            partial(add_trainable_lora, cfgs=self.lora_configs, dtype=self._train_dtype),
        )

    def save_lora_adapter_checkpoint(self, checkpoint_dir: Path):
        adapter_dir = checkpoint_dir / self.adapter_state_dict_dir
        transformer = self.unwrap_model(self.pipe.transformer)
        save_lora_adapter(
            transformer,
            self.lora_configs,
            adapter_dir,
            self.is_main_process,
        )

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

    # CFG
    mask_cfg_dropout: float = 0.0
    cfg_branch_probabilities: dict[str, float] | None = None
    mask_cfg_scale: float = 1.0
    interaction_cfg_scale: float | None = None

    def __post_init__(self):
        self.cfg_branch_probabilities = training_probabilities(
            self.cfg_branch_probabilities, self.text_cfg_dropout, self.mask_cfg_dropout
        )
        super().__post_init__()
        self.prompt_sampler = PromptSampler(**self.prompt_sampler_cfgs)

    def preprocess_train_batch(self, batch, step: int):
        # Sample prompt based on the given probability
        runtime_prompt = self.prompt_sampler.sample_batch(
            tuple(zip(batch["edit_instruction"], batch["prompt"])),
            step,
        )
        batch["prompt"] = runtime_prompt

        # A batch shares one condition layout. Keep the original mask and prompt;
        # the pipeline applies this selection only to model-visible conditions.
        draw = self.rng.random() * sum(self.cfg_branch_probabilities.values())
        cumulative = 0.0
        for branch, probability in self.cfg_branch_probabilities.items():
            cumulative += probability
            if draw < cumulative:
                break
        batch["cfg_branch"] = branch
        return batch

    def preprocess_eval_batch(self, batch, step: int):
        r"""
        `prompt`: Image editing prompt that *without* position cues.
        `edit_instruction`: Image editing prompt that *with* position cues.
        """
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
                output: dict[str, torch.Tensor] = self.pipe.eval_step(
                    batch=batch,
                    num_inference_steps=self.num_inference_steps,
                    text_cfg_scale=self.text_cfg_scale,
                    mask_cfg_scale=self.mask_cfg_scale,
                    interaction_cfg_scale=self.interaction_cfg_scale,
                    seed=self.eval_seed,
                )
                self._save_eval_batch(batch, output, save_dir, step, metadata_file)

        wait_for_everyone()
        self._merge_eval_metadata(save_dir, metadata_dir)

        if self.is_main_process:
            logger.info(f"Evaluation finished, saved to {save_dir}.")
        wait_for_everyone()
