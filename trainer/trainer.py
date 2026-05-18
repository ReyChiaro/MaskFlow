import os
import copy
import math
import torch
import random
import numpy as np
import torch.nn.functional as F

from diffusers.optimization import get_scheduler
from diffusers.utils.torch_utils import is_compiled_module

from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from accelerate import Accelerator

from typing import Any
from omegaconf import OmegaConf
from hydra.utils import instantiate
from loguru import logger
from safetensors.torch import save_file, load_file

from data_module.dataset import SchemaDataset
from pipelines.pipeline_manager import DiTPipelineManager
from utils.summary import get_summary_table


class Trainer:

    def __init__(
        self,
        data_loader_workers: int,
        batch_size_per_process: int,
        training_steps_per_process: int,
        save_steps_per_process: int,
        eval_steps_per_process: int,
        pipeline_configs: OmegaConf,
        optimizer_configs: OmegaConf,
        train_data_configs: OmegaConf,
        output_dir: str = "outputs",
        project_name: str = "outputs/project-train",
        backup_dir: str | None = None,
        checkpoint_dir: str = "outputs/checkpoints",
        evaluation_dir: str = "outputs/evaluations",
        log_dir: str = "logs",
        random_seed: int = 0,
        num_epochs: int | None = None,
        num_warmup_steps_per_process: int | None = None,
        max_grad_norm: float = 1.0,
        num_inference_steps: int = 50,
        enable_gradient_checkpoint: bool = True,
        eval_data_configs: OmegaConf | None = None,
        adapter_configs: OmegaConf | None = None,
        lr_scheduler_configs: OmegaConf | None = None,
        enable_async_inference: bool = False,
        cudnn_deterministic: bool = False,
        cudnn_benchmark: bool = True,
    ):
        self.pipeline_configs = pipeline_configs
        self.optimizer_configs = optimizer_configs
        self.train_data_configs = train_data_configs
        self.eval_data_configs = eval_data_configs
        self.adapter_configs = adapter_configs
        self.lr_scheduler_configs = lr_scheduler_configs

        self.output_dir = output_dir
        self.backup_dir = backup_dir
        self.project_name = project_name
        self.checkpoint_dir = checkpoint_dir
        self.evaluation_dir = evaluation_dir
        self.log_dir = log_dir

        self.data_loader_workers = data_loader_workers
        self.batch_size_per_process = batch_size_per_process
        self.training_steps_per_process = training_steps_per_process
        self.save_steps_per_process = save_steps_per_process
        self.eval_steps_per_process = eval_steps_per_process
        self.num_warmup_steps_per_process = num_warmup_steps_per_process
        self.random_seed = random_seed
        self.num_epochs = num_epochs
        self.max_grad_norm = max_grad_norm
        self.enable_gradient_checkpoint = enable_gradient_checkpoint
        # TODO
        self.enable_async_inference = enable_async_inference

        self.cudnn_benchmark = cudnn_benchmark
        self.cudnn_deterministic = cudnn_deterministic

        self.num_inference_steps = num_inference_steps

        self.initialization_handlers = [
            self._init_context,
            self._init_project,
            self._init_pipeline,
            self._init_adapter,
            self._init_optimizer,
            self._init_data_loader,
        ]

        self._train_dtype = None
        self._eval_dtype = torch.bfloat16

    def _init_context(self, accelerator: Accelerator):
        if self._train_dtype is None:
            self._train_dtype = torch.float32
            if accelerator.mixed_precision == "bf16":
                self._train_dtype = torch.bfloat16
            elif accelerator.mixed_precision == "fp16":
                self._train_dtype = torch.float16
        self._eval_dtype = torch.bfloat16

        self.rank = accelerator.process_index
        self.random_seed = self.random_seed + self.rank
        self.world_size = accelerator.num_processes
        self.device = torch.device(f"cuda:{self.rank}")
        self.generator = torch.Generator(self.device).manual_seed(self.random_seed)
        self.is_main_process = accelerator.is_main_process
        logger.info(f"Context initialized.")

    def _init_project(self, accelerator: Accelerator):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.evaluation_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # TODO: Add project backup is required.
        if self.backup_dir is not None:
            os.makedirs(self.backup_dir, exist_ok=True)
        logger.info(f"Project initialized.")

    def _init_pipeline(self, accelerator: Accelerator):
        self.pipeline: DiTPipelineManager = instantiate(self.pipeline_configs)
        self.pipeline.init_components(
            self.pipeline.pretrained_model_name_or_path,
            torch_dtype=self._train_dtype,
            device=accelerator.device,
        )
        logger.info(f"Pipeline initialized.")

    def _init_adapter(self, accelerator: Accelerator):
        r"""Add adapter (e.g. LoRA, Control Net) to the loaded pipeline."""
        pass

    def _init_optimizer(self, accelerator: Accelerator):
        self.optimizer: Optimizer = instantiate(self.optimizer_configs, params=self.pipeline.trainable_parameters)
        logger.info(f"Optimizer initialized.")

    def _init_data_loader(self, accelerator: Accelerator):
        # TODO: Maybe initialize worker_init_fn to make sure the random seeds differ between workers
        trainset: SchemaDataset = instantiate(self.train_data_configs)
        self.train_loader = DataLoader(
            dataset=trainset,
            batch_size=self.batch_size_per_process,
            num_workers=self.data_loader_workers,
        )
        logger.info(f"Train Dataloader initialized.")
        self.eval_loader = None
        if self.eval_data_configs is not None:
            evalset: SchemaDataset = instantiate(self.eval_data_configs)
            self.eval_loader = DataLoader(dataset=evalset, batch_size=1, num_workers=self.data_loader_workers)
            logger.info(f"Eval Dataloader initialized.")

    def init_everything(self, accelerator: Accelerator):
        # The seed has been rank-shifted
        seed = self.random_seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = self.cudnn_deterministic
        torch.backends.cudnn.benchmark = self.cudnn_benchmark

        for handler in self.initialization_handlers:
            handler(accelerator)

        self.pipeline.transformer, self.optimizer, self.train_loader = accelerator.prepare(
            self.pipeline.transformer, self.optimizer, self.train_loader
        )

        if self.enable_gradient_checkpoint:
            self.pipeline.transformer.enable_gradient_checkpointing()

        self.total_train_batch_size = (
            self.batch_size_per_process * accelerator.gradient_accumulation_steps * accelerator.num_processes
        )
        self.num_update_per_epoch = math.ceil(len(self.train_loader) / accelerator.gradient_accumulation_steps)
        if self.num_epochs is None:
            self.num_epochs = math.ceil(self.training_steps_per_process / self.num_update_per_epoch)

        if getattr(self, "lr_scheduler", None) is not None:
            self.lr_scheduler = get_scheduler(
                name=self.lr_scheduler_configs.name,
                optimizer=self.optimizer,
                step_rules=self.lr_scheduler_configs.step_rules,
                num_warmup_steps=self.lr_scheduler_configs.num_warmup_steps * accelerator.num_processes,
                num_training_steps=self.training_steps_per_process * accelerator.num_processes,
                num_cycles=self.lr_scheduler_configs.num_cycles,
                power=self.lr_scheduler_configs.power,
                last_epoch=self.lr_scheduler_configs.last_epoch,
            )
            self.lr_scheduler = accelerator.prepare(self.lr_scheduler)

    @staticmethod
    def unwrap_model(accelerator: Accelerator, model: torch.nn.Module):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def save_checkpoints(self, accelerator: Accelerator, global_step: int):
        transformer = self.unwrap_model(accelerator, self.pipeline.transformer)
        states_to_save = {}
        for n, p in transformer.named_parameters():
            if p.requires_grad:
                states_to_save[n] = p
        save_path = os.path.join(self.checkpoint_dir, f"checkpoints-{global_step}.safetensors")
        save_file(states_to_save, save_path)
        logger.info(f"Checkpoints saved to {save_path}.")

    def load_checkpoints(self, checkpoint_path: str, **kwargs):
        states = load_file(checkpoint_path)
        self.pipeline.transformer.load_state_dict(states)
        logger.info(f"Load checkpoints from {checkpoint_path}.")

    def save_image(self, output: torch.Tensor, save_path: str, others: list[torch.Tensor] | None = None):
        processed_tensors = []
        output = output.squeeze(2)
        others = [c.squeeze(2) for c in others]
        if others is not None:
            for t in others:
                if t.ndim == 4:
                    processed_tensors.append(t.squeeze(0))
                elif t.ndim == 3:
                    processed_tensors.append(t)
                else:
                    raise ValueError(f"Unsupported Tensor dimension: {t.shape}, only 3 or 4 is supported.")
        processed_tensors.append(output.squeeze(0) if output.ndim == 4 else output)
        max_height = max(t.shape[1] for t in processed_tensors)
        padded_tensors = []
        for t in processed_tensors:
            # C, H, W = t.shape
            current_height = t.shape[1]
            pad_bottom = max_height - current_height
            if pad_bottom > 0:
                t_padded = F.pad(t, (0, 0, 0, pad_bottom), mode="constant", value=0)
                padded_tensors.append(t_padded)
            else:
                padded_tensors.append(t)
        result_tensor = torch.cat(padded_tensors, dim=-1)
        save_image(result_tensor, save_path)
        return result_tensor

    @torch.inference_mode()
    def evaluation(self, global_step: int):
        if self.eval_loader is None:
            return
        self.pipeline.transformer.eval()
        save_dir = os.path.join(self.evaluation_dir, f"step-{global_step}")
        os.makedirs(save_dir, exist_ok=True)

        logger.info("\n" + " Start Inference ".center(50, "="))
        for i, sample in enumerate(self.eval_loader):
            raw_sample = copy.deepcopy(sample)
            raw_sample["conditions"] = [c.to(self.device, dtype=self._eval_dtype) for c in raw_sample["conditions"]]
            raw_sample["target"] = raw_sample["target"].to(self.device, dtype=self._eval_dtype)
            output = self.pipeline.eval_step(
                sample, self.num_inference_steps, self.generator, self.device, self._eval_dtype
            )
            image_name = raw_sample.get("image_name", f"eval-{i}.jpg")
            save_path = os.path.join(save_dir, image_name)
            self.save_image(output, save_path, raw_sample.get("conditions", []) + [raw_sample["target"]])
        logger.info(f"Inference Ended. Result saved to {save_dir}")
        self.pipeline.transformer.train()

    def train(self, accelerator: Accelerator):
        self.init_everything(accelerator)

        pipe_summary = self.pipeline.summary
        summary_table = get_summary_table(pipe_summary)
        logger.info(summary_table)

        sep = "=" * 25
        log_title = f"{sep} Start Training {sep}"
        info = (
            f"\n{log_title}"
            f"\n  World size               : {self.world_size}"
            f"\n  Random seed              : {self.random_seed} (Differ between ranks)"
            f"\n  Mixed precision          : {accelerator.mixed_precision}"
            f"\n  Num training batches     : {len(self.train_loader)}"
            f"\n  Batch size per device    : {self.batch_size_per_process}"
            f"\n  Total batch size         : {self.total_train_batch_size}"
            f"\n  Gradient accum steps     : {accelerator.gradient_accumulation_steps}"
            f"\n  Num epochs               : {self.num_epochs}"
            f"\n  Update steps per epoch   : {self.num_update_per_epoch}"
            f"\n  Max training steps       : {self.training_steps_per_process}"
            f"\n{'=' * len(log_title)}"
        )
        logger.info(info)

        global_step = 0
        metrics = {"loss": 0}
        align = len(f"{self.training_steps_per_process}")
        self.pipeline.transformer.train()
        for epoch in range(self.num_epochs):
            for batch in self.train_loader:
                with accelerator.accumulate(self.pipeline.transformer):
                    with accelerator.autocast():
                        loss: torch.Tensor = self.pipeline.forward_step(
                            batch, self.generator, self.device, self._train_dtype
                        )
                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(self.pipeline.trainable_parameters, self.max_grad_norm)
                    self.optimizer.step()
                    if getattr(self, "lr_scheduler", None) is not None:
                        self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                    metrics["loss"] += loss.item()

                if accelerator.sync_gradients:
                    global_step += 1

                    metrics["loss"] = metrics["loss"] / accelerator.gradient_accumulation_steps

                    if (
                        self.is_main_process
                        and (global_step == 1)
                        or (global_step == self.training_steps_per_process)
                        or (global_step % self.save_steps_per_process == 0)
                    ):
                        self.save_checkpoints(accelerator, global_step)

                    if (
                        self.is_main_process
                        and self.eval_loader is not None
                        and (global_step == 1)
                        or (global_step == self.training_steps_per_process)
                        or (global_step % self.eval_steps_per_process == 0)
                    ):
                        self.evaluation(global_step)

                    loginfo = (
                        f"[Step {global_step:{align}d}/{self.training_steps_per_process}] "
                        + f"Loss {metrics['loss']:.6f}"
                    )
                    logger.info(loginfo)

                    # Reset recorders
                    metrics["loss"] = 0

                if global_step > self.training_steps_per_process:
                    break
            if global_step > self.training_steps_per_process:
                break
        logger.info(f"End training, waiting for everyone.")
        accelerator.wait_for_everyone()
        if self.is_main_process and self.eval_loader is not None:
            self.evaluation(global_step)
        accelerator.end_training()

    @torch.inference_mode()
    def generate(self):
        r"""Inference pipeline"""
        pass
