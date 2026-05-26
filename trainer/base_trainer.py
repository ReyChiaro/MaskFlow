import os
import math
import random
import numpy as np

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.utils import save_image

from diffusers.optimization import get_scheduler
from diffusers.utils.torch_utils import is_compiled_module

from safetensors.torch import load_file, save_file
from dataclasses import dataclass
from typing import Literal
from omegaconf import OmegaConf
from hydra.utils import instantiate
from loguru import logger

from pipelines import BasePipeline
from data_module import SchemaDataset
from utils.summary import get_summary_table


class _BaseTrainer:

    def __init__(self):
        pass

    def _init_context(self):
        pass

    def _init_project(self):
        pass

    def _init_pipeline(self):
        pass

    def _init_trainable(self):
        r"""
        Initialize trainable parameters in the pipeline
        or add trainable adapters on it.
        """
        pass

    def _init_optimizer(self):
        pass

    def _init_lr_scheduler(self):
        pass

    def _init_data_loader(self):
        pass

    def init_everything(self):
        r"""
        Invoke the init handlers sequentially.
        """
        pass

    def train(self):
        r"""
        Train pipeline.
        """
        pass

    @torch.inference_mode()
    def evaluate(self, global_step: int):
        pass


@dataclass
class BaseTrainer(_BaseTrainer):

    # Modules
    pipe_configs: OmegaConf | None = None
    optimizer_configs: OmegaConf | None = None
    train_data_configs: OmegaConf | None = None
    eval_data_configs: OmegaConf | None = None
    lr_scheduler_configs: OmegaConf | None = None

    # Project
    base_seed: int = 42
    output_dir: str = "outputs"
    checkpoint_dir: str = "outputs/checkpoints"
    evaluation_dir: str = "outputs/evaluations"
    log_dir: str = "outputs/logs"
    resume_from: str | None = None

    # Strategy
    max_grad_norm: float = 1.0
    max_training_steps: int = 100
    save_steps: int = 10
    eval_steps: int = 10
    enable_gradient_checkpoint: bool = True
    mixed_precision: Literal["no", "bf16", "fp16"] = "bf16"
    gradient_accumulation_steps: int = 1
    cudnn_deterministic: bool = False
    cudnn_benchmark: bool = True

    # Data
    batch_size_per_process: int = 1
    data_loader_workers: int = 8

    # Inference
    num_inference_steps: int = 50
    cfg_scale: float = 0

    def __post_init__(self):
        self.init_handlers = [
            self._init_context,
            self._init_project,
            self._init_pipeline,
            self._init_trainable,
            self._init_optimizer,
            self._init_lr_scheduler,
            self._init_data_loader,
        ]
        self._initialized = False
        self._summary_table = {}

    def _init_context(self):
        # TODO: DDP implementation
        dist.init_process_group("nccl")

        self.world_size = dist.get_world_size()
        self.global_rank = dist.get_rank()
        self.local_rank = self.global_rank % torch.cuda.device_count()
        self.device = torch.device(self.local_rank)
        self.is_main_process = self.global_rank == 0
        torch.cuda.set_device(self.device)

        self.random_seed = self.global_rank + self.base_seed
        self.generator = torch.Generator(self.device).manual_seed(self.random_seed)
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed(self.random_seed)
        torch.cuda.manual_seed_all(self.random_seed)

        torch.backends.cudnn.deterministic = self.cudnn_deterministic
        torch.backends.cudnn.benchmark = self.cudnn_benchmark

        # TODO: Mixed precision
        self._train_dtype = torch.bfloat16
        self._eval_dtype = torch.bfloat16

        logger.info(f"Context initialized.")

    def _init_project(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.evaluation_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        logger.info(f"Project initialized.")

    def _init_pipeline(self):
        self.pipe: BasePipeline = instantiate(
            self.pipe_configs, generator=self.generator, device=self.device, dtype=self._eval_dtype
        )
        logger.info(f"Pipeline initialized.")

    def _init_trainable(self):
        r"""
        Initialize trainable parameters in the pipeline
        or add trainable adapters on it.
        """
        pass

    def _init_optimizer(self):
        self.optimizer: Optimizer = instantiate(self.optimizer_configs, params=self.pipe.trainable_params)
        logger.info(f"Optimizer initialized.")

    def _init_data_loader(self):
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

    def init_everything(self):
        r"""
        Invoke the init handlers sequentially.
        """
        if self._initialized:
            return
        for handler in self.init_handlers:
            handler()

        if len(self.pipe.trainable_params) > 0:
            for p in self.pipe.trainable_params:
                p = p.to(self.device, self._train_dtype)

            # self.pipe.transformer = DistributedDataParallel(
            #     self.pipe.transformer,
            #     device_ids=[self.local_rank],
            #     find_unused_parameters=True,
            # )

        self.update_steps_per_epoch = math.ceil(
            len(self.train_loader) / self.gradient_accumulation_steps / self.world_size
        )
        self.num_epochs = math.ceil(self.max_training_steps / self.update_steps_per_epoch)

        if self.resume_from is not None and os.path.exists(self.resume_from):
            self.load_checkpoints(self.resume_from)

        if self.enable_gradient_checkpoint:
            self.pipe.transformer.enable_gradient_checkpointing()

        self.lr_scheduler = None
        if self.lr_scheduler_configs is not None:
            self.lr_scheduler = get_scheduler(
                name=self.lr_scheduler_configs.name,
                optimizer=self.optimizer,
                step_rules=self.lr_scheduler_configs.step_rules,
                num_warmup_steps=self.lr_scheduler_configs.num_warmup_steps * self.world_size,
                num_training_steps=self.training_steps_per_process * self.world_size,
                num_cycles=self.lr_scheduler_configs.num_cycles,
                power=self.lr_scheduler_configs.power,
                last_epoch=self.lr_scheduler_configs.last_epoch,
            )
        self._initialized = True

    @staticmethod
    def unwrap_model(model: DistributedDataParallel | torch.nn.Module):
        if isinstance(model, DistributedDataParallel):
            model = model.module
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def save_checkpoints(self, global_step: int):
        if not self.is_main_process:
            return
        if not (global_step == 1 or (global_step % self.save_steps == 0) or global_step == self.max_training_steps):
            return
        transformer = self.unwrap_model(self.pipe.transformer)
        states_to_save = {}
        for n, p in transformer.named_parameters():
            if p.requires_grad:
                states_to_save[n] = p
        save_path = os.path.join(self.checkpoint_dir, f"checkpoints-{global_step}.safetensors")
        save_file(states_to_save, save_path)
        logger.info(f"Checkpoints saved to {save_path}.")

    def load_checkpoints(self, checkpoint_path: str, **kwargs):
        states = load_file(checkpoint_path)
        self.pipe.transformer.load_state_dict(states)
        logger.info(f"Load checkpoints from {checkpoint_path}.")

    def train(self):
        r"""
        Train pipeline.
        """
        self.init_everything()

        pipe_summary = self.pipe.summary
        summary_table = get_summary_table(pipe_summary)
        logger.info(summary_table)

        global_step = 0
        metrics = {"loss": 0.0}
        align = len(str(self.max_training_steps))
        for epoch in range(self.num_epochs):
            for step, batch in enumerate(self.train_loader):
                global_step += 1
                loss_dict = self.pipe.forward_step(batch)

                if isinstance(loss_dict, dict):
                    loss = loss_dict["loss"]
                else:
                    loss = loss_dict

                loss = loss / self.gradient_accumulation_steps
                loss.backward()

                metrics["loss"] += loss.item()

                if isinstance(loss_dict, dict):
                    for k, l in loss_dict.items():
                        if k not in metrics:
                            metrics[k] = 0
                        metrics[k] += l.item()

                if global_step % self.gradient_accumulation_steps != 0:
                    continue

                torch.nn.utils.clip_grad_norm_(self.pipe.trainable_params, self.max_grad_norm)
                self.optimizer.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                self.optimizer.zero_grad()

                metric_info = " | ".join([f"{k}:{l:.6f}" for k, l in metrics.items()])
                logger.info(f"Train [{global_step:->{align}}/{self.max_training_steps}]\n{metric_info}")
                self.save_checkpoints(global_step)
                self.evaluate(global_step)

                # Clear record
                for k in metrics:
                    metrics[k] = 0.0

                # dist.barrier()

        dist.destroy_process_group()
        logger.info(f"🌊 Training Finished.")

    @torch.inference_mode()
    def evaluate(self, global_step: int):
        if not self.is_main_process:
            return
        if self.eval_loader is None:
            return
        if not (global_step == 1 or (global_step % self.eval_steps == 0) or global_step == self.max_training_steps):
            return
        save_dir = os.path.join(self.evaluation_dir, f"step-{global_step}")
        os.makedirs(save_dir, exist_ok=True)
        logger.info(f"Evaluate start, save to {save_dir}.")
        for step, batch in enumerate(self.eval_loader):
            output = self.pipe.eval_step(batch, global_step, self.num_inference_steps, self.cfg_scale)

            if not isinstance(output, (tuple, list)):
                output = [output]

            conditions = batch["conditions"]
            target = batch["target"]
            image_name = batch["image_name"][0]

            # 4D
            tensors = [*conditions, target, *output]
            max_h = max([t.shape[-2] for t in tensors])
            tensors = [t.squeeze(0) for t in tensors]
            tensors = [
                F.pad(
                    input=t,
                    pad=(0, 0, 0, max_h - t.shape[1]),
                    mode="constant",
                    value=0,
                ).to(self.device, dtype=self._eval_dtype)
                for t in tensors
            ]
            tensors = torch.cat(tensors, dim=-1)
            save_path = os.path.join(save_dir, f"{image_name}.jpg")
            save_image(tensors, save_path)
            logger.info(f"Eval [{step+1}/{len(self.eval_loader)}] {image_name}")
        logger.info(f"Evaluation finished, saved to {save_dir}.")
