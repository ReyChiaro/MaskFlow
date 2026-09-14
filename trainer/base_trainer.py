import json
import math
import os
import random
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as DCP
import torch.nn.functional as F
from diffusers.optimization import get_scheduler
from diffusers.utils.torch_utils import is_compiled_module
from hydra.utils import instantiate
from loguru import logger
from omegaconf import OmegaConf
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torchvision.utils import save_image

from data_module.dataloader import get_dataloader
from data_module.dataset import SchemaDataset
from pipelines.base_pipeline import BasePipeline
from trainer.parallel.fsdp_strategy import FSDPStrategy
from trainer.parallel.handler import parallel_handler
from trainer.parallel.utils import is_fsdp_module, is_main_process, wait_for_everyone
from utils.logger import setup_logger
from utils.summary import get_summary_table


@dataclass
class BaseTrainer:
    # Modules
    pipe_configs: OmegaConf | None = None
    optimizer_configs: OmegaConf | None = None
    train_data_configs: OmegaConf | None = None
    eval_data_configs: OmegaConf | None = None
    lr_scheduler_configs: OmegaConf | None = None

    # Project
    base_seed: int = 42
    resume_from: str | None = None
    enable_save_optimizer: bool = True

    output_dir: str = "outputs"
    checkpoint_dir: str = "outputs/checkpoints"
    evaluation_dir: str = "outputs/evaluations"
    log_dir: str = "outputs/logs"
    model_state_dict_dir: str = "model"
    optimizer_state_dict_dir: str = "optimizer"
    data_sampler_state_dict_file: str = "data_sampler.pth"
    training_state_dict_file: str = "train_state.pth"

    # Strategy
    text_cfg_dropout: float = 0.0
    max_grad_norm: float = 1.0
    max_training_steps: int = 100
    save_steps: int = 10
    eval_steps: int = 10
    enable_gradient_checkpoint: bool = True
    mixed_precision: Literal["no", "bf16", "fp16"] = "bf16"
    gradient_accumulation_steps: int = 1
    cudnn_deterministic: bool = False
    cudnn_benchmark: bool = True
    distributed_timeout_seconds: int = 21600

    # Data Parallel
    fsdp_strategy: FSDPStrategy = FSDPStrategy.NO_SHARD
    # TODO SP if required

    # Data
    batch_size_per_process: int = 1
    data_loader_workers: int = 8

    # Inference
    eval_seed: int = 42
    num_inference_steps: int = 50
    text_cfg_scale: float = 1.0

    def __post_init__(self):
        r"""
        Initialize init_handlers
        NOTE: The sequence init_pipeline->init_trainable->init_parallel_modules->init_optimizer
            *cannot* be disrupted.
        """
        self.init_handlers = [
            self._init_context,
            self._init_project,
            self._init_pipeline,
            self._init_trainable,
            self._init_parallel_modules,  # Must be called after init_trainable
            self._init_optimizer,
            self._init_data_loader,
        ]
        self._initialized = False
        self._summary_table = {}

    def _init_context(self):
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.global_rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", self.global_rank % torch.cuda.device_count()))
        self.device = torch.device("cuda", self.local_rank)

        # Init FSDP if required
        if not dist.is_initialized():
            dist.init_process_group(
                "nccl",
                device_id=self.device,
                timeout=timedelta(seconds=self.distributed_timeout_seconds),
            )
        parallel_handler.setup_parallel()

        # Init FSDP attributes
        self.is_main_process = is_main_process()
        torch.cuda.set_device(self.device)

        self.dp_rank = parallel_handler.dp_rank

        # Init random states
        self.random_seed = self.global_rank + self.base_seed
        self.generator = torch.Generator(self.device).manual_seed(self.random_seed)
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed(self.random_seed)
        torch.cuda.manual_seed_all(self.random_seed)
        self.rng = random.Random(self.random_seed)

        torch.backends.cudnn.deterministic = self.cudnn_deterministic
        torch.backends.cudnn.benchmark = self.cudnn_benchmark

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

    def _init_parallel_modules(self):
        r""" """
        if FSDPStrategy.is_no_shard(self.fsdp_strategy):
            self.pipe.setup_fsdp_modules(
                fsdp_strategy=FSDPStrategy.NO_SHARD,
                device=self.device,
                dtype=self._train_dtype,
            )

        elif FSDPStrategy.is_full_shard(self.fsdp_strategy):
            self.pipe.setup_fsdp_modules(
                fsdp_strategy=FSDPStrategy.FULL_SHARD,
                device=self.device,
                dtype=self._train_dtype,
            )

        else:
            logger.warning(f"Unsupported FSDPStrategy: {self.fsdp_strategy}.")

        if self.enable_gradient_checkpoint:
            self.unwrap_model(self.pipe.transformer).enable_gradient_checkpointing()

        self.sync_trainable_parameters()

    def _init_optimizer(self):
        self.optimizer: Optimizer = instantiate(self.optimizer_configs, params=self.pipe.trainable_params)
        logger.info(f"Optimizer initialized.")

        self.lr_scheduler = None
        if self.lr_scheduler_configs is not None:
            self.lr_scheduler = get_scheduler(
                name=self.lr_scheduler_configs.name,
                optimizer=self.optimizer,
                step_rules=self.lr_scheduler_configs.step_rules,
                num_warmup_steps=self.lr_scheduler_configs.num_warmup_steps * self.world_size,
                num_training_steps=self.max_training_steps,
                num_cycles=self.lr_scheduler_configs.num_cycles,
                power=self.lr_scheduler_configs.power,
                last_epoch=self.lr_scheduler_configs.last_epoch,
            )

    def _init_data_loader(self):
        trainset: SchemaDataset = instantiate(self.train_data_configs)
        self.train_loader, self.train_sampler = get_dataloader(
            trainset,
            batch_size_per_process=self.batch_size_per_process,
            num_workers=self.data_loader_workers,
            num_replicas=self.world_size,
            global_rank=self.global_rank,
            global_seed=self.base_seed,
            drop_last=False,
            is_train=True,
        )
        logger.info(f"Train Dataloader and Sampler initialized, length: {len(self.train_loader)}.")
        self.eval_loader = None
        self.eval_sampler = None
        if self.eval_data_configs is not None:
            evalset: SchemaDataset = instantiate(self.eval_data_configs)
            self.eval_loader, self.eval_sampler = get_dataloader(
                evalset,
                batch_size_per_process=self.batch_size_per_process,
                num_workers=self.data_loader_workers,
                num_replicas=self.world_size,
                global_rank=self.global_rank,
                global_seed=self.base_seed,
                drop_last=False,
                is_train=False,
            )
            logger.info(f"Eval Dataloader and Sampler initialized, length: {len(self.eval_loader)}.")

    def init_everything(self):
        r"""
        Invoke the init handlers sequentially and initialize the training states.
        """
        if self._initialized:
            return

        for handler in self.init_handlers:
            handler()

        # Setup logger for every process
        setup_logger(self.is_main_process, self.global_rank, self.log_dir, "log", log_per_rank=True)

        # Define the update steps, accumulation steps during training
        self.update_steps_per_epoch = math.ceil(len(self.train_loader) / self.gradient_accumulation_steps)
        self.epoch_start = 0
        self.current_epoch = 0
        self.global_step = 0
        self.micro_step = 0
        self.num_epochs = (
            math.ceil(self.max_training_steps * self.gradient_accumulation_steps / max(len(self.train_loader), 1)) + 1
        )

        if self.resume_from is not None and os.path.exists(self.resume_from):
            self.load_checkpoints(self.resume_from)

        self._initialized = True

    def train_state_dict(self) -> dict:
        return {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "micro_step": self.micro_step,
        }

    def load_train_state_dict(self, state_dict):
        self.current_epoch = state_dict.get("epoch", 0)
        self.global_step = state_dict.get("global_step", 0)
        self.micro_step = state_dict.get("micro_step", self.global_step * self.gradient_accumulation_steps)

    @staticmethod
    def unwrap_model(model: DistributedDataParallel | torch.nn.Module):
        if isinstance(model, DistributedDataParallel):
            model = model.module
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def set_fsdp_gradient_sync(self, enabled: bool = True):
        for module in self.pipe.fsdp_modules:
            if is_fsdp_module(module):
                module.set_requires_gradient_sync(enabled, recurse=True)

    def sync_gradients(self):
        if self.world_size <= 1 or not FSDPStrategy.is_no_shard(self.fsdp_strategy):
            return
        for param in self.pipe.trainable_params:
            if param.grad is None:
                continue
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad.div_(self.world_size)

    def sync_trainable_parameters(self):
        if self.world_size <= 1 or not FSDPStrategy.is_no_shard(self.fsdp_strategy):
            return
        for param in self.pipe.trainable_params:
            dist.broadcast(param.data, src=0)

    def save_model_checkpoints(self, checkpoint_dir: Path):
        r"""
        Can be overriden if difference saving strategy is required,
        only parameters with gradients will be saved by default.
        """
        transformer_states = get_model_state_dict(
            model=self.pipe.transformer,
            options=StateDictOptions(full_state_dict=False, ignore_frozen_params=True),
        )
        DCP.save({"model": transformer_states}, checkpoint_id=str(checkpoint_dir / self.model_state_dict_dir))

    def load_model_checkpoints(self, checkpoint_dir: Path):
        model_path = checkpoint_dir / self.model_state_dict_dir
        if not model_path.exists():
            logger.warning(f"Model checkpoint not found: {model_path}.")
            return

        transformer_states = get_model_state_dict(
            self.pipe.transformer,
            options=StateDictOptions(full_state_dict=False, ignore_frozen_params=True),
        )
        DCP.load({"model": transformer_states}, checkpoint_id=str(model_path))
        set_model_state_dict(
            self.pipe.transformer,
            transformer_states,
            options=StateDictOptions(full_state_dict=False, strict=False),
        )

    def save_optimizer_checkpoints(self, checkpoint_dir: Path):
        if not self.enable_save_optimizer:
            return
        optimizer_states = get_optimizer_state_dict(
            model=self.pipe.transformer,
            optimizers=self.optimizer,
            options=StateDictOptions(full_state_dict=False, ignore_frozen_params=True),
        )
        DCP.save({"optimizer": optimizer_states}, checkpoint_id=str(checkpoint_dir / self.optimizer_state_dict_dir))

    def load_optimizer_checkpoints(self, checkpoint_dir: Path):
        if not self.enable_save_optimizer:
            return
        opt_path = checkpoint_dir / self.optimizer_state_dict_dir
        if not opt_path.exists():
            logger.warning(f"Optimizer checkpoints not found: {opt_path}.")
            return

        optimizer_states = get_optimizer_state_dict(
            model=self.pipe.transformer,
            optimizers=self.optimizer,
            options=StateDictOptions(full_state_dict=False, ignore_frozen_params=True),
        )
        DCP.load({"optimizer": optimizer_states}, checkpoint_id=str(opt_path))
        set_optimizer_state_dict(
            model=self.pipe.transformer,
            optimizers=self.optimizer,
            optim_state_dict=optimizer_states,
            options=StateDictOptions(full_state_dict=False, strict=False),
        )

    def save_checkpoints(self, global_step: int, force: bool = False):
        r"""
        - Training states
        - Data sampler
        - Model: checkpoints of *trainable* parameters of trasnformer by default.
        """
        if not (
            force or global_step == 1 or (global_step % self.save_steps == 0) or global_step == self.max_training_steps
        ):
            return

        checkpoint_dir = Path(self.checkpoint_dir) / f"step-{global_step}"
        if self.is_main_process:
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

        wait_for_everyone()

        self.save_model_checkpoints(checkpoint_dir)
        self.save_optimizer_checkpoints(checkpoint_dir)

        wait_for_everyone()

        if self.is_main_process:
            logger.info(f"Checkpoints saved to {checkpoint_dir}.")

    def load_checkpoints(self, checkpoint_path: str, **kwargs):
        checkpoint_dir = Path(checkpoint_path)
        loaded_components = []

        wait_for_everyone()

        # Data sampler
        ds_path = checkpoint_dir / self.data_sampler_state_dict_file
        if self.train_sampler is not None and ds_path.exists():
            ds_state = torch.load(ds_path)
            self.train_sampler.load_state_dict(ds_state)
            loaded_components.append("data_sampler")

        # Train state
        train_path = checkpoint_dir / self.training_state_dict_file
        if train_path.exists():
            train_state = torch.load(train_path)
            self.load_train_state_dict(train_state)
            loaded_components.append("train_state")

        # Model
        self.load_model_checkpoints(checkpoint_dir)
        loaded_components.append("model")

        # Optimizer
        self.load_optimizer_checkpoints(checkpoint_dir)
        loaded_components.append("optimizer")

        wait_for_everyone()
        logger.info(f"\nLoad checkpoints from {checkpoint_path}.\nSuccessfully load: {','.join(loaded_components)}.")

    def preprocess_train_batch(self, batch, step: int):
        if self.text_cfg_dropout is not None:
            batch["prompt"] = ["" if self.rng.random() < self.text_cfg_dropout else p for p in batch["prompt"]]
        return batch

    def preprocess_eval_batch(self, batch, step: int):
        return batch

    def on_train_end(self, global_step: int):
        r"""
        Hook for trainers to export or finalize artifacts before destroying
        the distributed process group.
        """
        pass

    @staticmethod
    def _safe_eval_file_component(value: object) -> str:
        r"""Return a path-safe, human-readable component for evaluation artifacts."""
        component = Path(str(value)).stem
        component = component.replace("/", "_").replace("\\", "_")
        return component or "sample"

    def _save_eval_batch(
        self,
        batch,
        output: dict[str, torch.Tensor],
        save_dir: Path,
        dataloader_step: int,
        metadata_file,
    ):
        r"""Save one rank-local evaluation batch without sharing writable files."""
        prompt: list[str] = batch.get("prompt", [""])
        neg_prompt: list[str] = batch.get("negative_prompt", [""])
        conditions: dict[str, torch.Tensor] | None = batch.get("conditions", None)
        target: torch.Tensor | None = batch.get("target", None)
        image_name = batch.get("image_name", None)

        for batch_idx in range(len(prompt)):
            tensors = []

            if conditions is not None:
                tensors.extend([conditions[k][batch_idx] for k in conditions])

            if target is not None:
                tensors.append(target[batch_idx])

            tensors.extend([output[k][batch_idx] for k in output])
            p = prompt[batch_idx]
            np = neg_prompt[batch_idx]

            source_name = image_name[batch_idx] if image_name is not None else "sample"
            source_name = self._safe_eval_file_component(source_name)
            save_name = f"{source_name}__rank-{self.global_rank:01d}_batch-{dataloader_step:04d}_item-{batch_idx:04d}"

            max_h = max([t.shape[-2] for t in tensors])
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

            save_image(tensors.float().cpu(), save_dir / f"{save_name}.png")
            metadata_file.write(
                json.dumps(
                    {
                        "image_name": save_name,
                        "source_image_name": source_name,
                        "prompt": p,
                        "negative_prompt": np,
                        "rank": self.global_rank,
                        "dataloader_step": dataloader_step,
                        "batch_index": batch_idx,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            logger.info(f"Eval [{dataloader_step + 1}/{len(self.eval_loader)}] {save_name}")

    def _merge_eval_metadata(self, save_dir: Path, metadata_dir: Path):
        r"""Merge rank-local JSONL files into one deterministic, atomic index."""
        if not self.is_main_process:
            return

        merged_path = save_dir / "prompt.jsonl"
        temporary_path = save_dir / "prompt.jsonl.tmp"
        with open(temporary_path, "w", encoding="utf-8") as merged_file:
            for rank in range(self.world_size):
                rank_path = metadata_dir / f"rank-{rank:05d}.jsonl"
                with open(rank_path, "r", encoding="utf-8") as rank_file:
                    for line in rank_file:
                        merged_file.write(line)
        os.replace(temporary_path, merged_path)

    def train(self):
        r"""
        Train pipeline.
        """
        self.init_everything()

        pipe_summary = self.pipe.summary
        summary_table = get_summary_table(pipe_summary)
        logger.info(summary_table)

        global_step = self.global_step
        micro_step = self.micro_step
        metrics = {"loss": 0.0}
        align = len(str(self.max_training_steps))
        resumed_sampler = self.train_sampler is not None and getattr(self.train_sampler, "resume_idx", 0) > 0
        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch
            if self.train_sampler is not None:
                if resumed_sampler:
                    resumed_sampler = False
                else:
                    self.train_sampler.set_epoch(epoch)
            if self.eval_sampler is not None:
                self.eval_sampler.set_epoch(epoch)

            for step, batch in enumerate(self.train_loader):
                micro_step += 1
                self.micro_step = micro_step
                is_sync_step = micro_step % self.gradient_accumulation_steps == 0
                self.set_fsdp_gradient_sync(is_sync_step)

                batch = self.preprocess_train_batch(batch, global_step)
                loss_dict = self.pipe.forward_step(batch)

                if isinstance(loss_dict, dict):
                    loss: torch.Tensor = loss_dict["loss"]
                else:
                    loss: torch.Tensor = loss_dict

                loss = loss / self.gradient_accumulation_steps
                loss.backward()

                if isinstance(loss_dict, dict):
                    for k, l in loss_dict.items():
                        if k not in metrics:
                            metrics[k] = 0
                        metrics[k] += l.item()
                else:
                    metrics["loss"] += loss.item()

                if not is_sync_step:
                    continue

                self.sync_gradients()
                torch.nn.utils.clip_grad_norm_(self.pipe.trainable_params, self.max_grad_norm)
                self.optimizer.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                self.optimizer.zero_grad()

                global_step += 1
                self.global_step = global_step
                metric_info = " | ".join([f"{k}:{l:.6f}" for k, l in metrics.items()])
                logger.info(f"Train [{global_step:->{align}}/{self.max_training_steps}]\n{metric_info}")

                self.save_checkpoints(global_step)
                self.evaluate(global_step)

                # Clear record
                for k in metrics:
                    metrics[k] = 0.0

                if global_step >= self.max_training_steps:
                    break
            if global_step >= self.max_training_steps:
                break

        self.on_train_end(global_step)
        wait_for_everyone()
        dist.destroy_process_group()
        logger.info(f"🌊 Training Finished.")

    @torch.inference_mode()
    def evaluate(self, global_step: int, force: bool = False):
        if self.eval_loader is None:
            return
        if not (
            force or global_step == 1 or (global_step % self.eval_steps == 0) or global_step == self.max_training_steps
        ):
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
                    batch,
                    self.num_inference_steps,
                    text_cfg_scale=self.text_cfg_scale,
                    seed=self.eval_seed,
                )
                self._save_eval_batch(batch, output, save_dir, step, metadata_file)

        wait_for_everyone()
        self._merge_eval_metadata(save_dir, metadata_dir)

        if self.is_main_process:
            logger.info(f"Evaluation finished, saved to {save_dir}.")
        wait_for_everyone()
