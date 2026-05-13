import os
import math
import torch

from diffusers.optimization import get_scheduler

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration

from omegaconf import OmegaConf
from hydra.utils import instantiate
from loguru import logger

from data_module.dataset import SchemaDataset
from pipelines.pipeline_manager import DiTPipelineManager
from utils.summary import get_summary_table


class Trainer:

    def __init__(
        self,
        data_loader_workers: int,
        batch_size_per_process: int,
        optimizer_lr: float,
        training_steps_per_process: int,
        log_steps_per_process: int,
        eval_steps_per_process: int,
        pipeline_configs: OmegaConf,
        optimizer_configs: OmegaConf,
        train_data_configs: OmegaConf,
        random_seed: int = 0,
        num_epochs: int | None = None,
        num_warmup_steps_per_process: int | None = None,
        max_grad_norm: float = 1.0,
        eval_data_configs: OmegaConf | None = None,
        adapter_configs: OmegaConf | None = None,
        lr_scheduler_configs: OmegaConf | None = None,
    ):
        self.pipeline_configs = pipeline_configs
        self.optimizer_configs = optimizer_configs
        self.train_data_configs = train_data_configs
        self.eval_data_configs = eval_data_configs
        self.adapter_configs = adapter_configs
        self.lr_scheduler_configs = lr_scheduler_configs

        self.data_loader_workers = data_loader_workers
        self.batch_size_per_process = batch_size_per_process
        self.optimizer_lr = optimizer_lr
        self.training_steps_per_process = training_steps_per_process
        self.log_steps_per_process = log_steps_per_process
        self.eval_steps_per_process = eval_steps_per_process
        self.num_warmup_steps_per_process = num_warmup_steps_per_process
        self.random_seed = random_seed
        self.num_epochs = num_epochs
        self.max_grad_norm = max_grad_norm

        self.initialization_handlers = [
            self._init_pipeline,
            self._init_adapter,
            self._init_optimizer,
            self._init_data_loader,
        ]

        self._train_dtype = None
        self._eval_dtype = torch.float32

    def _init_pipeline(self, accelerator: Accelerator):
        if self._train_dtype is None:
            self._train_dtype = torch.float32
            if accelerator.mixed_precision == "bf16":
                self._train_dtype = torch.bfloat16
            elif accelerator.mixed_precision == "fp16":
                self._train_dtype = torch.float16

        self.pipeline: DiTPipelineManager = instantiate(self.pipeline_configs)
        self.pipeline.init_components(
            self.pipeline.pretrained_model_name_or_path,
            torch_dtype=self._train_dtype,
            device=accelerator.device,
        )

    def _init_adapter(self, accelerator: Accelerator):
        r"""Add adapter (e.g. LoRA, Control Net) to the loaded pipeline."""
        pass

    def _init_optimizer(self, accelerator: Accelerator):
        self.optimizer: Optimizer = instantiate(
            self.optimizer_configs,
            lr=self.optimizer_lr,
            params=[p for p in self.pipeline.trainable_parameters.values()],
        )

    def _init_data_loader(self, accelerator: Accelerator):
        trainset: SchemaDataset = instantiate(self.train_data_configs)
        self.train_loader = DataLoader(
            dataset=trainset,
            batch_size=self.batch_size_per_process,
            num_workers=self.data_loader_workers,
        )
        self.eval_loader = None
        if self.eval_data_configs is not None:
            evalset: SchemaDataset = instantiate(self.eval_data_configs)
            self.eval_loader = DataLoader(dataset=evalset, batch_size=1, num_workers=self.data_loader_workers)

    def init_everything(self, accelerator: Accelerator):
        for handler in self.initialization_handlers:
            handler(accelerator)

        modules_to_prepare = [getattr(self.pipeline, m) for m in self.pipeline.trainable_modules]
        obj_to_prepare = [*modules_to_prepare, self.optimizer, self.train_loader]
        prepared = accelerator(*obj_to_prepare)

        for i, module_name in enumerate(self.pipeline.trainable_modules):
            setattr(self.pipeline, module_name, prepared[i])
        self.train_loader = prepared[-1]
        self.optimizer = prepared[-2]

        self.total_train_batch_size = (
            self.batch_size_per_process * accelerator.gradient_accumulation_steps * accelerator.num_processes
        )
        self.num_update_per_epoch = math.ceil(len(self.train_loader) / accelerator.gradient_accumulation_steps)
        if self.num_epochs is None:
            self.num_epochs = math.ceil(self.training_steps_per_process / self.num_update_per_epoch)

        if getattr(self, "lr_scheduler_configs", None) is not None:
            self.lr_scheduler = get_scheduler(
                name=self.lr_scheduler.name,
                optimizer=self.optimizer,
                step_rules=self.lr_scheduler.step_rules,
                num_warmup_steps=self.lr_scheduler.num_warmup_steps * accelerator.num_processes,
                num_training_steps=self.training_steps_per_process * accelerator.num_processes,
                num_cycles=self.lr_scheduler_configs.num_cycles,
                power=self.lr_scheduler_configs.power,
                last_epoch=self.lr_scheduler_configs.last_epoch,
            )
            self.lr_scheduler = accelerator.prepare(self.lr_scheduler)

    def save_checkpoints(self):
        pass

    @torch.no_grad()
    def eval_step(self):
        pass

    def forwrad_step(self):
        pass

    def train(self, accelerator: Accelerator):
        self.init_everything(accelerator)

        rank = accelerator.process_index
        world_size = accelerator.num_processes
        device = torch.device(f"cuda:{rank}")
        generator = torch.Generator(device).manual_seed(self.random_seed)
        is_main_process = accelerator.is_main_process

        modules_to_accum = [getattr(self.pipeline, m) for m in self.pipeline.trainable_modules]

        pipe_summary = self.pipeline.summary
        summary_table = get_summary_table(pipe_summary)
        logger.info(summary_table)

        sep = "=" * 25
        log_title = f"{sep} Start Training {sep}"
        info = (
            f"\n{log_title}"
            f"\n  World size               : {world_size}"
            f"\n  Random seed              : {self.random_seed}"
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
        for epoch in range(self.num_epochs):
            for batch in self.train_loader:
                with accelerator.accumulate(*modules_to_accum):
                    with torch.no_grad():
                        pass
                    with accelerator.autocast():
                        pass

                    if accelerator.sync_gradients:
                        torch.nn.utils.clip_grad_norm_(self.pipeline.trainable_parameters, self.max_grad_norm)
                    self.optimizer.step()
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                if accelerator.sync_gradients:
                    global_step += 1



    @torch.no_grad()
    def generate(self):
        pass
