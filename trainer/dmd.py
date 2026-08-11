import torch
import torch.distributed as dist
import torch.distributed.checkpoint as DCP
import torch.nn.functional as F

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from diffusers.optimization import get_scheduler
from hydra.utils import instantiate
from loguru import logger
from omegaconf import OmegaConf
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)

from trainer.base_trainer import BaseTrainer
from trainer.lora_utils import add_trainable_lora, merge_lora, save_lora_adapter
from trainer.parallel.fsdp_strategy import FSDPStrategy
from trainer.parallel.utils import is_fsdp_module, wait_for_everyone
from utils.summary import get_summary_table


@dataclass
class DMDTrainer(BaseTrainer):
    r"""DMD2 trainer with independent Student, Fake and Teacher DiTs."""

    lora_configs: OmegaConf | None = None
    fake_optimizer_configs: OmegaConf | None = None

    sft_lora_path: str | None = None
    sft_lora_scale: float = 1.0
    sft_adapter_name: str | None = None

    teacher_cfg_type: Literal["progressive", "condition_weighted"] = "condition_weighted"
    teacher_cfg_mode: Literal["combined", "text", "mask", "none"] = "combined"
    teacher_text_cfg_scale: float = 4.0
    teacher_mask_cfg_scale: float = 1.0  # no use

    fake_update_ratio: int = 5
    dmd_timestep_min: float = 0.02
    dmd_timestep_max: float = 0.98
    dmd_normalizer_min: float = 1e-3
    dmd_gradient_clip: float = 10.0
    dmd_mask_weight: float = 1.0
    dmd_background_weight: float = 0.1
    dmd_edge_weight: float = 0.0
    mask_loss_weight: float = 0.0

    rollout_end_step_mode: Literal["random", "last"] = "random"
    student_gradient_mode: Literal["full", "last_step"] = "full"
    poisson_in_dmd_score: bool = True

    student_model_state_dict_dir: str = "student_model"
    fake_model_state_dict_dir: str = "fake_model"
    student_optimizer_state_dict_dir: str = "student_optimizer"
    fake_optimizer_state_dict_dir: str = "fake_optimizer"
    student_adapter_state_dict_dir: str = "student_lora"
    fake_adapter_state_dict_dir: str = "fake_lora"

    def _init_pipeline(self):
        # Base pipeline owns the shared VAE/text encoder and the Student DiT.
        super()._init_pipeline()
        self.pipe.cfg_type = self.teacher_cfg_type

    def _prepare_role_transformer(self, trainable: bool):
        transformer = self.pipe.load_transformer()
        if self.sft_lora_path:
            merge_lora(
                transformer,
                self.sft_lora_path,
                self.sft_adapter_name,
                self.sft_lora_scale,
            )
        if trainable:
            params = add_trainable_lora(transformer, self.lora_configs, self.device, self._train_dtype)
        else:
            transformer.requires_grad_(False)
            params = []

        # Shard each extra DiT immediately after it is constructed so three full
        # unsharded DiTs do not have to coexist on every rank.
        if FSDPStrategy.is_full_shard(self.fsdp_strategy):
            self.pipe.setup_additional_transformer(
                transformer,
                FSDPStrategy.FULL_SHARD,
                self.device,
                self._train_dtype,
            )
        return transformer, params

    def _init_trainable(self):
        if self.sft_lora_path:
            merge_lora(
                self.pipe.transformer,
                self.sft_lora_path,
                self.sft_adapter_name,
                self.sft_lora_scale,
            )
        self.student_params = add_trainable_lora(
            self.pipe.transformer,
            self.lora_configs,
            self.device,
            self._train_dtype,
        )

        self.fake_transformer, self.fake_params = self._prepare_role_transformer(trainable=True)
        self.teacher_transformer, _ = self._prepare_role_transformer(trainable=False)
        self.teacher_transformer.eval()

    def _init_parallel_modules(self):
        self.pipe.setup_fsdp_modules(self.fsdp_strategy, self.device, self._train_dtype)
        if FSDPStrategy.is_no_shard(self.fsdp_strategy):
            self.pipe.setup_additional_transformer(
                self.fake_transformer,
                self.fsdp_strategy,
                self.device,
                self._train_dtype,
            )
            self.pipe.setup_additional_transformer(
                self.teacher_transformer,
                self.fsdp_strategy,
                self.device,
                self._train_dtype,
            )

        self.pipe.fsdp_modules.extend([self.fake_transformer, self.teacher_transformer])
        if self.enable_gradient_checkpoint:
            self.unwrap_model(self.pipe.transformer).enable_gradient_checkpointing()
            self.unwrap_model(self.fake_transformer).enable_gradient_checkpointing()

        self.sync_trainable_parameters()
        self._sync_parameters(self.fake_params)

    def _init_optimizer(self):
        super()._init_optimizer()
        self.fake_optimizer = instantiate(self.fake_optimizer_configs, params=self.fake_params)
        self.fake_lr_scheduler = None
        if self.lr_scheduler_configs is not None:
            self.fake_lr_scheduler = get_scheduler(
                name=self.lr_scheduler_configs.name,
                optimizer=self.fake_optimizer,
                step_rules=self.lr_scheduler_configs.step_rules,
                num_warmup_steps=self.lr_scheduler_configs.num_warmup_steps * self.world_size,
                num_training_steps=self.max_training_steps * self.fake_update_ratio,
                num_cycles=self.lr_scheduler_configs.num_cycles,
                power=self.lr_scheduler_configs.power,
                last_epoch=self.lr_scheduler_configs.last_epoch,
            )
        logger.info("Fake optimizer initialized.")

    def _sync_parameters(self, params):
        if self.world_size <= 1 or not FSDPStrategy.is_no_shard(self.fsdp_strategy):
            return
        for param in params:
            dist.broadcast(param.data, src=0)

    def _sync_role_gradients(self, params):
        if self.world_size <= 1 or not FSDPStrategy.is_no_shard(self.fsdp_strategy):
            return
        for param in params:
            if param.grad is None:
                continue
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad.div_(self.world_size)

    @staticmethod
    def _set_role_fsdp_gradient_sync(transformer, enabled: bool):
        if is_fsdp_module(transformer):
            transformer.set_requires_gradient_sync(enabled, recurse=True)

    def _teacher_cfg_scales(self) -> tuple[float, float]:
        if self.teacher_cfg_mode == "text":
            return self.teacher_text_cfg_scale, 1.0
        if self.teacher_cfg_mode == "mask":
            return 1.0, self.teacher_mask_cfg_scale
        if self.teacher_cfg_mode == "none":
            return 1.0, 1.0
        return self.teacher_text_cfg_scale, self.teacher_mask_cfg_scale

    @torch.no_grad()
    def _prepare_dmd_inputs(self, batch):
        text_scale, mask_scale = self._teacher_cfg_scales()
        preprocessed_data = self.pipe.preprocess_inputs(batch)
        return self.pipe.prepare_eval_inputs(preprocessed_data, text_scale, mask_scale)

    def _predict_branch(self, transformer, branch, xt, sigma):
        timestep = sigma.expand(xt.shape[0]).to(device=self.device, dtype=self._train_dtype)
        return self.pipe.denoise_cfg_branch(
            branch,
            xt,
            timestep,
            transformer=transformer,
        )

    def _predict_teacher(self, model_inputs, xt, sigma):
        text_scale, mask_scale = self._teacher_cfg_scales()
        branches = model_inputs.cfg_branches
        predictions = {
            name: self._predict_branch(self.teacher_transformer, branch, xt, sigma)
            for name, branch in branches.items()
        }
        return self.pipe.combine_cfg_predictions(
            predictions,
            text_scale,
            mask_scale,
            model_inputs.mask_latents,
        )

    def _poisson_enabled(self, sigma: torch.Tensor) -> bool:
        sigma_value = sigma.flatten()[0].item()
        return (
            self.pipe.enable_poisson_infer
            and self.pipe.poisson_steps[0] <= sigma_value <= self.pipe.poisson_steps[1]
        )

    def _apply_poisson(self, model_inputs, xt, pred, sigma, d_sigma_dt, noise):
        return self.pipe.apply_poisson_to_prediction(
            xt,
            pred,
            sigma,
            d_sigma_dt,
            model_inputs.conditions[0],
            model_inputs.mask_latents,
            noise,
            model_inputs.height,
            model_inputs.width,
            disable_progress_bar=True,
        )

    def _student_rollout(self, model_inputs, requires_grad: bool) -> torch.Tensor:
        noise = torch.randn_like(model_inputs.noise, generator=self.generator)
        xt = noise
        with self.pipe.scheduler.inference(self.num_inference_steps, img_seq_len=xt.shape[1]) as inferencer:
            schedule = list(inferencer)

        end_step = len(schedule) - 1
        if self.rollout_end_step_mode == "random":
            sampled_step = torch.randint(
                0,
                len(schedule),
                (1,),
                generator=self.generator,
                device=self.device,
            )
            if self.world_size > 1:
                dist.broadcast(sampled_step, src=0)
            end_step = sampled_step.item()

        x0_pred = None
        for step, (sigma, curr_sigma, next_sigma, d_sigma_dt) in enumerate(schedule[: end_step + 1]):
            sigma = sigma.to(self.device)
            curr_sigma = curr_sigma.to(self.device)
            next_sigma = next_sigma.to(self.device)
            d_sigma_dt = d_sigma_dt.to(self.device)

            step_requires_grad = requires_grad and (
                self.student_gradient_mode == "full" or step == end_step
            )
            if step_requires_grad and self.student_gradient_mode == "last_step":
                xt = xt.detach()

            with torch.set_grad_enabled(step_requires_grad):
                pred = self._predict_branch(
                    self.pipe.transformer,
                    model_inputs.cfg_branches["pm"],
                    xt,
                    sigma,
                )
                if self._poisson_enabled(sigma):
                    # Poisson stays in the forward sampler. Its iterative solver is
                    # treated as a projection and receives an identity STE gradient.
                    with torch.no_grad():
                        refined_pred = self._apply_poisson(
                            model_inputs,
                            xt,
                            pred,
                            curr_sigma,
                            d_sigma_dt,
                            noise,
                        )
                    pred = pred + (refined_pred - pred).detach()

                x0_pred = self.pipe.scheduler.predict_x0(
                    xt,
                    curr_sigma,
                    d_sigma_dt,
                    pred,
                    model_inputs.conditions[0],
                    model_inputs.mask_latents,
                    noise,
                )
                timestep = sigma.expand(xt.shape[0]).to(device=self.device, dtype=self._train_dtype)
                runtime_mask = self.pipe.inference_mask(timestep, model_inputs.mask_latents)
                xt = self.pipe.scheduler.step(
                    xt,
                    pred,
                    curr_sigma,
                    next_sigma,
                    d_sigma_dt,
                    model_inputs.conditions[0],
                    runtime_mask,
                    noise,
                )
        return x0_pred

    def _sample_dmd_sigma(self, batch_size: int, img_seq_len: int):
        # A shared sigma keeps all samples on the same Poisson/CFG execution path.
        t = torch.rand((1,), generator=self.generator, device=self.device)
        t = self.dmd_timestep_min + (self.dmd_timestep_max - self.dmd_timestep_min) * t
        sigma, d_sigma_dt = self.pipe.scheduler.get_sigmas(t, img_seq_len, return_d_sigmas_dt=True)
        return sigma.expand(batch_size), d_sigma_dt.expand(batch_size)

    def _loss_weights(self, model_inputs):
        mask = model_inputs.mask_latents.float()
        edge = model_inputs.edge_latents.float()
        weights = self.dmd_mask_weight * mask + self.dmd_background_weight * (1.0 - mask)
        if self.dmd_edge_weight > 0:
            weights = weights + self.dmd_edge_weight * edge
        normalizer = weights.reshape(weights.shape[0], -1).mean(dim=1).clamp_min(1e-6)
        return weights / normalizer.view(-1, 1, 1)

    def _project_score_prediction(
        self,
        model_inputs,
        xt,
        pred,
        sigma,
        d_sigma_dt,
        noise,
    ):
        if self.poisson_in_dmd_score and self._poisson_enabled(sigma):
            pred = self._apply_poisson(model_inputs, xt, pred, sigma, d_sigma_dt, noise)
        return self.pipe.scheduler.predict_x0(
            xt,
            sigma,
            d_sigma_dt,
            pred,
            model_inputs.conditions[0],
            model_inputs.mask_latents,
            noise,
        )

    def _student_loss(self, batch, model_inputs):
        generated = self._student_rollout(model_inputs, requires_grad=True)
        score_noise = torch.randn_like(generated, generator=self.generator)
        sigma, d_sigma_dt = self._sample_dmd_sigma(generated.shape[0], generated.shape[1])
        xt = self.pipe.scheduler.add_noise_by_sigmas(
            score_noise,
            generated.detach(),
            sigma,
            model_inputs.conditions[0],
            model_inputs.mask_latents,
        )

        with torch.no_grad():
            fake_pred = self._predict_branch(
                self.fake_transformer,
                model_inputs.cfg_branches["pm"],
                xt,
                sigma,
            )
            teacher_pred = self._predict_teacher(model_inputs, xt, sigma)
            fake_x0 = self._project_score_prediction(
                model_inputs, xt, fake_pred, sigma, d_sigma_dt, score_noise
            )
            teacher_x0 = self._project_score_prediction(
                model_inputs, xt, teacher_pred, sigma, d_sigma_dt, score_noise
            )
            normalizer = (generated.detach().float() - teacher_x0.float()).abs()
            normalizer = normalizer.reshape(normalizer.shape[0], -1).mean(dim=1)
            normalizer = normalizer.clamp_min(self.dmd_normalizer_min).view(-1, 1, 1)
            dmd_gradient = (fake_x0.float() - teacher_x0.float()) / normalizer
            if self.dmd_gradient_clip > 0:
                dmd_gradient = dmd_gradient.clamp(-self.dmd_gradient_clip, self.dmd_gradient_clip)
            target = (generated.float() - dmd_gradient).detach()

        weights = self._loss_weights(model_inputs)
        dmd_loss = 0.5 * (weights * F.mse_loss(generated.float(), target, reduction="none")).mean()
        loss = dmd_loss
        loss_dict = {"loss": loss, "dmd_loss": dmd_loss}

        if self.mask_loss_weight > 0:
            mask_loss_output = self.pipe.forward_step(batch)
            mask_loss = mask_loss_output["loss"] if isinstance(mask_loss_output, dict) else mask_loss_output
            loss = loss + self.mask_loss_weight * mask_loss
            loss_dict.update(loss=loss, mask_loss=mask_loss)
        return loss_dict

    def _fake_loss(self, model_inputs):
        with torch.no_grad():
            generated = self._student_rollout(model_inputs, requires_grad=False)

        noise = torch.randn_like(generated, generator=self.generator)
        sigma, _ = self._sample_dmd_sigma(generated.shape[0], generated.shape[1])
        xt = self.pipe.scheduler.add_noise_by_sigmas(
            noise,
            generated,
            sigma,
            model_inputs.conditions[0],
            model_inputs.mask_latents,
        )
        target = self.pipe.scheduler.get_velocity(
            noise,
            generated,
            model_inputs.conditions[0],
            model_inputs.mask_latents,
        )
        fake_pred = self._predict_branch(
            self.fake_transformer,
            model_inputs.cfg_branches["pm"],
            xt,
            sigma,
        )
        weights = self._loss_weights(model_inputs)
        return (weights * F.mse_loss(fake_pred.float(), target.float(), reduction="none")).mean()

    def _optimizer_update(self, batches, role: Literal["student", "fake"]):
        if role == "student":
            transformer = self.pipe.transformer
            params = self.student_params
            optimizer = self.optimizer
            scheduler = self.lr_scheduler
        else:
            transformer = self.fake_transformer
            params = self.fake_params
            optimizer = self.fake_optimizer
            scheduler = self.fake_lr_scheduler

        optimizer.zero_grad()
        metrics = {}
        for index, (batch, model_inputs) in enumerate(batches):
            self._set_role_fsdp_gradient_sync(transformer, index == len(batches) - 1)
            if role == "student":
                loss_dict = self._student_loss(batch, model_inputs)
                loss = loss_dict["loss"]
            else:
                fake_loss = self._fake_loss(model_inputs)
                loss = fake_loss
                loss_dict = {"fake_loss": fake_loss}

            (loss / len(batches)).backward()
            for name, value in loss_dict.items():
                metrics[name] = metrics.get(name, 0.0) + value.detach().item() / len(batches)

        self._sync_role_gradients(params)
        torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        return metrics

    def save_model_checkpoints(self, checkpoint_dir: Path):
        options = StateDictOptions(full_state_dict=False, ignore_frozen_params=True)
        student_states = get_model_state_dict(self.pipe.transformer, options=options)
        fake_states = get_model_state_dict(self.fake_transformer, options=options)
        DCP.save(
            {"model": student_states},
            checkpoint_id=str(checkpoint_dir / self.student_model_state_dict_dir),
        )
        DCP.save(
            {"model": fake_states},
            checkpoint_id=str(checkpoint_dir / self.fake_model_state_dict_dir),
        )
        self._save_lora_adapters(checkpoint_dir)

    def load_model_checkpoints(self, checkpoint_dir: Path):
        options = StateDictOptions(full_state_dict=False, ignore_frozen_params=True, strict=False)
        for transformer, directory in (
            (self.pipe.transformer, self.student_model_state_dict_dir),
            (self.fake_transformer, self.fake_model_state_dict_dir),
        ):
            model_path = checkpoint_dir / directory
            if not model_path.exists():
                logger.warning(f"Model checkpoint not found: {model_path}.")
                continue
            states = get_model_state_dict(transformer, options=options)
            DCP.load({"model": states}, checkpoint_id=str(model_path))
            set_model_state_dict(transformer, states, options=options)

    def save_optimizer_checkpoints(self, checkpoint_dir: Path):
        if not self.enable_save_optimizer:
            return
        options = StateDictOptions(full_state_dict=False, ignore_frozen_params=True)
        for transformer, optimizer, directory in (
            (self.pipe.transformer, self.optimizer, self.student_optimizer_state_dict_dir),
            (self.fake_transformer, self.fake_optimizer, self.fake_optimizer_state_dict_dir),
        ):
            states = get_optimizer_state_dict(transformer, optimizer, options=options)
            DCP.save({"optimizer": states}, checkpoint_id=str(checkpoint_dir / directory))

    def load_optimizer_checkpoints(self, checkpoint_dir: Path):
        if not self.enable_save_optimizer:
            return
        options = StateDictOptions(full_state_dict=False, ignore_frozen_params=True, strict=False)
        for transformer, optimizer, directory in (
            (self.pipe.transformer, self.optimizer, self.student_optimizer_state_dict_dir),
            (self.fake_transformer, self.fake_optimizer, self.fake_optimizer_state_dict_dir),
        ):
            optimizer_path = checkpoint_dir / directory
            if not optimizer_path.exists():
                logger.warning(f"Optimizer checkpoint not found: {optimizer_path}.")
                continue
            states = get_optimizer_state_dict(transformer, optimizer, options=options)
            DCP.load({"optimizer": states}, checkpoint_id=str(optimizer_path))
            set_optimizer_state_dict(transformer, optimizer, states, options=options)

    def _save_lora_adapters(self, checkpoint_dir: Path):
        save_lora_adapter(
            self.unwrap_model(self.pipe.transformer),
            self.lora_configs,
            checkpoint_dir / self.student_adapter_state_dict_dir,
            self.is_main_process,
        )
        save_lora_adapter(
            self.unwrap_model(self.fake_transformer),
            self.lora_configs,
            checkpoint_dir / self.fake_adapter_state_dict_dir,
            self.is_main_process,
        )

    def on_train_end(self, global_step: int):
        if global_step <= 0:
            return
        checkpoint_dir = Path(self.checkpoint_dir) / f"step-{global_step}"
        wait_for_everyone()
        self._save_lora_adapters(checkpoint_dir)
        wait_for_everyone()

    def train(self):
        self.init_everything()
        logger.info(get_summary_table(self.pipe.summary))

        global_step = self.global_step
        micro_step = self.micro_step
        align = len(str(self.max_training_steps))
        pending_batches = []
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

            for batch in self.train_loader:
                micro_step += 1
                self.micro_step = micro_step
                batch = self.preprocess_train_batch(batch, global_step)
                pending_batches.append((batch, self._prepare_dmd_inputs(batch)))
                if len(pending_batches) < self.gradient_accumulation_steps:
                    continue

                metrics = self._optimizer_update(pending_batches, "student")
                fake_metrics = {}
                for _ in range(self.fake_update_ratio):
                    fake_metrics = self._optimizer_update(pending_batches, "fake")
                metrics.update(fake_metrics)
                pending_batches.clear()

                global_step += 1
                self.global_step = global_step
                metric_info = " | ".join(f"{name}:{value:.6f}" for name, value in metrics.items())
                logger.info(f"DMD Train [{global_step:->{align}}/{self.max_training_steps}]\n{metric_info}")

                self.save_checkpoints(global_step)
                self.evaluate(global_step)
                if global_step >= self.max_training_steps:
                    break
            if global_step >= self.max_training_steps:
                break

        self.on_train_end(global_step)
        wait_for_everyone()
        dist.destroy_process_group()
        logger.info("DMD Training Finished.")
