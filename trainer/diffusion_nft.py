import math
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as DCP
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
from hydra.utils import instantiate
from loguru import logger
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from data_module.dataloader import collate_evaluation
from data_module.sampler import CheckpointDistributedSampler
from models.rewards.rewards import RewardModel
from pipelines.qwenimage.qwenimage_maskflow import QwenImageMaskFlow, QwenMaskFlowForwardOutput
from schedulers.mask_flow import MaskFlowScheduler
from trainer.lora import LoraTrainer
from trainer.lora_utils import merge_lora
from trainer.parallel.fsdp_strategy import FSDPStrategy
from trainer.parallel.utils import is_fsdp_module, wait_for_everyone


@dataclass
class DiffusionNFTTrainer(LoraTrainer):
    r"""Online NFT for Qwen MaskFlow, using one backbone and LoRA snapshots.

    B=batch_size_per_process distinct editing inputs per rank; K=group_size
    independent endpoints per input. Groups never span ranks or mix sources.
    R=rollout_batches_per_round data batches produce R*K training micro-batches
    of B endpoints. Every inner epoch reuses these endpoints with fresh noise.
    G=gradient_accumulation_steps micro-batches, each with T training timesteps,
    produce ONE actor/optimizer/global_step update (G*T backwards).
    A round has inner_epochs*ceil(R*K/G) updates; epoch tails and the last
    max_training_steps update may shorten it. Old is fixed throughout a round.
    """

    # Hydra RewardModel: reward(images=[B,C,H,W] in [0,1], batch=original batch)
    # returns B finite scalar scores, with larger meaning better. No gradients.
    reward_configs: OmegaConf | None = None
    group_size: int = 4
    rollout_batches_per_round: int = 1
    inner_epochs: int = 1
    train_timesteps: int = 9
    advantage_clip: float = 5.0
    global_reward_std: bool = True
    advantage_epsilon: float = 1e-4
    nft_beta: float = 1.0
    reference_weight: float = 1e-4
    reconstruction_epsilon: float = 1e-5
    old_decay: float = 0.5

    # Merge an existing SFT adapter before mounting the trainable NFT adapter.
    sft_lora_path: str | None = None
    sft_lora_scale: float = 1.0
    sft_adapter_name: str = "maskflow"
    nft_state_dict_dir: str = "nft"

    def __post_init__(self):
        super().__post_init__()
        for name in (
            "group_size", "rollout_batches_per_round", "inner_epochs", "train_timesteps",
            "gradient_accumulation_steps", "batch_size_per_process", "num_inference_steps",
            "max_training_steps", "save_steps", "eval_steps",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.group_size < 2 or self.train_timesteps > self.num_inference_steps:
            raise ValueError("NFT requires group_size >= 2 and train_timesteps <= num_inference_steps.")
        if not 0 <= self.old_decay <= 1 or self.reference_weight < 0:
            raise ValueError("old_decay must be in [0,1] and reference_weight >= 0.")
        if min(self.nft_beta, self.advantage_clip, self.advantage_epsilon, self.reconstruction_epsilon) <= 0:
            raise ValueError("NFT beta, advantage clip and numerical epsilons must be positive.")
        if self.mixed_precision not in ("no", "bf16"):
            raise ValueError("NFT supports no/bf16 precision; fp16 requires loss scaling.")
        if self.fsdp_strategy not in (FSDPStrategy.NO_SHARD, FSDPStrategy.FULL_SHARD):
            raise ValueError(f"Unsupported fsdp_strategy: {self.fsdp_strategy}.")
        if self.reward_configs is None or self.lora_configs is None:
            raise ValueError("reward_configs and lora_configs must be configured for NFT.")
        if self.lora_configs.lora_dropout != 0:
            raise ValueError("NFT requires lora_dropout=0 for consistent actor/old predictions.")
        if self.resume_from and not Path(self.resume_from).is_dir():
            raise ValueError(f"NFT checkpoint directory does not exist: {self.resume_from}.")
        self.rollout_step: int = 0
        self.data_batches_consumed: int = 0

    def _init_context(self):
        super()._init_context()
        self._train_dtype = torch.float32 if self.mixed_precision == "no" else torch.bfloat16
        self._eval_dtype = self._train_dtype

    def _init_pipeline(self):
        super()._init_pipeline()
        if not isinstance(self.pipe, QwenImageMaskFlow) or not isinstance(self.pipe.scheduler, MaskFlowScheduler):
            raise ValueError("DiffusionNFTTrainer currently requires QwenImageMaskFlow and MaskFlowScheduler.")
        self.reward_fn: RewardModel = instantiate(self.reward_configs, device=self.device)
        if not isinstance(self.reward_fn, RewardModel):
            raise TypeError("reward_configs must instantiate models.rewards.rewards.RewardModel.")

    def _init_trainable(self):
        if self.sft_lora_path:
            merge_lora(self.pipe.transformer, self.sft_lora_path, self.sft_adapter_name, self.sft_lora_scale)
        super()._init_trainable()
        if self.world_size > 1 and FSDPStrategy.is_full_shard(self.fsdp_strategy):
            # FSDP2 does not broadcast initialization. Synchronize freshly random
            # LoRA factors before slicing them; NO_SHARD does this in BaseTrainer.
            with torch.no_grad():
                for param in self.pipe.trainable_params:
                    dist.broadcast(param, src=0)

    def _init_parallel_modules(self):
        super()._init_parallel_modules()
        # Capture AFTER sharding and rank-0 synchronization. Never cache pre-FSDP
        # Parameter objects. DTensor clones retain the same mesh/placements.
        self.actor_params: dict[str, torch.nn.Parameter] = {
            n: p for n, p in self.pipe.transformer.named_parameters() if p.requires_grad
        }
        self.old_params: dict[str, torch.Tensor] = {n: p.detach().clone() for n, p in self.actor_params.items()}
        self.reference_params: dict[str, torch.Tensor] = {n: p.detach().clone() for n, p in self.actor_params.items()}

    def _init_data_loader(self):
        super()._init_data_loader()
        # Use the existing checkpoint sampler even on one GPU. Count consumed
        # batches ourselves: DataLoader prefetch must not advance the saved cursor.
        self.train_sampler = CheckpointDistributedSampler(
            self.train_loader.dataset, self.batch_size_per_process, self.world_size,
            self.global_rank, shuffle=True, seed=self.base_seed, drop_last=True,
        )
        self.train_loader = DataLoader(
            self.train_loader.dataset, batch_size=self.batch_size_per_process,
            sampler=self.train_sampler, num_workers=self.data_loader_workers,
            drop_last=True, collate_fn=collate_evaluation,
        )
        if len(self.train_loader) == 0:
            raise ValueError("Dataset must contain at least world_size*batch_size_per_process editing inputs.")

    def init_everything(self):
        super().init_everything()
        # BaseTrainer's one-pass SFT estimate does not account for groups/reuse.
        batches = self.train_sampler.num_samples // self.batch_size_per_process
        rounds, tail = divmod(batches, self.rollout_batches_per_round)
        self.update_steps_per_epoch = self.inner_epochs * (
            rounds * math.ceil(self.rollout_batches_per_round * self.group_size / self.gradient_accumulation_steps)
            + math.ceil(tail * self.group_size / self.gradient_accumulation_steps)
        )
        self.num_epochs = math.ceil(self.max_training_steps / self.update_steps_per_epoch)

    def _reshard_actor(self):
        # reshard() is not recursive. In particular, discard the root's cached
        # full weights when its policy is reshard_after_forward=False.
        for module in self.pipe.transformer.modules():
            if is_fsdp_module(module):
                module.reshard()

    @contextmanager
    def _use_weights(self, weights: dict[str, torch.Tensor]) -> Iterator[None]:
        r"""No-grad role switch; only call with no outstanding actor graph.

        NO_SHARD copies ordinary tensors; FULL_SHARD copies matching DTensors
        locally, without gathering a full model. Optimizer Parameter identities,
        requires_grad flags and accumulated actor gradients are never replaced.
        """
        self._reshard_actor()
        with torch.no_grad():
            saved = {n: p.detach().clone() for n, p in self.actor_params.items()}
            try:
                for name, param in self.actor_params.items():
                    param.copy_(weights[name])
                yield
            finally:
                self._reshard_actor()
                for name, param in self.actor_params.items():
                    param.copy_(saved[name])

    @torch.no_grad()
    def _sync_old(self):
        self._reshard_actor()
        for name, param in self.actor_params.items():
            self.old_params[name].lerp_(param.detach(), 1.0 - self.old_decay)

    def _predict(self, inputs: QwenMaskFlowForwardOutput, xt: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        # Bypass denoise_cfg_branch's inference_mode wrapper for actor autograd.
        predictions = {}
        for name, branch in inputs.cfg_branches.items():
            predictions[name] = self.pipe.denoise(
                hidden_states=torch.cat([xt] + branch.conditions, dim=1),
                timesteps=sigma.expand(xt.shape[0]).to(device=self.device, dtype=self._train_dtype),
                prompt_embeds=branch.prompt_embeds, prompt_embeds_mask=branch.prompt_embeds_mask,
                img_shapes=branch.image_shapes, img_seq_len=xt.shape[1],
            )
        # At scale=1 use the raw velocity, avoiding unnecessary norm rescaling.
        if self.text_cfg_scale == 1.0:
            return predictions["pm"]
        return self.pipe.combine_cfg_predictions(predictions, self.text_cfg_scale)

    @torch.no_grad()
    def _rollout_group(
        self, batch: dict,
    ) -> tuple[QwenMaskFlowForwardOutput, list[torch.Tensor], torch.Tensor, torch.Tensor]:
        data = self.pipe.preprocess_inputs(batch)
        inputs = self.pipe.prepare_eval_inputs(data, self.text_cfg_scale)
        if inputs.mask_latents is None:
            raise ValueError("NFT editing groups require source and mask conditions.")
        with self.pipe.scheduler.inference(self.num_inference_steps, img_seq_len=inputs.noise.shape[1]) as steps:
            schedule = [tuple(value.to(self.device) for value in step) for step in steps]
        endpoints, rewards = [], []
        for _ in range(self.group_size):
            noise = torch.randn(inputs.noise.shape, device=self.device, dtype=inputs.noise.dtype, generator=self.generator)
            xt = noise
            for sigma, curr_sigma, next_sigma, derivative in schedule:
                prediction = self._predict(inputs, xt, sigma)
                if self.pipe.enable_poisson_infer and self.pipe.poisson_steps[0] <= sigma.item() <= self.pipe.poisson_steps[1]:
                    prediction = self.pipe.apply_poisson_to_prediction(
                        xt, prediction, curr_sigma, derivative, inputs.conditions[0], inputs.mask_latents,
                        noise, inputs.height, inputs.width, disable_progress_bar=True,
                    )
                xt = self.pipe.scheduler.step(
                    xt, prediction, curr_sigma, next_sigma, derivative,
                    inputs.conditions[0], inputs.mask_latents, noise,
                )
            images = self.pipe.decode_image(QwenImageEditPlusPipeline._unpack_latents(
                xt, inputs.height, inputs.width, self.pipe.vae_scale_factor,
            ))
            if self.pipe.enable_pixel_blend:
                images = data.mask * images + (1 - data.mask) * data.raw_source
            scores = torch.as_tensor(self.reward_fn(images=images, batch=batch), device=self.device, dtype=torch.float32)
            if scores.shape != (xt.shape[0],) or not torch.isfinite(scores).all():
                raise ValueError("reward_fn must return one finite scalar per image, shape [B].")
            endpoints.append(xt.detach())
            rewards.append(scores)
        # [K,B] scores preserve group identity even when two inputs share text.
        return inputs, endpoints, torch.stack(rewards), torch.cat([step[0] for step in schedule])

    def _advantages(self, rewards: list[torch.Tensor]) -> list[torch.Tensor]:
        std = None
        if self.global_reward_std:
            values = torch.cat([r.flatten() for r in rewards]).double()
            # Two-pass population variance avoids cancellation for large offsets.
            stats = torch.stack([values.sum(), values.new_tensor(values.numel())])
            if self.world_size > 1:
                dist.all_reduce(stats)
            variance = ((values - stats[0] / stats[1]) ** 2).sum()
            if self.world_size > 1:
                dist.all_reduce(variance)
            std = (variance / stats[1]).sqrt().float()
        return [
            ((r - r.mean(dim=0, keepdim=True)) / (
                (std if std is not None else r.std(dim=0, keepdim=True, correction=0)) + self.advantage_epsilon
            )).clamp(-self.advantage_clip, self.advantage_clip)
            for r in rewards
        ]

    def _policy_loss(
        self, actor: torch.Tensor, old: torch.Tensor, xt: torch.Tensor,
        target: torch.Tensor, sigma: torch.Tensor, advantage: torch.Tensor,
    ) -> torch.Tensor:
        sigma = sigma.reshape(-1, *([1] * (xt.ndim - 1)))
        losses = []
        for direction in (1, -1):
            velocity = old.float() + direction * self.nft_beta * (actor.float() - old.float())
            error = xt.float() - sigma * velocity - target.float()
            normalizer = error.detach().abs().flatten(1).mean(1).clamp_min(self.reconstruction_epsilon)
            losses.append(error.square().flatten(1).mean(1) / normalizer)
        r = 0.5 + advantage / (2 * self.advantage_clip)
        return (self.advantage_clip / self.nft_beta * (r * losses[0] + (1 - r) * losses[1])).mean()

    def _train_micro_batch(self, sample: tuple, sync: bool, accumulation_size: int) -> dict[str, float]:
        inputs, x0, advantage, grid = sample
        # Independent uniform sampling without replacement on the rollout grid.
        indices = torch.stack([
            torch.randperm(len(grid), device=self.device, generator=self.generator)[:self.train_timesteps]
            for _ in range(x0.shape[0])
        ])
        metrics = {"policy_loss": 0.0, "reference_loss": 0.0}
        for index in range(self.train_timesteps):
            self.set_fsdp_gradient_sync(sync and index == self.train_timesteps - 1)
            sigma = grid[indices[:, index]]
            noise = torch.randn(x0.shape, device=self.device, dtype=x0.dtype, generator=self.generator)
            xt = self.pipe.scheduler.add_noise_by_sigmas(noise, x0, sigma, inputs.conditions[0], inputs.mask_latents)
            # The MaskFlow clean endpoint includes source outside the mask.
            # Computing it via the scheduler also supports its unmask_with modes.
            target = self.pipe.scheduler.add_noise_by_sigmas(
                noise, x0, torch.zeros_like(sigma), inputs.conditions[0], inputs.mask_latents,
            )
            with self._use_weights(self.old_params):
                old = self._predict(inputs, xt, sigma).detach()
            reference = None
            if self.reference_weight:
                with self._use_weights(self.reference_params):
                    reference = self._predict(inputs, xt, sigma).detach()
            actor = self._predict(inputs, xt, sigma)
            policy_loss = self._policy_loss(actor, old, xt, target, sigma, advantage)
            reference_loss = actor.new_zeros((), dtype=torch.float32)
            if reference is not None:
                reference_loss = (actor.float() - reference.float()).square().mean()
            loss = policy_loss + self.reference_weight * reference_loss
            (loss / (accumulation_size * self.train_timesteps)).backward()
            metrics["policy_loss"] += policy_loss.detach().item() / self.train_timesteps
            metrics["reference_loss"] += reference_loss.detach().item() / self.train_timesteps
        self.micro_step += 1  # One endpoint micro-batch, independent of T.
        return metrics

    def train_state_dict(self) -> dict:
        return {**super().train_state_dict(), "rollout_step": self.rollout_step,
                "data_batches_consumed": self.data_batches_consumed}

    def load_train_state_dict(self, state_dict: dict):
        super().load_train_state_dict(state_dict)
        self.rollout_step = state_dict["rollout_step"]
        self.data_batches_consumed = state_dict["data_batches_consumed"]

    def save_model_checkpoints(self, checkpoint_dir: Path):
        self._reshard_actor()
        super().save_model_checkpoints(checkpoint_dir)
        # Save detached snapshots too: resuming must not silently reset old or ref.
        DCP.save({"old": self.old_params, "reference": self.reference_params},
                 checkpoint_id=str(checkpoint_dir / self.nft_state_dict_dir))
        torch.save({"generator": self.generator.get_state(), "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state(self.device), "prompt": self.rng.getstate(),
                    "world_size": self.world_size,
                    "lr_scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None},
                   checkpoint_dir / self.nft_state_dict_dir / f"rank-{self.global_rank}.pt")

    def load_model_checkpoints(self, checkpoint_dir: Path):
        super().load_model_checkpoints(checkpoint_dir)
        DCP.load({"old": self.old_params, "reference": self.reference_params},
                 checkpoint_id=str(checkpoint_dir / self.nft_state_dict_dir))
        state = torch.load(checkpoint_dir / self.nft_state_dict_dir / f"rank-{self.global_rank}.pt", weights_only=False)
        if state["world_size"] != self.world_size:
            raise ValueError("NFT resume requires the same world_size to preserve groups and RNG streams.")
        self.generator.set_state(state["generator"])
        torch.set_rng_state(state["torch"])
        torch.cuda.set_rng_state(state["cuda"], self.device)
        self.rng.setstate(state["prompt"])
        if self.lr_scheduler is not None and state["lr_scheduler"] is not None:
            self.lr_scheduler.load_state_dict(state["lr_scheduler"])

    def train(self):
        self.init_everything()
        logger.info(
            f"NFT B={self.batch_size_per_process}, K={self.group_size}, "
            f"R={self.rollout_batches_per_round}, G={self.gradient_accumulation_steps}, T={self.train_timesteps}; "
            f"full actor update batch={self.world_size * self.batch_size_per_process * self.gradient_accumulation_steps} "
            f"endpoints; full round updates={self.inner_epochs * math.ceil(self.rollout_batches_per_round * self.group_size / self.gradient_accumulation_steps)}."
        )
        self.optimizer.zero_grad()
        while self.global_step < self.max_training_steps:
            self.train_sampler.set_epoch(self.current_epoch)
            self.train_sampler.set_resume_idx(self.data_batches_consumed * self.batch_size_per_process)
            iterator = iter(self.train_loader)
            exhausted = False
            while not exhausted and self.global_step < self.max_training_steps:
                groups = []
                self.pipe.transformer.eval()
                with self._use_weights(self.old_params):
                    for _ in range(self.rollout_batches_per_round):
                        batch = next(iterator, None)
                        if batch is None:
                            exhausted = True
                            break
                        batch = self.preprocess_train_batch(batch, self.global_step)
                        groups.append(self._rollout_group(batch))
                        self.data_batches_consumed += 1
                if not groups:
                    break
                advantages = self._advantages([group[2] for group in groups])
                samples = [(inputs, x0, adv[k], grid)
                           for (inputs, endpoints, _, grid), adv in zip(groups, advantages, strict=True)
                           for k, x0 in enumerate(endpoints)]
                start_step = self.global_step
                self.pipe.transformer.train()
                for _ in range(self.inner_epochs):
                    order = torch.randperm(len(samples), generator=self.generator, device=self.device).tolist()
                    for start in range(0, len(samples), self.gradient_accumulation_steps):
                        chunk = order[start:start + self.gradient_accumulation_steps]
                        metrics = {"policy_loss": 0.0, "reference_loss": 0.0}
                        for index, sample_index in enumerate(chunk):
                            values = self._train_micro_batch(samples[sample_index], index == len(chunk) - 1, len(chunk))
                            for name, value in values.items():
                                metrics[name] += value / len(chunk)
                        self.sync_gradients()
                        grad_norm = torch.nn.utils.clip_grad_norm_(list(self.actor_params.values()), self.max_grad_norm)
                        # clip_grad_norm_ operates on DTensors; materialize only
                        # the scalar norm for logging, never individual shards.
                        if hasattr(grad_norm, "full_tensor"):
                            grad_norm = grad_norm.full_tensor()
                        self.optimizer.step()
                        if self.lr_scheduler is not None:
                            self.lr_scheduler.step()
                        self.optimizer.zero_grad()
                        self.global_step += 1
                        values = torch.tensor(list(metrics.values()), device=self.device)
                        if self.world_size > 1:
                            dist.all_reduce(values)
                            values /= self.world_size
                        logger.info(f"NFT actor step {self.global_step}/{self.max_training_steps}, "
                                    f"rollout={self.rollout_step}, loss={values.tolist()}, grad_norm={grad_norm.item():.6f}")
                        if self.global_step >= self.max_training_steps:
                            break
                    if self.global_step >= self.max_training_steps:
                        break
                self._sync_old()
                self.rollout_step += 1
                self.pipe.transformer.eval()
                # Checkpoint only at a round boundary: no partially trained
                # rollout buffer or pending gradient accumulation to serialize.
                # Crossing a save interval defers it to this boundary.
                if start_step == 0 or start_step // self.eval_steps < self.global_step // self.eval_steps or self.global_step == self.max_training_steps:
                    self.evaluate(self.global_step, force=True)
                # Save after evaluation so its RNG consumption is included.
                if start_step == 0 or start_step // self.save_steps < self.global_step // self.save_steps or self.global_step == self.max_training_steps:
                    self.save_checkpoints(self.global_step, force=True)
            if exhausted:
                self.current_epoch += 1
                self.data_batches_consumed = 0
        self.on_train_end(self.global_step)
        wait_for_everyone()
        dist.destroy_process_group()
        logger.info("NFT Training Finished.")
