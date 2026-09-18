"""NFT CPU regressions; distributed tests use two local Gloo ranks, no downloads.

Run: .venv/bin/python -m unittest discover -s tests -p 'test_diffusion_nft.py'
"""
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from diffusers.loaders.peft import PeftAdapterMixin
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.utils.checkpoint import checkpoint

from data_module.sampler import CheckpointDistributedSampler
from models.rewards.rewards import RewardModel, Rewards
from models.rewards.clip import CLIPReward
from models.rewards.dinov2 import DINOv2Reward
from pipelines.qwenimage.qwenimage_maskflow import QwenMaskFlowCFGBranch, QwenMaskFlowForwardOutput
from schedulers.mask_flow import MaskFlowScheduler
from trainer.diffusion_nft import DiffusionNFTTrainer
from trainer.lora_utils import add_trainable_lora
from utils.config import register_distributed_timestamp_resolver


class TinyTransformer(torch.nn.Module, PeftAdapterMixin):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(2, 2, bias=False)

    def forward(self, x):
        return checkpoint(self.projection, x, use_reentrant=False) if self.training else self.projection(x)


def make_trainer(**kwargs):
    cfg = OmegaConf.create({"r": 1, "lora_alpha": 1, "lora_dropout": 0,
                           "adapter_name": "nft", "target_modules": ["projection"]})
    options = dict(lora_configs=cfg, reward_configs={}, mixed_precision="no",
                   num_inference_steps=3, train_timesteps=2)
    options.update(kwargs)
    trainer = DiffusionNFTTrainer(**options)
    trainer.device = torch.device("cpu")
    trainer._train_dtype = torch.float32
    trainer.world_size = 1
    trainer.global_rank = 0
    trainer.generator = torch.Generator().manual_seed(17)
    trainer.rng = random.Random(17)
    trainer.micro_step = 0
    trainer.global_step = 0
    trainer.current_epoch = 0
    torch.manual_seed(23)
    model = TinyTransformer()
    add_trainable_lora(model, cfg, torch.float32)
    trainer.pipe = SimpleNamespace(
        transformer=model, fsdp_modules=[model],
        scheduler=MaskFlowScheduler(unmask_with="noisy_target"),
        denoise=lambda hidden_states, **kwargs: model(hidden_states),
    )
    trainer.actor_params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    trainer.pipe.trainable_params = list(trainer.actor_params.values())
    trainer.old_params = {n: p.detach().clone() for n, p in trainer.actor_params.items()}
    trainer.reference_params = {n: p.detach().clone() for n, p in trainer.actor_params.items()}
    trainer.optimizer = torch.optim.SGD(trainer.actor_params.values(), lr=0.01, momentum=0.9)
    trainer.lr_scheduler = None
    return trainer


def sample(batch_size=2):
    branch = QwenMaskFlowCFGBranch(torch.zeros(batch_size, 1, 2), torch.ones(batch_size, 1), [], [])
    inputs = QwenMaskFlowForwardOutput(
        cfg_branches={"pm": branch}, conditions=[torch.zeros(batch_size, 2, 2)],
        mask_latents=torch.ones(batch_size, 2, 2),
    )
    x0 = torch.arange(batch_size * 4).reshape(batch_size, 2, 2).float() / 8
    return inputs, x0, torch.ones(batch_size), torch.tensor([1.0, 0.6, 0.2])


def distributed_worker(rank, init_file, output_dir, strategy):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
    trainer = make_trainer(fsdp_strategy=strategy)
    trainer.world_size = 2
    trainer.global_rank = rank
    model = trainer.pipe.transformer
    if strategy == "full_shard":
        mesh = init_device_mesh("cpu", (2,))
        fully_shard(model.projection, mesh=mesh, reshard_after_forward=True)
        fully_shard(model, mesh=mesh, reshard_after_forward=False)
        trainer.actor_params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        trainer.pipe.trainable_params = list(trainer.actor_params.values())
        trainer.old_params = {n: p.detach().clone() for n, p in trainer.actor_params.items()}
        trainer.reference_params = {n: p.detach().clone() for n, p in trainer.actor_params.items()}
        trainer.optimizer = torch.optim.SGD(trainer.actor_params.values(), lr=0.01, momentum=0.9)
    # Different local data, three optimizer updates, two accumulated micro-batches
    # each with two times. Old/ref switches occur between backwards with grads pending.
    for step in range(3):
        trainer.pipe.transformer.train()
        for micro in range(2):
            inputs, x0, advantage, grid = sample()
            x0 = x0 + rank * 0.3 + micro * 0.1
            trainer._train_micro_batch((inputs, x0, advantage, grid), micro == 1, 2)
        trainer.sync_gradients()
        torch.nn.utils.clip_grad_norm_(list(trainer.actor_params.values()), trainer.max_grad_norm)
        trainer.optimizer.step()
        trainer.optimizer.zero_grad()
        trainer._sync_old()
    trainer._reshard_actor()
    values = {name: p.full_tensor() if hasattr(p, "full_tensor") else p.detach()
              for name, p in trainer.actor_params.items()}
    if rank == 0:
        torch.save(values, Path(output_dir) / f"{strategy}.pt")
    # Exercise actual checkpoint methods, including actor, role snapshots,
    # optimizer momentum and rank-specific RNG. Only CUDA RNG is mocked on CPU.
    checkpoint_dir = Path(output_dir) / strategy
    checkpoint_dir.mkdir(exist_ok=True)
    with patch("torch.cuda.get_rng_state", return_value=torch.get_rng_state()):
        trainer.save_model_checkpoints(checkpoint_dir)
    trainer.save_optimizer_checkpoints(checkpoint_dir)
    expected = {n: p.clone() for n, p in trainer.old_params.items()}
    for p in trainer.old_params.values():
        p.zero_()
    momentum = [trainer.optimizer.state[p]["momentum_buffer"].clone() for p in trainer.actor_params.values()]
    for state in trainer.optimizer.state.values():
        state["momentum_buffer"].zero_()
    with patch("torch.cuda.set_rng_state"):
        trainer.load_model_checkpoints(checkpoint_dir)
    trainer.load_optimizer_checkpoints(checkpoint_dir)
    for name, p in trainer.old_params.items():
        actual = p.to_local() if hasattr(p, "to_local") else p
        wanted = expected[name].to_local() if hasattr(p, "to_local") else expected[name]
        torch.testing.assert_close(actual, wanted)
    for p, wanted in zip(trainer.actor_params.values(), momentum, strict=True):
        actual = trainer.optimizer.state[p]["momentum_buffer"]
        if hasattr(actual, "to_local"):
            actual, wanted = actual.to_local(), wanted.to_local()
        torch.testing.assert_close(actual, wanted)
    dist.destroy_process_group()


class NFTTests(unittest.TestCase):
    def test_config_composes_without_initializing_models(self):
        register_distributed_timestamp_resolver()
        with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / "configs"), version_base="1.2"):
            for reward in ("clip", "clip_dinov2"):
                config = compose(config_name="train_nft", overrides=[f"reward={reward}"])
                trainer = instantiate(config.trainer)
                self.assertEqual(trainer.lora_configs.adapter_name, "nft")
                self.assertEqual(trainer.group_size, trainer.gradient_accumulation_steps)

    def test_advantages_group_by_editing_input_and_handle_ties(self):
        trainer = make_trainer(global_reward_std=False)
        rewards = [torch.tensor([[1., 100., 4.], [3., 104., 4.]])]
        advantages = trainer._advantages(rewards)[0]
        self.assertTrue(torch.all(advantages[0, :2] < 0))
        self.assertTrue(torch.all(advantages[1, :2] > 0))
        torch.testing.assert_close(advantages[:, 2], torch.zeros(2))
        torch.testing.assert_close(advantages.mean(0), torch.zeros(3))

    def test_policy_sign_and_zero_advantage_cancellation(self):
        trainer = make_trainer(reference_weight=0)
        old = torch.zeros(1, 2, 2)
        xt = torch.ones_like(old)
        for advantage, expected_sign in [(1., -1), (-1., 1), (0., 0)]:
            actor = old.clone().requires_grad_()
            loss = trainer._policy_loss(actor, old, xt, torch.zeros_like(old), torch.tensor([0.5]), torch.tensor([advantage]))
            loss.backward()
            self.assertEqual(actor.grad.mean().sign().item(), expected_sign)

    def test_role_switch_restores_parameters_on_error_and_preserves_grads(self):
        trainer = make_trainer()
        actor = {n: p.clone() for n, p in trainer.actor_params.items()}
        for p in trainer.actor_params.values():
            p.grad = torch.ones_like(p)
        old = {n: torch.full_like(p, 2.) for n, p in trainer.actor_params.items()}
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            with trainer._use_weights(old):
                for p in trainer.actor_params.values():
                    torch.testing.assert_close(p, torch.full_like(p, 2.))
                raise RuntimeError("intentional")
        for name, p in trainer.actor_params.items():
            torch.testing.assert_close(p, actor[name])
            torch.testing.assert_close(p.grad, torch.ones_like(p))
            self.assertTrue(p.requires_grad)

    def test_rollout_samples_independent_endpoints_and_keeps_group_scores(self):
        trainer = make_trainer(group_size=3)
        inputs, x0, _, _ = sample()
        inputs.noise, inputs.height, inputs.width = torch.zeros_like(x0), 2, 2
        data = SimpleNamespace(mask=torch.ones(2, 1, 2, 2), raw_source=torch.zeros(2, 3, 2, 2))
        trainer.pipe.preprocess_inputs = lambda batch: data
        trainer.pipe.prepare_eval_inputs = lambda data, scale: inputs
        trainer.pipe.enable_poisson_infer = False
        trainer.pipe.enable_pixel_blend = True
        trainer.pipe.vae_scale_factor = 1
        trainer.pipe.decode_image = lambda xt: xt.unsqueeze(1).expand(-1, 3, -1, -1)
        trainer.reward_fn = lambda images, batch: images.flatten(1).mean(1)
        trainer.pipe.transformer.eval()
        with (
            trainer._use_weights(trainer.old_params),
            patch("trainer.diffusion_nft.QwenImageEditPlusPipeline._unpack_latents", side_effect=lambda xt, *args: xt),
        ):
            _, endpoints, rewards, grid = trainer._rollout_group({"prompt": ["same", "same"]})
        self.assertEqual(rewards.shape, (3, 2))
        self.assertEqual(len(grid), trainer.num_inference_steps)
        self.assertFalse(torch.equal(endpoints[0], endpoints[1]))
        self.assertFalse(any(xt.requires_grad for xt in endpoints))
        for k, xt in enumerate(endpoints):
            torch.testing.assert_close(rewards[k], xt.flatten(1).mean(1))

    def test_actual_loop_counts_rounds_and_tail_accumulation(self):
        trainer = make_trainer(group_size=3, rollout_batches_per_round=2, inner_epochs=2,
                               gradient_accumulation_steps=4, max_training_steps=7)
        batches = [{"prompt": [str(i)]} for i in range(3)]
        trainer.train_loader = batches
        trainer.train_sampler = CheckpointDistributedSampler(batches, num_replicas=1, rank=0)
        trainer.init_everything = lambda: None
        trainer.on_train_end = lambda step: None
        trainer.evaluate = lambda *args, **kwargs: None
        saved = []
        trainer.save_checkpoints = lambda step, **kwargs: saved.append(step)
        rollouts = []

        def rollout(batch):
            rollouts.append((trainer.global_step, batch["prompt"][0]))
            inputs, x0, _, grid = sample()
            endpoints = [x0 + 0.1 * k for k in range(trainer.group_size)]
            return inputs, endpoints, torch.tensor([[0., 0.], [1., 2.], [2., 4.]]), grid

        trainer._rollout_group = rollout
        with patch("trainer.diffusion_nft.dist.destroy_process_group"):
            trainer.train()
        self.assertEqual(trainer.global_step, 7)
        self.assertEqual(trainer.rollout_step, 3)
        self.assertEqual(trainer.micro_step, 12 + 6 + 4)
        self.assertEqual([step for step, _ in rollouts], [0, 0, 4, 6, 6])
        self.assertEqual(saved[-1], 7)

    def test_two_rank_fsdp_matches_unsharded_updates_and_snapshot_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            for strategy in ("no_shard", "full_shard"):
                mp.spawn(distributed_worker, args=(str(Path(directory) / f"init-{strategy}"), directory, strategy),
                         nprocs=2, join=True)
            dense = torch.load(Path(directory) / "no_shard.pt", weights_only=True)
            sharded = torch.load(Path(directory) / "full_shard.pt", weights_only=True)
            for name in dense:
                torch.testing.assert_close(dense[name], sharded[name], rtol=1e-5, atol=1e-6)


class ConstantReward(RewardModel):
    def __call__(self, images, batch):
        return torch.arange(len(images), device=images.device).float()


class RewardTests(unittest.TestCase):
    def test_clip_and_dino_keep_pair_alignment_across_micro_batches(self):
        class Inputs(dict):
            def to(self, device):
                return self

        def processor(images, **kwargs):
            return Inputs(pixel_values=torch.stack(list(images)))

        def clip_forward(pixel_values):
            features = torch.nn.functional.normalize(pixel_values.flatten(1), dim=-1)
            return SimpleNamespace(image_embeds=features, text_embeds=features)

        def dino_forward(pixel_values):
            return SimpleNamespace(last_hidden_state=pixel_values.flatten(1).unsqueeze(1))

        images = torch.arange(1, 37).reshape(3, 3, 2, 2).float() / 36
        clip = object.__new__(CLIPReward)
        clip.device, clip.batch_size, clip.prompt_key = "cpu", 2, "edit_instruction"
        clip.processor = processor
        clip.model = SimpleNamespace()
        # A callable stub with the same model configuration used by preprocessing.
        class ClipModel:
            config = SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=77))
            __call__ = staticmethod(clip_forward)
        clip.model = ClipModel()
        torch.testing.assert_close(clip(images, {"edit_instruction": ["a", "b", "c"]}), torch.ones(3))
        dino = object.__new__(DINOv2Reward)
        dino.device, dino.batch_size, dino.reference = "cpu", 2, "target"
        dino.processor, dino.model = processor, dino_forward
        torch.testing.assert_close(dino(images, {"target": images.clone()}), torch.ones(3))
        with self.assertRaisesRegex(ValueError, "target"):
            dino(images, {"target": None})

    def test_ensemble_preserves_per_image_scores_and_weights(self):
        config = OmegaConf.create({"a": {"weight": 2., "model": {}}, "b": {"weight": -0.5, "model": {}}})
        with patch("models.rewards.rewards.instantiate", return_value=ConstantReward()):
            rewards = Rewards(models=config)
        scores = rewards(torch.zeros(3, 3, 2, 2), {})
        torch.testing.assert_close(scores, torch.tensor([0., 1.5, 3.]))

    def test_ensemble_rejects_scalar_reward(self):
        with patch("models.rewards.rewards.instantiate", return_value=ConstantReward()):
            rewards = Rewards(models=OmegaConf.create({"bad": {"weight": 1., "model": {}}}))
        rewards.reward_models["bad"] = lambda **kwargs: torch.tensor(1.)
        with self.assertRaisesRegex(ValueError, "shape"):
            rewards(torch.zeros(3, 3, 2, 2), {})


if __name__ == "__main__":
    unittest.main()
