"""CPU regressions for evaluation sharding, LoRA config, and result saving.

Run with: .venv/bin/python -m unittest discover -s tests -p 'test_evaluate.py'
"""
import os
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset

import evaluate


class ToyDataset(Dataset):
    def __init__(self, size):
        self.samples = [{"target": f"target/sample_{i}.png"} for i in range(size)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return {"image_name": f"sample_{index}", "prompt": "edit", "edit_instruction": "position edit"}


class EvaluationTests(unittest.TestCase):
    def test_each_sample_saved_once_with_uneven_shards_and_batches(self):
        for size, world_size in [(7, 3), (2, 4), (5, 1), (0, 3)]:
            with self.subTest(size=size, world_size=world_size), tempfile.TemporaryDirectory() as tmp:
                dataset = ToyDataset(size)
                seen = []
                cfg = OmegaConf.create({
                    "base_seed": 42, "eval_seed": 123, "weight_dtype": "fp32", "batch_size_per_process": 2,
                    "num_workers": 0, "evalset": {}, "pipeline": {}, "adapters": {},
                    "text_cfg_scale": 4.0, "num_inference_steps": 2,
                    "eval_with_position_prompt": True,
                })
                pipe = SimpleNamespace(
                    transformer=torch.nn.Linear(1, 1), vae=torch.nn.Identity(),
                    text_pipeline=SimpleNamespace(text_encoder=torch.nn.Identity()),
                    load_pretrained_weights=Mock(),
                )

                def eval_step(batch, **kwargs):
                    self.assertEqual(kwargs["seed"], 123)
                    self.assertTrue(torch.is_inference_mode_enabled())
                    self.assertFalse(pipe.transformer.training)
                    self.assertEqual(batch["prompt"], batch["edit_instruction"])
                    seen.extend(batch["image_name"])
                    count = len(batch["image_name"])
                    return {"output": torch.ones(count, 3, 4, 4), "mask": torch.ones(count, 1, 4, 4)}

                pipe.eval_step = eval_step
                sizes = []
                for rank in range(world_size):
                    before = len(seen)
                    objects = [dataset, pipe] if rank < size else [dataset]
                    with patch.object(evaluate, "instantiate", side_effect=objects):
                        evaluate.run_evaluation(cfg, torch.device("cpu"), rank, world_size, Path(tmp))
                    sizes.append(len(seen) - before)
                self.assertEqual(Counter(seen), Counter(f"sample_{i}" for i in range(size)))
                self.assertLessEqual(max(sizes) - min(sizes), 1)
                self.assertEqual(len(list((Path(tmp) / "predictions").glob("*.png"))), size)
                self.assertEqual(len(list((Path(tmp) / "mask").glob("*.png"))), size)

    def test_lora_config_interpolation_and_missing_path(self):
        cfg = OmegaConf.create({
            "sft_adapter": {"adapter_name": "maskflow"},
            "adapters": {"sft": {"path": "weights.safetensors", "cfg": "${sft_adapter}", "lora_scale": 0.25}},
        })
        pipe = SimpleNamespace(transformer=Mock())
        with patch.object(evaluate, "merge_lora") as merge:
            evaluate.load_lora_adapters(pipe, cfg.adapters)
            merge.assert_called_once_with(pipe.transformer, "weights.safetensors", "maskflow", lora_scale=0.25)
            cfg.adapters.sft.path = None
            with self.assertRaisesRegex(ValueError, "adapters.sft.path"):
                evaluate.load_lora_adapters(pipe, cfg.adapters)

    def test_worker_uses_broadcast_directory_and_cleans_up_on_failure(self):
        cfg = OmegaConf.create({"project": {"evaluation_dir": "worker-local-time"}})

        def broadcast(value, **kwargs):
            value[0] = "/tmp/shared-evaluation"

        with (
            patch.dict(os.environ, {"WORLD_SIZE": "3", "RANK": "1", "LOCAL_RANK": "1"}),
            patch.object(torch.cuda, "set_device") as set_device,
            patch.object(evaluate.dist, "init_process_group") as init,
            patch.object(evaluate.dist, "broadcast_object_list", side_effect=broadcast),
            patch.object(evaluate.dist, "is_initialized", return_value=True),
            patch.object(evaluate.dist, "destroy_process_group") as destroy,
            patch.object(evaluate, "run_evaluation", side_effect=RuntimeError("inference failed")) as run,
        ):
            with self.assertRaisesRegex(RuntimeError, "inference failed"):
                evaluate.evaluate.__wrapped__(cfg)
            set_device.assert_called_once_with(torch.device("cuda:1"))
            init.assert_called_once_with("nccl")
            run.assert_called_once_with(cfg, torch.device("cuda:1"), 1, 3, Path("/tmp/shared-evaluation"))
            destroy.assert_called_once()


if __name__ == "__main__":
    unittest.main()
