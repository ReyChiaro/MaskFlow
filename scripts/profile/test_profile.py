"""CPU regressions and an optional CUDA smoke test for the independent profiler.

Run: python -m unittest discover -s scripts/profile -p 'test_*.py'
"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from hydra import compose, initialize_config_dir
from PIL import Image

import profile_evaluate as entry
from utils.stage_timing import StageTimer, TimingTarget


def config():
    with initialize_config_dir(version_base=None, config_dir=str(Path(__file__).resolve().parents[2] / "configs")):
        return compose(config_name="profile_qwenimage_maskflow")


class Echo(torch.nn.Module):
    def forward(self, value, *, scale=1):
        return value * scale


class FakeVAE(Echo):
    def encode(self, value):
        return value

    def decode(self, value):
        return value


class FakePipeline:
    def __init__(self):
        self.vae = FakeVAE()
        self.text_pipeline = SimpleNamespace(text_encoder=Echo())
        self.transformer = Echo()
        self.calls = 0

    def eval_step(self, batch, **kwargs):
        from pipelines import maskflow_utils

        assert torch.is_inference_mode_enabled()
        self.calls += 1
        source = self.vae.encode(batch["conditions"]["source"])
        self.vae.encode(batch["conditions"]["mask"])
        self.text_pipeline.text_encoder(source)
        for _ in range(kwargs["num_inference_steps"]):
            source = self.transformer(source)
        mask = batch["conditions"]["mask"]
        source = maskflow_utils.poisson_refine(
            source, source, mask > 0, mask,
            poisson_lambda_e=1, poisson_lambda_s=1, poisson_num_iter=1,
            poisson_momentum=0.1, disable_progress_bar=True,
        )
        return {"output": self.vae.decode(source)}


class StageTimerTests(unittest.TestCase):
    def test_preserves_module_hooks_kwargs_outputs_and_restores_inherited_method(self):
        module = Echo()
        hook = Mock(return_value=None)
        handle = module.register_forward_hook(hook)
        original_function = module.forward.__func__
        timer = StageTimer([TimingTarget("model", module, "forward")], device="cpu", backend="cpu")
        with timer:
            module(torch.tensor(2), scale=3)  # Warmup is not counted.
            for index in range(2):
                with timer.iteration(index):
                    self.assertEqual(module(torch.tensor(2), scale=3).item(), 6)
                    module(torch.tensor(1))
        handle.remove()
        self.assertEqual(hook.call_count, 5)
        self.assertNotIn("forward", vars(module))
        self.assertIs(module.forward.__func__, original_function)
        self.assertEqual(timer.report()["stages"]["model"]["per_call"]["count"], 4)
        self.assertEqual(len(timer.iterations), 2)

    def test_exception_restores_owned_function_and_discards_partial_iteration(self):
        original = lambda value: value
        owner = SimpleNamespace(function=original)
        timer = StageTimer([TimingTarget("plain", owner, "function")], device="cpu", backend="cpu")
        with self.assertRaisesRegex(ValueError, "inference failed"):
            with timer:
                with timer.iteration(0):
                    self.assertEqual(owner.function(8), 8)
                    raise ValueError("inference failed")
        self.assertIs(owner.function, original)
        self.assertEqual(timer.calls, [])
        self.assertEqual(timer.iterations, [])

    def test_partial_install_failure_restores_earlier_patches(self):
        owner = Echo()
        timer = StageTimer([
            TimingTarget("first", owner, "forward"), TimingTarget("missing", owner, "missing"),
        ], device="cpu", backend="cpu")
        with self.assertRaises(AttributeError):
            with timer:
                pass
        self.assertNotIn("forward", vars(owner))

    def test_cuda_events_are_read_only_after_iteration_synchronization(self):
        owner = SimpleNamespace(call=lambda value: value)
        sync = Mock()
        events = []

        def event_factory(**kwargs):
            self.assertTrue(kwargs["enable_timing"])
            event = Mock()

            def elapsed(end):
                self.assertEqual(sync.call_count, 2)
                return 3.0

            event.elapsed_time.side_effect = elapsed
            events.append(event)
            return event

        with (
            patch("torch.cuda.Event", side_effect=event_factory),
            patch("torch.cuda.current_stream", return_value="stream") as stream,
            patch("torch.cuda.synchronize", sync),
        ):
            timer = StageTimer([TimingTarget("call", owner, "call")], device="cuda:0")
            with timer:
                with timer.iteration(0):
                    owner.call(1)
                    owner.call(2)
                    self.assertEqual(sync.call_count, 1)
                    for event in events:
                        event.elapsed_time.assert_not_called()
        self.assertEqual(sync.call_count, 2)
        self.assertEqual(stream.call_count, 2)
        for event in events:
            event.record.assert_called_once_with("stream")
        self.assertEqual(timer.report()["stages"]["call"]["per_iteration"]["mean_ms"], 6.0)

    def test_no_call_stages_and_csv(self):
        owner = Echo()
        timer = StageTimer([TimingTarget("unused", owner, "forward")], device="cpu", backend="cpu")
        with timer:
            with timer.iteration(0):
                pass
        stats = timer.report()["stages"]["unused"]
        self.assertEqual(stats["per_call"]["count"], 0)
        self.assertEqual(stats["per_iteration"]["count"], 1)
        self.assertEqual(stats["per_iteration"]["mean_ms"], 0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calls.csv"
            timer.write_calls(path)
            self.assertEqual(path.read_text().strip(), "iteration,stage,elapsed_ms")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_real_cuda_events_preserve_result(self):
        module = torch.nn.Linear(128, 128).cuda().eval()
        value = torch.randn(16, 128, device="cuda")
        with torch.inference_mode():
            expected = module(value)
            timer = StageTimer([TimingTarget("model", module, "forward")], device=value.device)
            with timer:
                with timer.iteration(0):
                    actual = module(value)
        torch.testing.assert_close(actual, expected)
        self.assertGreater(timer.calls[0]["elapsed_ms"], 0)
        self.assertNotIn("forward", vars(module))


class EntryTests(unittest.TestCase):
    def test_fsdp_summary_takes_max_for_each_iteration_not_sum_or_max_of_means(self):
        reports = [
            dict(rank=rank, stages={}, inference_wall={},
                 iterations=[dict(iteration=i, wall_ms=ms) for i, ms in enumerate(values)])
            for rank, values in enumerate(([10, 2], [3, 20]))
        ]
        summary = entry.summarize_ranks(reports)
        self.assertEqual(summary["slowest_rank_wall"]["mean_ms"], 15)
        self.assertEqual([row["wall_ms"] for row in summary["slowest_rank_iterations"]], [10, 20])

    def test_default_config_composes(self):
        cfg = config()
        self.assertEqual(cfg.pipeline._target_.split(".")[-1], "QwenImageMaskFlow")
        self.assertIn("outputs/profiling/", cfg.output.dir)

    def test_lora_shard_load_order(self):
        cfg = config()
        cfg.checkpoint.path = "adapter.safetensors"
        cfg.runtime.fsdp_strategy = "full_shard"
        pipe = FakePipeline()
        events = []
        pipe.configure_model = Mock(side_effect=lambda *a, **kw: events.append("configure"))
        pipe.setup_fsdp_modules = Mock(side_effect=lambda *a, **kw: events.append("shard"))
        pipe.load_pretrained_weights = Mock(side_effect=lambda *a, **kw: events.append("load"))
        with patch.object(entry, "instantiate", return_value=pipe):
            self.assertIs(entry.build_pipeline(cfg, torch.device("cpu")), pipe)
        self.assertEqual(events, ["configure", "shard", "load"])
        pipe.load_pretrained_weights.assert_called_once_with(broadcast=True)
        self.assertFalse(pipe.transformer.training)
        transform = pipe.configure_model.call_args.args[1]
        self.assertEqual(transform.keywords["lora_path"], "adapter.safetensors")

    def test_standalone_entry_reports_warmup_exclusion_and_restores_methods(self):
        cfg = config()
        cfg.runtime.warmup = 1
        cfg.runtime.repeats = 2
        cfg.runtime.num_inference_steps = 3
        cfg.preprocessing.max_resolution = 32 * 32
        pipe = FakePipeline()
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "image.png"
            Image.new("RGB", (32, 32), "white").save(image)
            cfg.input.source = cfg.input.mask = str(image)
            cfg.output.dir = str(Path(tmp) / "results")
            with (
                patch.dict("os.environ", {"RANK": "0", "WORLD_SIZE": "1", "LOCAL_RANK": "0"}),
                patch.object(entry, "build_pipeline", return_value=pipe),
                patch.object(entry, "StageTimer", side_effect=lambda targets, **kw:
                             StageTimer(targets, device="cpu", backend="cpu")),
                patch("torch.cuda.is_available", return_value=True),
                patch("torch.cuda.set_device"), patch("torch.cuda.synchronize"),
                patch("torch.cuda.get_device_name", return_value="CPU test double"),
            ):
                entry.main.__wrapped__(cfg)
            report = json.loads((Path(cfg.output.dir) / "timings_rank_0.json").read_text())
            self.assertEqual(pipe.calls, 3)
            self.assertEqual(report["stages"]["transformer"]["per_call"]["count"], 6)
            self.assertEqual(report["stages"]["vae_encode"]["per_call"]["count"], 4)
            self.assertEqual(report["stages"]["poisson"]["per_call"]["count"], 2)
            self.assertEqual(report["inference_wall"]["count"], 2)
            self.assertTrue((Path(cfg.output.dir) / "summary.json").exists())
            self.assertTrue((Path(cfg.output.dir) / "calls_rank_0.csv").exists())
            self.assertNotIn("forward", vars(pipe.transformer))
            self.assertNotIn("encode", vars(pipe.vae))


if __name__ == "__main__":
    unittest.main()
