"""Temporary method instrumentation; CUDA events are resolved once per iteration."""

import csv
import functools
import math
import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class TimingTarget:
    name: str
    owner: object
    attribute: str


def summarize(values):
    """Summarize milliseconds, including zero-call iterations when supplied."""
    values = sorted(values)
    if not values:
        return dict(count=0, total_ms=0.0, mean_ms=0.0, median_ms=0.0, p95_ms=0.0)
    return dict(
        count=len(values), total_ms=sum(values), mean_ms=statistics.mean(values),
        median_ms=statistics.median(values), p95_ms=values[math.ceil(0.95 * len(values)) - 1],
    )


class StageTimer:
    """Single-threaded, process-local patches restored even after exceptions.

    CUDA timings are current-stream elapsed intervals, not sums of kernel times.
    CPU mode exists for CPU workloads/tests and is never selected automatically.
    """

    def __init__(self, targets, *, device, backend="cuda"):
        if backend not in {"cuda", "cpu"}:
            raise ValueError("backend must be cuda or cpu")
        self.targets = list(targets)
        keys = [(id(t.owner), t.attribute) for t in self.targets]
        if len(keys) != len(set(keys)):
            raise ValueError("Cannot patch the same attribute twice")
        if len({t.name for t in self.targets}) != len(self.targets):
            raise ValueError("Stage names must be unique")
        self.device = torch.device(device)
        if backend == "cuda" and self.device.type != "cuda":
            raise ValueError("CUDA timing requires a CUDA device")
        self.backend = backend
        self.calls = []
        self.iterations = []
        self._patches = []
        self._pending = None
        self._entered = False

    def _wrap(self, original, name):
        @functools.wraps(original)
        def measured(*args, **kwargs):
            if self._pending is None:
                return original(*args, **kwargs)
            if self.backend == "cuda":
                stream = torch.cuda.current_stream(self.device)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record(stream)
            else:
                start = time.perf_counter()
            result = original(*args, **kwargs)
            if self.backend == "cuda":
                end.record(stream)
            else:
                end = time.perf_counter()
            self._pending.append((name, start, end))
            return result
        return measured

    def __enter__(self):
        if self._entered:
            raise RuntimeError("Timer context is already installed")
        self._entered = True
        try:
            for target in self.targets:
                original = getattr(target.owner, target.attribute)
                if not callable(original):
                    raise TypeError(f"{target.name} is not callable")
                # Restore attribute ownership as well as its effective value.
                had_own = target.attribute in vars(target.owner)
                own_value = vars(target.owner).get(target.attribute)
                setattr(target.owner, target.attribute, self._wrap(original, target.name))
                self._patches.append((target.owner, target.attribute, had_own, own_value))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for owner, attribute, had_own, value in reversed(self._patches):
            if had_own:
                setattr(owner, attribute, value)
            else:
                delattr(owner, attribute)
        self._patches.clear()
        self._pending = None
        self._entered = False

    @contextmanager
    def iteration(self, index):
        """Measure one complete inference; warmups run outside this context."""
        if not self._entered or self._pending is not None:
            raise RuntimeError("Install timer first; iterations cannot be nested")
        if index in {row["iteration"] for row in self.iterations}:
            raise ValueError("Iteration IDs must be unique")
        if self.backend == "cuda":
            torch.cuda.synchronize(self.device)
        self._pending = []
        started = time.perf_counter()
        try:
            yield
            if self.backend == "cuda":
                torch.cuda.synchronize(self.device)
            wall_ms = (time.perf_counter() - started) * 1000
            # Resolve events only after inference has completed, outside wall timing.
            rows = [
                dict(iteration=index, stage=name,
                     elapsed_ms=start.elapsed_time(end) if self.backend == "cuda" else (end - start) * 1000)
                for name, start, end in self._pending
            ]
            self.calls.extend(rows)
            self.iterations.append(dict(iteration=index, wall_ms=wall_ms))
        finally:
            # Discard partial iterations rather than reporting incomplete inference.
            self._pending = None

    def report(self):
        stages = {}
        for target in self.targets:
            calls = [row for row in self.calls if row["stage"] == target.name]
            totals = [
                sum(row["elapsed_ms"] for row in calls if row["iteration"] == iteration["iteration"])
                for iteration in self.iterations
            ]
            stages[target.name] = dict(
                per_call=summarize([row["elapsed_ms"] for row in calls]),
                per_iteration=summarize(totals),
            )
        return dict(
            backend=self.backend, stages=stages, iterations=list(self.iterations),
            inference_wall=summarize([row["wall_ms"] for row in self.iterations]),
        )

    def write_calls(self, path):
        with Path(path).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["iteration", "stage", "elapsed_ms"])
            writer.writeheader()
            writer.writerows(self.calls)


def maskflow_targets(pipe):
    from pipelines import maskflow_utils

    return [
        TimingTarget("vae_encode", pipe.vae, "encode"),
        TimingTarget("text_encoder", pipe.text_pipeline.text_encoder, "forward"),
        TimingTarget("transformer", pipe.transformer, "forward"),
        TimingTarget("poisson", maskflow_utils, "poisson_refine"),
        TimingTarget("vae_decode", pipe.vae, "decode"),
    ]
