# QwenImage MaskFlow module timing

`profile_evaluate.py` is a standalone CUDA benchmark. It does not import the
existing evaluation or inference entry points, datasets, or their launchers.
It reuses the production pipeline and model configuration without modifying
their implementation. Supply one aligned source/mask pair and an edit prompt;
each iteration repeats that workload with the same seed. Batch size replicates
the pair to measure the requested batch shape, rather than sampling a dataset.

## Run

Run from the repository root in the project's CUDA Python environment:

```bash
bash scripts/profile/qwenimage_maskflow.sh \
  pipeline.pretrained_model=/path/to/Qwen-Image-Edit-2511 \
  checkpoint.path=/path/to/pytorch_lora_weights.safetensors \
  input.source=/path/to/source.png \
  input.mask=/path/to/mask.png \
  'input.prompt="Replace the red car with a blue car"' \
  runtime.warmup=2 runtime.repeats=5
```

Omit `checkpoint.path` to benchmark base model weights. The mask must have the
same dimensions as the source; white indicates the region to edit. Image load,
shared crop/resize, model loading, LoRA merging, and output saving happen outside
the timed inference. Pipeline-internal preprocessing remains in inference wall
time. `preprocessing.max_resolution` sets the image pixel-area budget (default
1024 × 1024). `runtime.batch_size` defaults to 1. `output.save_image=true` saves
the last measured output, after timing has finished.

For FSDP2, all workers run the same batch and repeat count:

```bash
NPROC_PER_NODE=8 bash scripts/profile/qwenimage_maskflow.sh \
  pipeline.pretrained_model=/path/to/Qwen-Image-Edit-2511 \
  checkpoint.path=/path/to/pytorch_lora_weights.safetensors \
  input.source=/path/to/source.png input.mask=/path/to/mask.png \
  'input.prompt="Replace the red car with a blue car"'
```

The launcher selects `full_shard` for multiple processes; single-process runs
default to `no_shard`. Set `PROFILE_PYTHON` to choose the Python executable.
Other useful overrides are `pipeline.enable_poisson_infer=false`,
`pipeline.poisson_steps=[0.0,1.0]`, `pipeline.poisson_num_iter=50`,
`runtime.num_inference_steps=50`, `runtime.text_cfg_scale=4.0`, and
`runtime.mask_cfg_scale=1.0`. Velocity visualization is rejected because it
would introduce an additional full trajectory. Both the Python entry and the
launcher support `--help` and `--cfg job` without loading a model.

## Measurement boundaries

| Stage | Temporarily wrapped method |
| --- | --- |
| `vae_encode` | `pipe.vae.encode` |
| `text_encoder` | `pipe.text_pipeline.text_encoder.forward` |
| `transformer` | `pipe.transformer.forward` |
| `poisson` | `pipelines.maskflow_utils.poisson_refine` |
| `vae_decode` | `pipe.vae.decode` |

Each call records CUDA events on the current stream for the selected device.
There is no per-call synchronization. Each measured iteration synchronizes
before inference and after inference, then resolves its events. The separately
reported inference wall time includes CPU work, GPU waiting, and instrumentation
overhead, but excludes the pre-inference synchronization and FSDP barrier.
Warmups run with recording disabled. Patches are restored on context exit,
including exceptions; incomplete iterations are discarded. This instrumentation
assumes single-threaded inference per process: the Poisson patch is module-wide
within that process while installed.

These are **stream elapsed intervals**, not pure GPU kernel sums. CPU launch
gaps can be included; unrelated streams are not automatically covered. A module
`forward` boundary excludes its own outer pre/post hooks; specifically, root
FSDP hook work is not fully attributed to that module's forward. Nested FSDP
block work may occur inside it. Use inference wall time to compare complete
FSDP runs; do not interpret stage timings as a complete communication breakdown.
VAE encode excludes latent sampling/normalization, text encoder excludes prompt
preprocessing, Poisson excludes surrounding prediction conversions, and VAE
decode excludes pipeline image postprocessing. Stage totals therefore need not
sum to inference wall time. No backward/optimizer timing is performed.

## Outputs

The default directory is `outputs/profiling/qwenimage_maskflow/<timestamp>`;
override it with `output.dir=/path/to/run`.

- `timings_rank_N.json`: resolved configuration, device, input shape, per-stage
  per-call and per-iteration statistics, and per-iteration inference wall times.
- `calls_rank_N.csv`: one row per completed module call, in completion order,
  with iteration ID, stage, and elapsed milliseconds.
- `summary.json`: per-rank stage summaries and the slowest rank's inference wall
  time for each aligned iteration, summarized across repeats. Rank times are
  never summed because FSDP workers collaborate on the same batch.

Statistics include count, total, mean, median and nearest-rank P95. Per-call
statistics divide by actual calls; per-iteration statistics sum calls in each
iteration, including zero for stages not called. For example, disabling Poisson
reports zero calls rather than omitting the stage. CFG changes text encoder and
transformer call counts; record both counts and elapsed times when comparing.
Console stage summaries show rank 0, while JSON preserves every rank.

## Verification

```bash
python -m unittest discover -s scripts/profile -p 'test_*.py'
bash -n scripts/profile/qwenimage_maskflow.sh
python profile_evaluate.py --cfg job
```

CPU regressions cover patch restoration, module hooks, exception handling,
deferred event resolution, configuration, and the independent reporting flow.
The CUDA smoke test is skipped when CUDA is unavailable. Actual model/FSDP
performance requires running the benchmark on the target GPU hardware.
