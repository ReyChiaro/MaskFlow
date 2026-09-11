#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
# Resolve once so all evaluation workers share a run directory.
EVAL_TIMESTAMP="${EVAL_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"

.venv/bin/python3 -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc-per-node="${NPROC_PER_NODE:-8}" \
    evaluate.py \
    --config-path configs \
    --config-name eval_flux2_maskflow \
    project=evaluation \
    project.project_name=eval_FLUX2dev-ImgEdit-r256 \
    project.timestamp="$EVAL_TIMESTAMP" \
    pipeline=flux2_maskflow \
    pipeline.pretrained_model=/root/models/FLUX.2-dev \
    pipeline.enable_pixel_blend=false \
    evalset=imgedit_benchmark \
    evalset.data_file=dataset/ImgEdit-Benchmark/test.jsonl \
    adapter@sft_adapter=flux2_lora \
    adapters.sft.path="outputs/experiments/FLUX2dev-MaskFlow-r256/20260910-085019/checkpoints/step-1250/lora_adapter/pytorch_lora_weights.safetensors" \
    base_seed=42 \
    num_inference_steps=28 \
    eval_with_position_prompt=false \
    "$@"
