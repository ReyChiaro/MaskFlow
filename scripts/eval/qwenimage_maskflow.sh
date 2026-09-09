#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
# Resolve once so all workers use the same Hydra run directory.
EVAL_TIMESTAMP="${EVAL_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"

.venv/bin/python3 -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc-per-node="${NPROC_PER_NODE:-8}" \
    evaluate.py \
    --config-path configs \
    --config-name eval_maskflow \
    project=evaluation \
    project.project_name=eval_QwenImage-MaskFlow-r256 \
    project.timestamp="$EVAL_TIMESTAMP" \
    pipeline=qwenimage_maskflow \
    pipeline.pretrained_model=/root/models/Qwen-Image-Edit-2511 \
    pipeline.enable_pixel_blend=false \
    evalset=mask_edit \
    evalset.data_file=dataset/MaskEdit/scene/test.jsonl \
    adapter@sft_adapter=qwenimage_lora \
    adapters.sft.path="outputs/experiments/QwenImage-MaskFlow-r256/20260908-144736/checkpoints/step-1250/lora_adapter/pytorch_lora_weights.safetensors" \
    base_seed=42 \
    num_inference_steps=50 \
    "$@"
