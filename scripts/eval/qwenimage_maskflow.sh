#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
# Resolve once so all workers use the same Hydra run directory.
EVAL_TIMESTAMP="${EVAL_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"

python -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc-per-node="${NPROC_PER_NODE:-8}" \
    evaluate.py \
    --config-path configs \
    --config-name eval_qwenimage_maskflow \
    project=evaluation \
    project.project_name=eval_QwenImage-MaskFlow-r256 \
    project.timestamp="$EVAL_TIMESTAMP" \
    pipeline=qwenimage_maskflow \
    pipeline.pretrained_model=/root/intern-xr/models/Qwen-Image-Edit-2511 \
    pipeline.rescale_cfg=true \
    pipeline.enable_masked_loss=true \
    pipeline.enable_pixel_blend=false \
    pipeline.enable_poisson_train=true \
    pipeline.enable_poisson_infer=true \
    pipeline.poisson_lambda_e=1.0 \
    pipeline.poisson_lambda_s=1.0 \
    pipeline.poisson_num_iter=50 \
    pipeline.poisson_momentum=0.1 \
    evalset=hf_mask_edit \
    evalset.subsets=scene \
    evalset.split=test \
    evalset.load_end=1.0 \
    adapter@sft_adapter=qwenimage_lora \
    adapters.sft.path="outputs/experiments/QwenImage-MaskFlow-r256/20260913-030556/checkpoints/step-1250/lora_adapter/pytorch_lora_weights.safetensors" \
    base_seed=42 \
    weight_dtype=bf16 \
    batch_size_per_process=1 \
    num_workers=8 \
    text_cfg_scale=4.0 \
    mask_cfg_scale=1.25 \
    interaction_cfg_scale=4.25 \
    num_inference_steps=50 \
    eval_with_position_prompt=false \
