#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

.venv/bin/python3 -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc-per-node="${NPROC_PER_NODE:-8}" \
    finetune.py \
    --config-path configs \
    --config-name sft_flux2 \
    "$@"
