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
    --config-name eval_flux2 \
    project.timestamp="$EVAL_TIMESTAMP" \
    "$@"
