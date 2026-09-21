#!/usr/bin/env bash
# Independent benchmark launcher. All arguments are Hydra overrides.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PROFILE_TIMESTAMP="${PROFILE_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)-$$}"

PROFILE_PYTHON="${PROFILE_PYTHON:-python}"
PROFILE_NPROC="${NPROC_PER_NODE:-1}"
if ! [[ "$PROFILE_NPROC" =~ ^[1-9][0-9]*$ ]]; then
    echo "NPROC_PER_NODE must be a positive integer" >&2
    exit 2
fi
if [[ "$PROFILE_NPROC" == "1" ]]; then
    exec "$PROFILE_PYTHON" profile_evaluate.py "$@"
else
    exec "$PROFILE_PYTHON" -m torch.distributed.run \
        --standalone --nnodes=1 --nproc-per-node="$PROFILE_NPROC" \
        profile_evaluate.py runtime.fsdp_strategy=full_shard "$@"
fi
