#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# Reuse MaskFlow evaluation settings; all GPUs jointly infer each batch.
# Pass the trained checkpoint with adapters.sft.path=...
exec bash scripts/eval/flux2_maskflow.sh \
    fsdp_strategy=full_shard \
    project.project_name=eval_FLUX2dev-MaskFlow-FSDP-r256 \
    "$@"
