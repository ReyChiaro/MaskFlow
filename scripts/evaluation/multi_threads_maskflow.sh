#!/usr/bin/env bash
set -euo pipefail

export PIPELINE="${PIPELINE:-qwenimage_mask_flow}"
export PROJECT_NAME="${PROJECT_NAME:-eval_maskflow_lora}"

bash scripts/evaluation/multi_threads.sh "$@"
