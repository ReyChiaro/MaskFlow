#!/usr/bin/env bash
set -euo pipefail

export PIPELINE="${PIPELINE:-qwenimage_edit_plus_2511}"
export PROJECT_NAME="${PROJECT_NAME:-eval_editplus}"

bash scripts/evaluation/multi_threads.sh "$@"
