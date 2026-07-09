#!/usr/bin/env bash
set -euo pipefail

EXPERIMENTS="${EXPERIMENTS:-20260706-121217/baseline+noisy_source}"
GPU_LIST="${GPUS:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for --gpus. Usage: bash calculate_metrics.sh --gpus 0,1"
                exit 1
            fi
            GPU_LIST="$2"
            shift 2
            ;;
        *)
            GPU_LIST="${GPU_LIST:+${GPU_LIST} }$1"
            shift
            ;;
    esac
done

GPU_LIST="${GPU_LIST:-0}"
GPU_LIST="${GPU_LIST//,/ }"
read -r -a GPU_IDS <<< "${GPU_LIST}"

if [[ ${#GPU_IDS[@]} -eq 0 ]]; then
    echo "No GPU specified. Usage: GPUS=0,1 bash calculate_metrics.sh"
    exit 1
fi

run_foreground() {
    local gpu="$1"
    echo "Running foreground metrics on GPU ${gpu}"
    CUDA_VISIBLE_DEVICES="${gpu}" python calculate_metrics.py \
        --stage "foreground" \
        --source "evaluations/maskflow_ablations/${EXPERIMENTS}/rank_*/evaluations/output" \
        --target "evaluations/target" \
        --mask "evaluations/maskflow_ablations/${EXPERIMENTS}/rank_*/evaluations/mask" \
        --rank 0 \
        --save-to "evaluations/maskflow_ablations/${EXPERIMENTS}/metrics-foreground.json"
}

run_background() {
    local gpu="$1"
    echo "Running background metrics on GPU ${gpu}"
    CUDA_VISIBLE_DEVICES="${gpu}" python calculate_metrics.py \
        --stage "background" \
        --source "evaluations/maskflow_ablations/${EXPERIMENTS}/rank_*/evaluations/output" \
        --target "evaluations/source" \
        --mask "evaluations/maskflow_ablations/${EXPERIMENTS}/rank_*/evaluations/mask" \
        --rank 0 \
        --save-to "evaluations/maskflow_ablations/${EXPERIMENTS}/metrics-background.json"
}

if [[ ${#GPU_IDS[@]} -eq 1 ]]; then
    run_foreground "${GPU_IDS[0]}"
    run_background "${GPU_IDS[0]}"
else
    run_foreground "${GPU_IDS[0]}" &
    foreground_pid=$!

    run_background "${GPU_IDS[1]}" &
    background_pid=$!

    status=0
    wait "${foreground_pid}" || status=$?
    wait "${background_pid}" || status=$?
    exit "${status}"
fi
