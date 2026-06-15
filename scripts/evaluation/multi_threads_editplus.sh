#!/usr/bin/env bash
set -euo pipefail

export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

NUM_GPUS="${NUM_GPUS:-8}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29666}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

PIPELINE="${PIPELINE:-qwenimage_edit_plus_2511}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen-Image-Edit-2511}"
IMAGE_ROOT="${IMAGE_ROOT:-dataset/MaskEdit/scene}"
LORA_MODEL="${LORA_MODEL:-pretrained_weights/lora_adapter}"
DATA_FILE="${DATA_FILE:-${IMAGE_ROOT}/testset2.jsonl}"

PROJECT_NAME="${PROJECT_NAME:-eval_editplus}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/evaluation/${PROJECT_NAME}/${RUN_TIMESTAMP}}"
EVALUATION_DIR="${EVALUATION_DIR:-${OUTPUT_DIR}/evaluations}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs}"
SHARD_DIR="${SHARD_DIR:-${OUTPUT_DIR}/data_shards}"

mkdir -p "$EVALUATION_DIR" "$LOG_DIR" "$SHARD_DIR"

"$PYTHON_BIN" scripts/utils/split_eval_data.py \
    --data-file "$DATA_FILE" \
    --num-shards "$NUM_GPUS" \
    --output-dir "$SHARD_DIR" \
    --prefix "eval_rank"

pids=()

for rank in $(seq 0 $((NUM_GPUS - 1))); do
    shard_file="${SHARD_DIR}/eval_rank_${rank}.jsonl"
    master_port=$((MASTER_PORT_BASE + rank))
    log_file="${LOG_DIR}/rank_${rank}.log"

    (
        # export CUDA_VISIBLE_DEVICES="$rank"
        torchrun \
            --nnodes=1 \
            --nproc-per-node=1 \
            --master-addr 127.0.0.1 \
            --master-port "$master_port" \
            evaluate.py \
            --config-path=configs \
            --config-name=evaluation \
            project.project_name="$PROJECT_NAME" \
            project.output_dir="$OUTPUT_DIR" \
            project.evaluation_dir="$EVALUATION_DIR" \
            project.log_dir="$LOG_DIR" \
            evalset=mask_edit \
            evalset.image_root="$IMAGE_ROOT" \
            evalset.data_file="$shard_file" \
            pipeline="$PIPELINE" \
            pipeline.pretrained_model="$BASE_MODEL" \
            adapter=lora \
            base_seed=42 \
            resume_from="$LORA_MODEL" \
            +is_fsdp_checkpoint=false \
            +lora_safetensors_dir=pretrained_weights/lora_adapter \
            cfg_scale=1.0 \
            num_inference_steps=50
    ) >"$log_file" 2>&1 &

    pids+=("$!")
    echo "Started rank ${rank} on GPU ${rank}, master_port=${master_port}, log=${log_file}"
done

failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        failed=1
    fi
done

if [[ "$failed" -ne 0 ]]; then
    echo "One or more evaluation ranks failed. Check logs in ${LOG_DIR}."
    exit 1
fi

echo "All evaluation ranks finished. Images are saved in ${EVALUATION_DIR}."
