#!/usr/bin/env bash
set -euo pipefail

# Unified multi-GPU evaluation launcher.
# Common overrides can be set through environment variables; pipeline-specific
# or experimental overrides can be appended as normal Hydra args:
#   PIPELINE=qwenimage_mask_flow bash scripts/evaluation/multi_threads.sh pipeline.mask_dilation_kernel=45

export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

GPUS_CSV="${GPUS:-1,2,3,5}"
IFS=";" read -r -a GPUS <<< "$GPUS_CSV"
NUM_GPUS=${#GPUS[@]}

MASTER_PORT_BASE="${MASTER_PORT_BASE:-29655}"

PIPELINE="${PIPELINE:-qwenimage_mask_flow}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen-Image-Edit-2511}"
IMAGE_ROOT="${IMAGE_ROOT:-dataset/MaskEdit/scene}"
DATA_FILE="${DATA_FILE:-${IMAGE_ROOT}/test.jsonl}"
LORA_MODEL="${LORA_MODEL:-pretrained_weights/lora_adapter}"
LORA_SAFETENSORS_DIR="${LORA_SAFETENSORS_DIR:-$LORA_MODEL}"
ADAPTER_NAME="${ADAPTER_NAME:-maskflow}"
IS_FSDP_CHECKPOINT="${IS_FSDP_CHECKPOINT:-false}"

BASE_SEED="${BASE_SEED:-42}"
CFG_SCALE="${CFG_SCALE:-4.0}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
BATCH_SIZE_PER_PROCESS="${BATCH_SIZE_PER_PROCESS:-1}"
DATA_LOADER_WORKERS="${DATA_LOADER_WORKERS:-0}"

PROJECT_NAME="${PROJECT_NAME:-eval_${PIPELINE}}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/evaluation}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${PROJECT_NAME}/${RUN_TIMESTAMP}}"
EVALUATION_DIR="${OUTPUT_DIR}/evaluations"
LOG_DIR="${OUTPUT_DIR}/logs"
SHARD_DIR="${OUTPUT_DIR}/data_shards"

mkdir -p "$EVALUATION_DIR" "$LOG_DIR" "$SHARD_DIR"

pipeline_args() {
    case "$PIPELINE" in
        qwenimage_mask_flow)
            printf '%s\n' \
                "pipeline.mask_dilation_kernel=${MASK_DILATION_KERNEL:-25}" \
                "pipeline.mask_blur_kernel=${MASK_BLUR_KERNEL:-25}" \
                "pipeline.mask_blur_sigma=${MASK_BLUR_SIGMA:-25.0}" \
                "pipeline.mask_edge_width=${MASK_EDGE_WIDTH:-50}" \
                "pipeline.mask_loss_weight=${MASK_LOSS_WEIGHT:-0.5}" \
                "pipeline.edge_loss_weight=${EDGE_LOSS_WEIGHT:-0}" \
                "pipeline.enable_vae_mask_encoding=${ENABLE_VAE_MASK_ENCODING:-true}" \
                "pipeline.enable_masked_loss=true" \
                "pipeline.mask_denoise_steps=[0.0,1.0]" \
                "pipeline.mask_denoise_train=true" \
                "pipeline.mask_denoise_infer=true"
            ;;
        qwenimage_edit_plus_2511)
            ;;
        *)
            ;;
    esac
}

PIPELINE_ARGS=()
while IFS= read -r arg; do
    [[ -n "$arg" ]] && PIPELINE_ARGS+=("$arg")
done < <(pipeline_args)

if [[ -n "${PIPELINE_EXTRA_ARGS:-}" ]]; then
    # Space-separated Hydra overrides for custom or experimental pipeline knobs.
    # Prefer command-line args for values that may contain spaces.
    read -r -a extra_args <<< "$PIPELINE_EXTRA_ARGS"
    PIPELINE_ARGS+=("${extra_args[@]}")
fi

BASE_ARGS=(
    "--config-path=configs"
    "--config-name=evaluation"
    "project.project_name=$PROJECT_NAME"
    "project.output_dir=$OUTPUT_DIR"
    "project.evaluation_dir=$EVALUATION_DIR"
    "project.log_dir=$LOG_DIR"
    "evalset=mask_edit"
    "evalset.image_root=$IMAGE_ROOT"
    "pipeline=$PIPELINE"
    "pipeline.pretrained_model=$BASE_MODEL"
    "adapter=lora"
    "adapter.adapter_name=$ADAPTER_NAME"
    "base_seed=$BASE_SEED"
    "resume_from=$LORA_MODEL"
    "+is_fsdp_checkpoint=$IS_FSDP_CHECKPOINT"
    "+lora_safetensors_dir=$LORA_SAFETENSORS_DIR"
    "cfg_scale=$CFG_SCALE"
    "num_inference_steps=$NUM_INFERENCE_STEPS"
    "+batch_size_per_process=$BATCH_SIZE_PER_PROCESS"
    "+data_loader_workers=$DATA_LOADER_WORKERS"
)

USER_ARGS=("$@")

python3 scripts/utils/split_eval_data.py \
    --data-file "$DATA_FILE" \
    --num-shards "$NUM_GPUS" \
    --output-dir "$SHARD_DIR" \
    --prefix "eval_rank"

echo "========================================================================="
echo "[Evaluation] pipeline=$PIPELINE"
echo "[GPUs] ${GPUS[*]}"
echo "[Data] $DATA_FILE"
echo "[Output] $OUTPUT_DIR"
echo "========================================================================="

pids=()

run_eval_rank() {
    local rank=$1
    local gpu_id=$2
    local master_port=$3
    local shard_file=$4
    local log_file=$5

    local rank_args=(
        "${BASE_ARGS[@]}"
        "evalset.data_file=$shard_file"
        "${PIPELINE_ARGS[@]}"
        "${USER_ARGS[@]}"
    )

    {
        echo "========================================================================="
        echo "[Rank Start Time]: $(date +'%Y-%m-%d %H:%M:%S')"
        echo "[Rank]: $rank"
        echo "[GPU]: $gpu_id"
        echo "[Port]: $master_port"
        echo "[Shard]: $shard_file"
        echo "[Command]:"
        printf 'CUDA_VISIBLE_DEVICES=%q torchrun --nnodes=1 --nproc-per-node=1 --master-addr 127.0.0.1 --master-port %q evaluate.py' "$gpu_id" "$master_port"
        printf ' %q' "${rank_args[@]}"
        echo
        echo "========================================================================="
        echo
    } > "$log_file"

    HYDRA_FULL_ERROR="$HYDRA_FULL_ERROR" \
    OMP_NUM_THREADS="$OMP_NUM_THREADS" \
    CUDA_VISIBLE_DEVICES="$gpu_id" \
    torchrun \
        --nnodes=1 \
        --nproc-per-node=1 \
        --master-addr 127.0.0.1 \
        --master-port "$master_port" \
        evaluate.py \
        "${rank_args[@]}" >> "$log_file" 2>&1
}

for rank in "${!GPUS[@]}"; do
    gpu_id="${GPUS[$rank]}"
    shard_file="${SHARD_DIR}/eval_rank_${rank}.jsonl"
    master_port=$((MASTER_PORT_BASE + rank))
    log_file="${LOG_DIR}/rank_${rank}.log"

    run_eval_rank "$rank" "$gpu_id" "$master_port" "$shard_file" "$log_file" &
    pids+=("$!")

    echo "Started rank ${rank} on GPU ${gpu_id}, master_port=${master_port}, log=${log_file}"
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
