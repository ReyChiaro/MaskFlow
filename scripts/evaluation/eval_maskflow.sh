#!/usr/bin/env bash
set -euo pipefail

# Evaluate MaskFlow ablation experiments sequentially.
# Each experiment uses all configured GPUs in parallel, then exits fully before
# the next experiment starts so GPU memory can be released.

EXPERIMENTS=(
    "mf-baseline"
    "mf-mask_inp-0.1-1.0"
    "mf-mask_inp-0.0-0.9"
    "mf-mask_inp-0.1-0.9"
    "mf_ps-constant_p0.2"
    "mf_ps-constant_p0.5"
    "mf_ps-linear1.0-0.5"
    "mf_ps-linear1.0-0.0"
)

export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

GPUS_CSV="${GPUS:-1,2}"
IFS="," read -r -a GPUS <<< "$GPUS_CSV"
NUM_GPUS=${#GPUS[@]}

ABLATIONS_ROOT="${ABLATIONS_ROOT:-ablations}"
IMAGE_ROOT="${IMAGE_ROOT:-dataset/MaskEdit/scene}"
DATA_FILE="${DATA_FILE:-${IMAGE_ROOT}/test.jsonl}"

MASTER_PORT_BASE="${MASTER_PORT_BASE:-29655}"
BASE_SEED="${BASE_SEED:-42}"
CFG_SCALE="${CFG_SCALE:-4.0}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
BATCH_SIZE_PER_PROCESS="${BATCH_SIZE_PER_PROCESS:-1}"
DATA_LOADER_WORKERS="${DATA_LOADER_WORKERS:-0}"

RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/evaluation/maskflow_ablations}"
PROJECT_NAME_PREFIX="${PROJECT_NAME_PREFIX:-eval_}"

USER_ARGS=("$@")
CURRENT_PIDS=()

cleanup() {
    local pid
    for pid in "${CURRENT_PIDS[@]:-}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
}
trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM

die() {
    echo "[Error] $*" >&2
    exit 1
}

latest_run_dir() {
    local exp_name=$1
    local exp_dir="${ABLATIONS_ROOT}/${exp_name}"

    [[ -d "$exp_dir" ]] || die "Experiment directory not found: $exp_dir"

    local latest
    latest=$(find "$exp_dir" -mindepth 1 -maxdepth 1 -type d -print | sort | tail -n 1)
    [[ -n "$latest" ]] || die "No timestamped run directories found in: $exp_dir"

    printf '%s\n' "$latest"
}

latest_checkpoint_dir() {
    local run_dir=$1
    local checkpoint_root="${run_dir}/checkpoints"

    [[ -d "$checkpoint_root" ]] || die "Checkpoint directory not found: $checkpoint_root"

    local latest_step=""
    local latest_step_num=-1
    local step_dir
    local step_name
    local step_num

    while IFS= read -r step_dir; do
        step_name=$(basename "$step_dir")
        step_num=${step_name#step-}
        [[ "$step_num" =~ ^[0-9]+$ ]] || continue
        [[ -d "${step_dir}/model" ]] || continue

        if (( step_num > latest_step_num )); then
            latest_step_num=$step_num
            latest_step=$step_dir
        fi
    done < <(find "$checkpoint_root" -mindepth 1 -maxdepth 1 -type d -name 'step-*' -print)

    [[ -n "$latest_step" ]] || die "No checkpoints/step-*/model directory found under: $checkpoint_root"

    printf '%s\n' "$latest_step"
}

split_eval_data() {
    local shard_dir=$1

    python3 scripts/utils/split_eval_data.py \
        --data-file "$DATA_FILE" \
        --num-shards "$NUM_GPUS" \
        --output-dir "$shard_dir" \
        --prefix "eval_rank"
}

has_lora_safetensors() {
    local lora_safetensors_dir=$1
    [[ -d "$lora_safetensors_dir" ]] || return 1
    find "$lora_safetensors_dir" -type f -name '*.safetensors' -print -quit | grep -q .
}

ensure_lora_safetensors() {
    local exp_name=$1
    local run_dir=$2
    local step_dir=$3
    local exp_output_dir=$4
    local log_dir=$5

    local hydra_config_dir="${run_dir}/hydra-configs/.hydra"
    local model_dir="${step_dir}/model"
    local lora_safetensors_dir="${step_dir}/lora_adapter"
    local convert_dir="${exp_output_dir}/convert_lora"
    local empty_data_file="${convert_dir}/empty.jsonl"
    local log_file="${log_dir}/convert_lora.log"
    local gpu_id="${GPUS[0]}"
    local master_port=$((MASTER_PORT_BASE + NUM_GPUS))

    if has_lora_safetensors "$lora_safetensors_dir"; then
        echo "[LoRA] Reusing existing safetensors: $lora_safetensors_dir"
        return
    fi

    mkdir -p "$convert_dir"
    : > "$empty_data_file"

    local convert_args=(
        "--config-path=${hydra_config_dir}"
        "--config-name=config"
        "hydra.run.dir=${convert_dir}/hydra-configs"
        "project.project_name=${PROJECT_NAME_PREFIX}${exp_name}_convert_lora"
        "project.output_dir=${convert_dir}"
        "project.evaluation_dir=${convert_dir}/evaluations"
        "project.log_dir=${convert_dir}/logs"
        "evalset.image_root=${IMAGE_ROOT}"
        "evalset.data_file=${empty_data_file}"
        "++pipe_configs=\${pipeline}"
        "++eval_data_configs=\${evalset}"
        "++lora_configs=\${adapter}"
        "++base_seed=${BASE_SEED}"
        "++resume_from=${model_dir}"
        "++is_fsdp_checkpoint=true"
        "++lora_safetensors_dir=${lora_safetensors_dir}"
        "++cfg_scale=${CFG_SCALE}"
        "++num_inference_steps=${NUM_INFERENCE_STEPS}"
        "++batch_size_per_process=${BATCH_SIZE_PER_PROCESS}"
        "++data_loader_workers=0"
        "${USER_ARGS[@]}"
    )

    {
        echo "========================================================================="
        echo "[LoRA Convert Start Time]: $(date +'%Y-%m-%d %H:%M:%S')"
        echo "[Experiment]: $exp_name"
        echo "[GPU]: $gpu_id"
        echo "[Port]: $master_port"
        echo "[FSDP checkpoint]: $model_dir"
        echo "[Safetensors output]: $lora_safetensors_dir"
        echo "[Command]:"
        printf 'CUDA_VISIBLE_DEVICES=%q torchrun --nnodes=1 --nproc-per-node=1 --master-addr 127.0.0.1 --master-port %q evaluate.py' "$gpu_id" "$master_port"
        printf ' %q' "${convert_args[@]}"
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
        "${convert_args[@]}" >> "$log_file" 2>&1

    has_lora_safetensors "$lora_safetensors_dir" || die "LoRA safetensors conversion failed. Check log: $log_file"
}

run_eval_rank() {
    local rank=$1
    local gpu_id=$2
    local master_port=$3
    local exp_name=$4
    local run_dir=$5
    local step_dir=$6
    local shard_file=$7
    local rank_output_dir=$8
    local log_file=$9

    local hydra_config_dir="${run_dir}/hydra-configs/.hydra"
    local lora_safetensors_dir="${step_dir}/lora_adapter"
    local rank_eval_dir="${rank_output_dir}/evaluations"
    local rank_log_dir="${rank_output_dir}/logs"
    local rank_hydra_dir="${rank_output_dir}/hydra-configs"

    mkdir -p "$rank_eval_dir" "$rank_log_dir" "$rank_hydra_dir"

    local rank_args=(
        "--config-path=${hydra_config_dir}"
        "--config-name=config"
        "hydra.run.dir=${rank_hydra_dir}"
        "project.project_name=${PROJECT_NAME_PREFIX}${exp_name}"
        "project.output_dir=${rank_output_dir}"
        "project.evaluation_dir=${rank_eval_dir}"
        "project.log_dir=${rank_log_dir}"
        "evalset.image_root=${IMAGE_ROOT}"
        "evalset.data_file=${shard_file}"
        "++pipe_configs=\${pipeline}"
        "++eval_data_configs=\${evalset}"
        "++lora_configs=\${adapter}"
        "++base_seed=${BASE_SEED}"
        "++resume_from=${lora_safetensors_dir}"
        "++is_fsdp_checkpoint=false"
        "++lora_safetensors_dir=${lora_safetensors_dir}"
        "++cfg_scale=${CFG_SCALE}"
        "++num_inference_steps=${NUM_INFERENCE_STEPS}"
        "++batch_size_per_process=${BATCH_SIZE_PER_PROCESS}"
        "++data_loader_workers=${DATA_LOADER_WORKERS}"
        "${USER_ARGS[@]}"
    )

    {
        echo "========================================================================="
        echo "[Rank Start Time]: $(date +'%Y-%m-%d %H:%M:%S')"
        echo "[Experiment]: $exp_name"
        echo "[Run Dir]: $run_dir"
        echo "[Checkpoint Step]: $step_dir"
        echo "[Rank]: $rank"
        echo "[GPU]: $gpu_id"
        echo "[Port]: $master_port"
        echo "[Shard]: $shard_file"
        echo "[Output]: $rank_output_dir"
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

run_experiment() {
    local exp_name=$1
    local run_dir
    local step_dir
    local exp_output_dir
    local log_dir
    local shard_dir

    run_dir=$(latest_run_dir "$exp_name")
    step_dir=$(latest_checkpoint_dir "$run_dir")

    [[ -f "${run_dir}/hydra-configs/.hydra/config.yaml" ]] || die "Hydra config not found: ${run_dir}/hydra-configs/.hydra/config.yaml"
    [[ -d "${run_dir}/logs" ]] || die "Training logs directory not found: ${run_dir}/logs"

    exp_output_dir="${OUTPUT_ROOT}/${RUN_TIMESTAMP}/${exp_name}"
    log_dir="${exp_output_dir}/logs"
    shard_dir="${exp_output_dir}/data_shards"
    mkdir -p "$log_dir" "$shard_dir"

    echo "========================================================================="
    echo "[Experiment] $exp_name"
    echo "[Run] $run_dir"
    echo "[Checkpoint] $step_dir"
    echo "[LoRA safetensors cache] ${step_dir}/lora_adapter"
    echo "[GPUs] ${GPUS[*]}"
    echo "[Data] $DATA_FILE"
    echo "[Output] $exp_output_dir"
    echo "========================================================================="

    ensure_lora_safetensors "$exp_name" "$run_dir" "$step_dir" "$exp_output_dir" "$log_dir"
    split_eval_data "$shard_dir"

    CURRENT_PIDS=()
    local rank
    for rank in "${!GPUS[@]}"; do
        local gpu_id="${GPUS[$rank]}"
        local shard_file="${shard_dir}/eval_rank_${rank}.jsonl"
        local master_port=$((MASTER_PORT_BASE + rank))
        local rank_output_dir="${exp_output_dir}/rank_${rank}"
        local log_file="${log_dir}/rank_${rank}.log"

        run_eval_rank "$rank" "$gpu_id" "$master_port" "$exp_name" "$run_dir" "$step_dir" "$shard_file" "$rank_output_dir" "$log_file" &
        CURRENT_PIDS+=("$!")

        echo "Started rank ${rank} on GPU ${gpu_id}, master_port=${master_port}, log=${log_file}"
    done

    local failed=0
    local pid
    for pid in "${CURRENT_PIDS[@]}"; do
        if ! wait "$pid"; then
            failed=1
        fi
    done
    CURRENT_PIDS=()

    if [[ "$failed" -ne 0 ]]; then
        die "Experiment failed: ${exp_name}. Check logs in ${log_dir}."
    fi

    echo "[Done] ${exp_name}. Images are saved under ${exp_output_dir}/rank_*/evaluations."
    sleep "${SLEEP_BETWEEN_EXPERIMENTS:-5}"
}

[[ "$NUM_GPUS" -gt 0 ]] || die "No GPUs configured. Set GPUS as a comma-separated list, e.g. GPUS=0,1,2,3,4,5,6,7."
[[ -f "$DATA_FILE" ]] || die "Evaluation data file not found: $DATA_FILE"

for exp_name in "${EXPERIMENTS[@]}"; do
    run_experiment "$exp_name"
done

echo "All MaskFlow ablation evaluations finished."
