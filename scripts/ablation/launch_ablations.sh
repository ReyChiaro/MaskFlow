#!/usr/bin/env bash

set -o pipefail

# ==========================================
# 1. User Configs
# ==========================================
# Override examples:
#   GPU_LIST="0 1 2 3" GPUS_PER_TASK=2 bash scripts/ablation/launch_ablations.sh
#   DRY_RUN=1 GPU_LIST="0 1 2 3" GPUS_PER_TASK=2 bash scripts/ablation/launch_ablations.sh
#   bash scripts/ablation/launch_ablations.sh trainer.max_training_steps=1000

GPUS=(${GPU_LIST:-0 1 2 3 4 5 6 7})
GPUS_PER_TASK=${GPUS_PER_TASK:-1}
BASE_PORT=${BASE_PORT:-28600}
DRY_RUN=${DRY_RUN:-0}

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen-Image-Edit-2511}"
OUTPUT_ROOT="${OUTPUT_ROOT:-ablation_experiments/maskflow}"
IMAGE_ROOT="${IMAGE_ROOT:-dataset/MaskEdit/scene}"
LOG_ROOT="${LOG_ROOT:-ablation_experiments_logs/maskflow}"

OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}

# Additional overrides passed to this script are appended to every experiment.
GLOBAL_EXTRA_OVERRIDES=("$@")

# ==========================================
# 2. Common Hydra Overrides
# ==========================================
COMMON_OVERRIDES=(
    "project.output_dir=${OUTPUT_ROOT}/\${project.project_name}/\${project.timestamp}"

    "trainset=mask_edit"
    "evalset=mask_edit"
    "trainset.image_root=${IMAGE_ROOT}"
    "trainset.data_file=${IMAGE_ROOT}/train.jsonl"
    "evalset.image_root=${IMAGE_ROOT}"
    "evalset.data_file=${IMAGE_ROOT}/runtime_testset2.jsonl"

    "pipeline=qwenimage_mask_flow"
    "pipeline.pretrained_model=${BASE_MODEL}"
    "pipeline.mask_dilation_kernel=25"
    "pipeline.mask_blur_kernel=25"
    "pipeline.mask_blur_sigma=25.0"
    "pipeline.mask_edge_width=50"
    "pipeline.mask_loss_weight=1.0"
    "pipeline.edge_loss_weight=0"
    "pipeline.enable_vae_mask_encoding=true"
    "pipeline.enable_masked_loss=true"
    "pipeline.enable_local_denoise_train=false"
    "pipeline.enable_local_denoise_infer=false"
    "pipeline.local_denoise_steps=[0.1,1.0]"
    "pipeline.enable_pixel_blend=false"
    "pipeline.enable_poisson_train=true"
    "pipeline.enable_poisson_infer=true"
    "pipeline.poisson_steps=[0.0,1.0]"
    "pipeline.poisson_lambda_e=0.1"
    "pipeline.poisson_lambda_s=1.0"
    "pipeline.poisson_num_iter=50"
    "pipeline.poisson_momentum=0.1"

    "adapter=lora"
    "adapter.r=256"
    "adapter.lora_alpha=256"
    "adapter.adapter_name=maskflow"

    "trainer=maskflow"
    "trainer.enable_save_optimizer=false"
    "trainer.base_seed=42"
    "trainer.cfg_scale=4.0"
    "trainer.cfg_dropout=0"
    "trainer.max_training_steps=10000"
    "trainer.save_steps=1000"
    "trainer.eval_steps=2500"
    "trainer.mixed_precision=bf16"
    "trainer.enable_gradient_checkpoint=true"
    "trainer.gradient_accumulation_steps=1"
    "trainer.batch_size_per_process=1"
    "trainer.num_inference_steps=50"
    "trainer.fsdp_strategy=no_shard"
    "trainer.prompt_sampler_cfgs.name=linear-decay"
    "trainer.prompt_sampler_cfgs.start_p=1.0"
    "trainer.prompt_sampler_cfgs.end_p=0.0"
)

# ==========================================
# 3. Experiments
# ==========================================
EXPERIMENT_NAMES=()
EXPERIMENT_OVERRIDES=()

add_exp() {
    local exp_name=$1
    shift

    local joined=""
    local override
    for override in "$@"; do
        joined+="${override}"$'\n'
    done

    EXPERIMENT_NAMES+=("${exp_name}")
    EXPERIMENT_OVERRIDES+=("${joined}")
}

# Keep each experiment readable: one Hydra override per line.
# add_exp "mf-baseline" # Poisson, mask loss, linear-decay prompt-scheduler

add_exp "mf-prompt-constant-p0.5" \
    "trainer.prompt_sampler_cfgs.name=constant" \
    "trainer.prompt_sampler_cfgs.p=0.5"

add_exp "mf-prompt-constant-p0" \
    "trainer.prompt_sampler_cfgs.name=constant" \
    "trainer.prompt_sampler_cfgs.p=0"

add_exp "mf-prompt-linear-decay-1.0-0.5" \
    "trainer.prompt_sampler_cfgs.name=linear-decay" \
    "trainer.prompt_sampler_cfgs.start_p=1.0" \
    "trainer.prompt_sampler_cfgs.end_p=0.5"

add_exp "mf-prompt-linear-decay-1.0-0.0" \
    "trainer.prompt_sampler_cfgs.name=linear-decay" \
    "trainer.prompt_sampler_cfgs.start_p=1.0" \
    "trainer.prompt_sampler_cfgs.end_p=0.0"

# More examples:
# add_exp "mf-loss-mask0.5-edge0" \
#     "pipeline.mask_loss_weight=0.5" \
#     "pipeline.edge_loss_weight=0"
#
# add_exp "mf-mask-dilation-75" \
#     "pipeline.mask_dilation_kernel=75"

# ==========================================
# 4. Helpers
# ==========================================
die() {
    echo "[Error] $*" >&2
    exit 1
}

validate_configs() {
    local num_gpus=${#GPUS[@]}
    [[ ${num_gpus} -gt 0 ]] || die "No GPU is configured. Set GPUS or GPU_LIST."
    [[ ${GPUS_PER_TASK} =~ ^[0-9]+$ ]] || die "GPUS_PER_TASK must be a positive integer."
    [[ ${GPUS_PER_TASK} -gt 0 ]] || die "GPUS_PER_TASK must be greater than 0."
    [[ ${GPUS_PER_TASK} -le ${num_gpus} ]] || die "GPUS_PER_TASK=${GPUS_PER_TASK} exceeds available GPUs=${num_gpus}."

    local tasks_per_wave=$((num_gpus / GPUS_PER_TASK))
    [[ ${tasks_per_wave} -gt 0 ]] || die "No task can be scheduled with the current GPU config."

    local unused=$((num_gpus % GPUS_PER_TASK))
    if [[ ${unused} -ne 0 ]]; then
        echo "[Warn] ${unused} GPU(s) will be unused because ${num_gpus} is not divisible by GPUS_PER_TASK=${GPUS_PER_TASK}."
    fi
}

task_gpus() {
    local task_slot=$1
    local start=$((task_slot * GPUS_PER_TASK))
    local visible=""
    local i

    for ((i = 0; i < GPUS_PER_TASK; i++)); do
        if [[ ${i} -gt 0 ]]; then
            visible+=","
        fi
        visible+="${GPUS[$((start + i))]}"
    done

    printf '%s' "${visible}"
}

append_overrides_from_string() {
    local overrides_string=$1
    local line

    while IFS= read -r line; do
        [[ -n "${line}" ]] && CURRENT_OVERRIDES+=("${line}")
    done <<< "${overrides_string}"
}

cleanup_gpu_processes() {
    local gpu
    local pid
    local pids

    for gpu in "$@"; do
        pids=$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)
        [[ -n "${pids}" ]] || continue

        while IFS= read -r pid; do
            [[ -n "${pid}" ]] || continue
            echo "  -> kill residual process ${pid} on GPU ${gpu}"
            kill -9 "${pid}" 2>/dev/null || true
        done <<< "${pids}"
    done
}

run_one_exp() {
    local visible_gpus=$1
    local master_port=$2
    local exp_name=$3
    local log_file=$4
    shift 4

    local nproc_per_node=${GPUS_PER_TASK}
    local project_timestamp
    project_timestamp=$(date +'%Y%m%d-%H%M%S')
    local overrides=(
        "project.project_name=${exp_name}"
        "project.timestamp=${project_timestamp}"
        "${COMMON_OVERRIDES[@]}"
        "$@"
        "${GLOBAL_EXTRA_OVERRIDES[@]}"
    )
    local cmd=(
        torchrun
        "--nnodes=1"
        "--nproc-per-node=${nproc_per_node}"
        "--master-addr=${MASTER_ADDR}"
        "--master-port=${master_port}"
        finetune.py
        "--config-path=configs"
        "--config-name=train"
    )

    mkdir -p "$(dirname "${log_file}")"

    {
        echo "========================================================================="
        echo "[Experiment] ${exp_name}"
        echo "[Start Time] $(date +'%Y-%m-%d %H:%M:%S')"
        echo "[Dry Run] ${DRY_RUN}"
        echo "[GPUs] ${visible_gpus}"
        echo "[Master Port] ${master_port}"
        echo "[Project Timestamp] ${project_timestamp}"
        echo "[Command]"
        printf 'HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=%q OMP_NUM_THREADS=%q MASTER_PORT=%q ' \
            "${visible_gpus}" "${OMP_NUM_THREADS}" "${master_port}"
        printf '%q ' "${cmd[@]}" "${overrides[@]}"
        echo
        echo "========================================================================="
        echo
    } > "${log_file}"

    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "[Dry Run] Skip launching ${exp_name}."
        echo "[Dry Run] Skip launching ${exp_name}." >> "${log_file}"
        return 0
    fi

    HYDRA_FULL_ERROR=1 \
    CUDA_VISIBLE_DEVICES="${visible_gpus}" \
    OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
    MASTER_PORT="${master_port}" \
    "${cmd[@]}" "${overrides[@]}" >> "${log_file}" 2>&1

    local status=$?
    {
        echo
        echo "========================================================================="
        echo "[End Time] $(date +'%Y-%m-%d %H:%M:%S')"
        echo "[Exit Code] ${status}"
        echo "========================================================================="
    } >> "${log_file}"

    return "${status}"
}

ACTIVE_PIDS=()
ACTIVE_GPUS=()

abort_all() {
    echo
    echo "[Abort] Stopping running experiments..."

    local pid
    for pid in "${ACTIVE_PIDS[@]:-}"; do
        kill "${pid}" 2>/dev/null || true
    done

    sleep 2
    cleanup_gpu_processes "${ACTIVE_GPUS[@]:-}"
    exit 130
}

trap abort_all INT TERM

# ==========================================
# 5. Launch
# ==========================================
validate_configs

TOTAL_EXPS=${#EXPERIMENT_NAMES[@]}
NUM_GPUS=${#GPUS[@]}
TASKS_PER_WAVE=$((NUM_GPUS / GPUS_PER_TASK))
FAILED=0

echo "[Ablation] total experiments: ${TOTAL_EXPS}"
echo "[Ablation] GPUs: ${GPUS[*]}"
echo "[Ablation] GPUS_PER_TASK: ${GPUS_PER_TASK}"
echo "[Ablation] tasks per wave: ${TASKS_PER_WAVE}"
echo "[Ablation] logs: ${LOG_ROOT}"
echo "[Ablation] dry run: ${DRY_RUN}"

for ((wave_start = 0; wave_start < TOTAL_EXPS; wave_start += TASKS_PER_WAVE)); do
    PIDS=()
    PID_NAMES=()
    ACTIVE_PIDS=()
    ACTIVE_GPUS=()

    echo
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Start a new experiment wave..."

    for ((slot = 0; slot < TASKS_PER_WAVE; slot++)); do
        exp_idx=$((wave_start + slot))
        [[ ${exp_idx} -lt ${TOTAL_EXPS} ]] || break

        exp_name=${EXPERIMENT_NAMES[${exp_idx}]}
        visible_gpus=$(task_gpus "${slot}")
        master_port=$((BASE_PORT + slot))
        log_file="${LOG_ROOT}/${exp_name}.log"

        CURRENT_OVERRIDES=()
        append_overrides_from_string "${EXPERIMENT_OVERRIDES[${exp_idx}]}"

        echo "  -> [GPU ${visible_gpus}][Port ${master_port}] launch ${exp_name}"
        run_one_exp "${visible_gpus}" "${master_port}" "${exp_name}" "${log_file}" "${CURRENT_OVERRIDES[@]}" &

        pid=$!
        PIDS+=("${pid}")
        PID_NAMES+=("${exp_name}")
        ACTIVE_PIDS+=("${pid}")

        IFS=',' read -r -a VISIBLE_GPU_ARRAY <<< "${visible_gpus}"
        ACTIVE_GPUS+=("${VISIBLE_GPU_ARRAY[@]}")
    done

    for ((i = 0; i < ${#PIDS[@]}; i++)); do
        if wait "${PIDS[${i}]}"; then
            echo "  -> finished: ${PID_NAMES[${i}]}"
        else
            status=$?
            FAILED=1
            echo "  -> failed: ${PID_NAMES[${i}]} (exit ${status})"
        fi
    done

    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Wave finished. Cleaning GPU memory..."
    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "  -> dry run: skip GPU cleanup"
    else
        sleep 2
        cleanup_gpu_processes "${ACTIVE_GPUS[@]}"
        sleep 3
    fi
done

if [[ ${FAILED} -ne 0 ]]; then
    echo
    echo "[Ablation] Finished with failed experiment(s). Check logs under ${LOG_ROOT}."
    exit 1
fi

echo
echo "[Ablation] All experiments finished successfully."
