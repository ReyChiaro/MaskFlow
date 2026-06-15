# ==========================================
# 1. Global Augments
# ==========================================
GPUS=(0 1 2)
NUM_GPUS=${#GPUS[@]}
BASE_PORT=29800

BASE_MODEL="Qwen/Qwen-Image-Edit-2511"
OUTPUT_ROOT="ablations"
IMAGE_ROOT="dataset/MaskEdit/scene"

run_base_experiment() {
    local gpu_id=$1
    local master_port=$2
    local exp_name=$3
    local log_file=$4
    shift 4

    local base_args=(
        "project.project_name=$exp_name"
        "project.output_dir='$OUTPUT_ROOT/\${project.project_name}/\${project.timestamp}'"
        "trainset=mask_edit"
        "evalset=mask_edit"
        "trainset.image_root=$IMAGE_ROOT"
        "trainset.data_file=$IMAGE_ROOT/trainset2.jsonl"
        "evalset.image_root=$IMAGE_ROOT"
        "evalset.data_file=$IMAGE_ROOT/runtime_testset2.jsonl"
        "pipeline=qwenimage_mask_flow"
        "pipeline.pretrained_model=$BASE_MODEL"
        "pipeline.cfg_dropout=0.1"
        "pipeline.mask_dilation_kernel=25"
        "pipeline.mask_blur_kernel=25"
        "pipeline.mask_blur_sigma=25.0"
        "pipeline.mask_edge_width=50"
        "pipeline.mask_loss_weight=0"
        "pipeline.edge_loss_weight=0"
        "pipeline.enable_vae_mask_encoding=true"
        "pipeline.enable_masked_loss=true"
        "pipeline.inpainting_denoising_steps=50" # Inference only
        "pipeline.scheduler.unmask_with=noisy_source"
        "adapter=lora"
        "adapter.r=256"
        "adapter.lora_alpha=256"
        "adapter.adapter_name=mask_flow"
        "trainer=lora"
        "trainer.enable_save_optimizer=true"
        "trainer.base_seed=304"
        "trainer.cfg_scale=4.0"
        "trainer.max_training_steps=5000"
        "trainer.save_steps=500"
        "trainer.eval_steps=2500"
        "trainer.mixed_precision=bf16"
        "trainer.enable_gradient_checkpoint=true"
        "trainer.gradient_accumulation_steps=1"
        "trainer.batch_size_per_process=1"
        "trainer.num_inference_steps=50"
        "trainer.fsdp_strategy=no_shard"
    )

    local extra_args=("$@")

    local full_cmd="CUDA_VISIBLE_DEVICES=$gpu_id OMP_NUM_THREADS=8 NPROC_PER_NODE=1 MASTER_PORT=$master_port bash scripts/train.sh ${base_args[*]} ${extra_args[*]}"
    
    {
        echo "========================================================================="
        echo "[Experiment Start Time]: $(date +'%Y-%m-%d %H:%M:%S')"
        echo "[Full Command]:"
        echo "$full_cmd"
        echo "========================================================================="
        echo -e "\n--- Logging Start ---\n"
    } > "$log_file"

    HYDRA_FULL_ERROR=1 \
    CUDA_VISIBLE_DEVICES=$gpu_id \
    OMP_NUM_THREADS=8 \
    NPROC_PER_NODE=1 \
    MASTER_PORT=$master_port \
    torchrun \
    --nnode 1 \
    --nproc-per-node 1 \
    --master-addr "127.0.0.1" \
    --master-port $master_port \
    finetune.py \
    --config-path configs \
    --config-name train \
    "${base_args[@]}" "${extra_args[@]}" >> "$log_file" 2>&1
}

# ==========================================
# 2. Experiments
# ==========================================
EXPERIMENTS=(
    "baseline|pipeline.enable_masked_loss=false"
    "mf_loss|"
    "mf_loss-mask1.0_edge0|pipeline.mask_loss_weight=1.0,pipeline.edge_loss_weight=0.0"
    "mf_loss-mask0_edge1.0|pipeline.mask_loss_weight=0.0,pipeline.edge_loss_weight=1.0"

    "mf_loss-mask0.5_edge0|pipeline.mask_loss_weight=0.5,pipeline.edge_loss_weight=0"
    "mf_loss-mask2.0_edge0|pipeline.mask_loss_weight=2.0,pipeline.edge_loss_weight=0"

    "mf_loss-mask0_edge0.5|pipeline.mask_loss_weight=0,pipeline.edge_loss_weight=0.5"
    "mf_loss-mask0_edge2.0|pipeline.mask_loss_weight=0,pipeline.edge_loss_weight=2.0"

    "mf_loss-mask0.5_edge0.5|pipeline.mask_loss_weight=0.5,pipeline.edge_loss_weight=0.5"
    "mf_loss-mask0.5_edge1.0|pipeline.mask_loss_weight=0.5,pipeline.edge_loss_weight=1.0"
    "mf_loss-mask1.0_edge0.5|pipeline.mask_loss_weight=1.0,pipeline.edge_loss_weight=0.5"
    "mf_loss-mask1.0_edge1.0|pipeline.mask_loss_weight=1.0,pipeline.edge_loss_weight=1.0"

    "mf_mask-dila0|pipeline.mask_loss_weight=1.0,pipeline.edge_loss_weight=0.0,pipeline.mask_dilation_kernel=0"
    "mf_mask-dila75|pipeline.mask_loss_weight=1.0,pipeline.edge_loss_weight=0.0,pipeline.mask_dilation_kernel=75"

    "mf_mask-blur0|pipeline.mask_loss_weight=1.0,pipeline.edge_loss_weight=0.0,pipeline.mask_blur_kernel=0"
    "mf_mask-blur75|pipeline.mask_loss_weight=1.0,pipeline.edge_loss_weight=0.0,pipeline.mask_blur_kernel=75"
)

TOTAL_EXPS=${#EXPERIMENTS[@]}

# ==========================================
# 3. Multi-threads
# ==========================================
for ((i=0; i<$TOTAL_EXPS; i+=$NUM_GPUS)); do
    PIDS=()
    echo -e "\n[$(date +'%Y-%m-%d %H:%M:%S')] 🚀 New experiments start..."

    for ((g=0; g<$NUM_GPUS; g++)); do
        EXP_IDX=$((i + g))
        if [ $EXP_IDX -ge $TOTAL_EXPS ]; then break; fi

        CUDA_DEVICE=${GPUS[$g]}
        CURRENT_PORT=$((BASE_PORT + g))
        
        IFS="|" read -r EXP_NAME EXTRA_ARGS_STR <<< "${EXPERIMENTS[$EXP_IDX]}"
        IFS="," read -r -a EXTRA_ARGS_ARRAY <<< "$EXTRA_ARGS_STR"
        LOG_FILE="multi_threads_logs/${EXP_NAME}.log"
        mkdir -p multi_threads_logs
        
        echo "  -> [GPU $CUDA_DEVICE][Port $CURRENT_PORT] 启动: $EXP_NAME"

        run_base_experiment "$CUDA_DEVICE" "$CURRENT_PORT" "$EXP_NAME" "$LOG_FILE" "${EXTRA_ARGS_ARRAY[@]}" &

        PIDS+=($!)
    done

    for pid in "${PIDS[@]}"; do wait "$pid"; done
    sleep 2

    for CUDA_DEVICE in "${GPUS[@]}"; do
        REMAINING_PIDS=$(nvidia-smi -i "$CUDA_DEVICE" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null)
        if [ ! -z "$REMAINING_PIDS" ]; then
            echo "$REMAINING_PIDS" | xargs -r kill -9 2>/dev/null
        fi
    done
    echo "------------------------------------------"
done

echo "🎉 All alblation studies have finished!"