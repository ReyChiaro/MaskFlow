#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
trap 'exit 130' INT
trap 'exit 143' TERM

# Select the backbone while keeping the experiment settings identical.
PIPELINE="${PIPELINE:-qwenimage_maskflow}"
case "$PIPELINE" in
    qwenimage_maskflow)
        TRAIN_CONFIG=sft_maskflow
        PROJECT_NAME=QwenImage-MaskFlow-r256
        ADAPTER=qwenimage_lora
        PRETRAINED_MODEL="${PRETRAINED_MODEL:-/root/intern-xr/models/Qwen-Image-Edit-2511}"
        ;;
    flux2_maskflow)
        TRAIN_CONFIG=sft_flux2_maskflow
        PROJECT_NAME=FLUX2dev-MaskFlow-r256
        ADAPTER=flux2_lora
        PRETRAINED_MODEL="${PRETRAINED_MODEL:-/root/models/FLUX.2-dev}"
        ;;
    *) printf 'Unsupported pipeline: %s\n' "$PIPELINE" >&2; exit 1 ;;
esac

# Each row: pipeline.mask_loss_weight; pipeline.scheduler.background_noise_power
# Loss = full-image MSE + mask_loss_weight * area-normalized masked MSE.
# Background noise = shifted_sigma ** background_noise_power; 1.0 is the baseline.
EXPERIMENTS=(
    # H100✅ "1.0; 1.0" # Baseline.
    # H100✅ "0.0; 1.0" # No extra masked loss.
    # H100✅ "2.0; 1.0" # Stronger masked loss.

    # H100✅ "1.0; 1.5" # Earlier background, baseline masked loss.
    # H100✅ "1.0; 2.0" # Further reduce background noise.

    # H200✅ "1.0; 5.0" # Further reduce background noise.

    # H100✅ "2.0; 1.25"
    # H100✅ "2.0; 1.5"
    # H100✅ "2.0; 2.0"

    # H100✅ "2.0; 1.1"
    # H100✅ "2.0; 1.05"

    "0.75; 1.0"
    "0.5; 1.0"


    # "0.0; 1.5"
    # "2.0; 1.5"

    # "0.0; 2.0"
    # "2.0; 2.0"
)

RUN_ID="$(date +%Y%m%d-%H%M%S)-$$"
EXPERIMENT_INDEX=0
FAILED_EXPERIMENTS=0

for exp in "${EXPERIMENTS[@]}"; do
    EXPERIMENT_INDEX=$((EXPERIMENT_INDEX + 1))
    IFS='; ' read -r MASK_LOSS_WEIGHT BACKGROUND_NOISE_POWER <<< "$exp"

    EXPERIMENT_ID="maskloss-bgnoise/${RUN_ID}-${EXPERIMENT_INDEX}/maskloss-${MASK_LOSS_WEIGHT}-bgpower-${BACKGROUND_NOISE_POWER}"
    TRAIN_DIR="outputs/experiments/$PROJECT_NAME/$EXPERIMENT_ID"
    EVAL_DIR="outputs/evaluation/eval_$PROJECT_NAME/$EXPERIMENT_ID"
    CHECKPOINT="$TRAIN_DIR/checkpoints/step-1250/lora_adapter/pytorch_lora_weights.safetensors"
    printf '\nExperiment %s: %s\nTraining: %s\nInference: %s\n' "$EXPERIMENT_INDEX" "$exp" "$TRAIN_DIR" "$EVAL_DIR"

    # 1. Train this row. Keep CFG, data, seeds and all other settings fixed.
    if python -m torch.distributed.run \
        --standalone \
        --nnodes=1 \
        --nproc-per-node="${NPROC_PER_NODE:-8}" \
        --master-port 29666 \
        --master-addr "127.0.0.1" \
        finetune.py \
        --config-path configs \
        --config-name "$TRAIN_CONFIG" \
        project=train \
        project.project_name="$PROJECT_NAME" \
        project.timestamp="$EXPERIMENT_ID" \
        trainer=maskflow \
        trainer.enable_save_optimizer=false \
        trainer.base_seed=42 \
        trainer.eval_seed=42 \
        trainer.resume_from='' \
        trainer.text_cfg_scale=4.0 \
        trainer.mask_cfg_scale=1.0 \
        trainer.interaction_cfg_scale=null \
        trainer.cfg_branch_probabilities.pm=0.9 \
        trainer.cfg_branch_probabilities.pn=0.0 \
        trainer.cfg_branch_probabilities.nm=0.1 \
        trainer.cfg_branch_probabilities.nn=0.0 \
        trainer.max_grad_norm=1.0 \
        trainer.max_training_steps=1250 \
        trainer.save_steps=1250 \
        trainer.eval_steps=1250 \
        trainer.mixed_precision=bf16 \
        trainer.enable_gradient_checkpoint=true \
        trainer.gradient_accumulation_steps=1 \
        trainer.fsdp_strategy="${FSDP_STRATEGY:-no_shard}" \
        trainer.batch_size_per_process=1 \
        trainer.data_loader_workers=8 \
        trainer.num_inference_steps=50 \
        trainer.prompt_sampler_cfgs.name=constant \
        trainer.prompt_sampler_cfgs.p=0.0 \
        trainset=hf_mask_edit \
        trainset.subsets=scene \
        trainset.split=train \
        trainset.load_end=1.0 \
        evalset=hf_mask_edit \
        evalset.subsets=scene \
        evalset.split=test \
        evalset.load_end=0 \
        pipeline="$PIPELINE" \
        pipeline.pretrained_model="$PRETRAINED_MODEL" \
        pipeline.rescale_cfg=true \
        pipeline.enable_masked_loss=true \
        pipeline.mask_loss_weight="$MASK_LOSS_WEIGHT" \
        pipeline.scheduler.unmask_with=noisy_source \
        pipeline.scheduler.background_noise_power="$BACKGROUND_NOISE_POWER" \
        pipeline.enable_pixel_blend=false \
        pipeline.enable_poisson_train=true \
        pipeline.enable_poisson_infer=true \
        pipeline.poisson_lambda_e=1.0 \
        pipeline.poisson_lambda_s=1.0 \
        pipeline.poisson_num_iter=50 \
        pipeline.poisson_momentum=0.1

    then
        printf 'Training completed: %s\n' "$TRAIN_DIR"
    else
        status=$?
        if [[ "$status" -eq 130 || "$status" -eq 143 ]]; then exit "$status"; fi
        printf 'Training failed (exit %s); skipping this experiment.\n' "$status" >&2
        FAILED_EXPERIMENTS=$((FAILED_EXPERIMENTS + 1))
        continue
    fi

    # The foreground training process has exited and released its GPU resources.
    if [[ ! -f "$CHECKPOINT" ]]; then
        printf 'Checkpoint missing; skipping this experiment: %s\n' "$CHECKPOINT" >&2
        FAILED_EXPERIMENTS=$((FAILED_EXPERIMENTS + 1))
        continue
    fi

    # 2. Evaluate this checkpoint with the SAME background noise power as training.
    if python -m torch.distributed.run \
        --standalone \
        --nnodes=1 \
        --nproc-per-node="${NPROC_PER_NODE:-8}" \
        evaluate.py \
        --config-path configs \
        --config-name "eval_$PIPELINE" \
        project=evaluation \
        project.project_name="eval_$PROJECT_NAME" \
        project.timestamp="$EXPERIMENT_ID" \
        pipeline="$PIPELINE" \
        pipeline.pretrained_model="$PRETRAINED_MODEL" \
        pipeline.rescale_cfg=true \
        pipeline.enable_masked_loss=true \
        pipeline.mask_loss_weight="$MASK_LOSS_WEIGHT" \
        pipeline.scheduler.unmask_with=noisy_source \
        pipeline.scheduler.background_noise_power="$BACKGROUND_NOISE_POWER" \
        pipeline.enable_pixel_blend=false \
        pipeline.enable_poisson_train=true \
        pipeline.enable_poisson_infer=true \
        pipeline.poisson_lambda_e=1.0 \
        pipeline.poisson_lambda_s=1.0 \
        pipeline.poisson_num_iter=50 \
        pipeline.poisson_momentum=0.1 \
        evalset=hf_mask_edit \
        evalset.subsets=scene \
        evalset.split=test \
        evalset.load_end=1.0 \
        adapter@sft_adapter="$ADAPTER" \
        adapters.sft.path="$CHECKPOINT" \
        adapters.sft.lora_scale=1.0 \
        base_seed=42 \
        eval_seed=42 \
        weight_dtype=bf16 \
        batch_size_per_process=1 \
        num_workers=8 \
        text_cfg_scale=4.0 \
        mask_cfg_scale=1.0 \
        interaction_cfg_scale=null \
        num_inference_steps=50 \
        eval_with_position_prompt=false

    then
        printf 'Inference completed: %s/evaluations/predictions\n' "$EVAL_DIR"
    else
        status=$?
        if [[ "$status" -eq 130 || "$status" -eq 143 ]]; then exit "$status"; fi
        printf 'Inference failed (exit %s); continuing to the next experiment.\n' "$status" >&2
        FAILED_EXPERIMENTS=$((FAILED_EXPERIMENTS + 1))
        continue
    fi

    # Keep metrics disabled so overnight runs proceed directly to the next experiment.
    # python calculate_metrics.py \
    #     --data-root dataset/MaskEdit-10k \
    #     --subsets scene \
    #     --split test \
    #     --pred-dir "$EVAL_DIR/evaluations/predictions" \
    #     --reference target \
    #     --region whole \
    #     --metrics CLIP DINO FID PSNR SSIM LPIPS VGG-CONTENT \
    #     --enable-pixel-blend \
    #     --device cuda \
    #     --output "$EVAL_DIR/metrics.json"
done

printf '\nFinished %s experiments; %s failed.\n' "$EXPERIMENT_INDEX" "$FAILED_EXPERIMENTS"
if [[ "$FAILED_EXPERIMENTS" -gt 0 ]]; then exit 1; fi
