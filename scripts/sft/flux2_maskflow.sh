#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

python -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc-per-node="${NPROC_PER_NODE:-8}" \
    --master-port 29666 \
    --master-addr "127.0.0.1" \
    finetune.py \
    --config-path configs \
    --config-name sft_flux2_maskflow \
    project=train \
    project.project_name=FLUX2dev-MaskFlow-r256 \
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
    pipeline=flux2_maskflow \
    pipeline.pretrained_model="${PRETRAINED_MODEL:-/root/models/FLUX.2-dev}" \
    pipeline.rescale_cfg=true \
    pipeline.enable_masked_loss=true \
    pipeline.mask_loss_weight=1.0 \
    pipeline.scheduler.unmask_with=noisy_source \
    pipeline.scheduler.background_noise_power=1.0 \
    pipeline.enable_pixel_blend=false \
    pipeline.enable_poisson_train=true \
    pipeline.enable_poisson_infer=true \
    pipeline.poisson_lambda_e=1.0 \
    pipeline.poisson_lambda_s=1.0 \
    pipeline.poisson_num_iter=50 \
    pipeline.poisson_momentum=0.1 \
    "$@"
