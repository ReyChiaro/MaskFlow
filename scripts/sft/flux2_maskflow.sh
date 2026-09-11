#!/usr/bin/env bash
set -euo pipefail

export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

.venv/bin/python3 -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc-per-node="${NPROC_PER_NODE:-8}" \
    --master-port 29665 \
    --master-addr "127.0.0.1" \
    finetune.py \
    --config-path configs \
    --config-name sft_flux2_maskflow \
    project.project_name=FLUX2dev-MaskFlow-r256 \
    trainer=maskflow \
    trainer.fsdp_strategy=full_shard \
    trainer.max_training_steps=1250 \
    trainer.save_steps=1250 \
    trainer.eval_steps=1250 \
    trainer.batch_size_per_process=1 \
    trainer.text_cfg_dropout=0.1 \
    trainer.num_inference_steps=28 \
    trainset=mask_edit \
    evalset=mask_edit \
    evalset.data_file=dataset/MaskEdit/scene/runtime_testset.jsonl \
    pipeline=flux2_maskflow \
    pipeline.pretrained_model=/root/models/FLUX.2-dev \
    pipeline.enable_pixel_blend=false \
    pipeline.enable_poisson_train=true \
    pipeline.enable_poisson_infer=true \
    pipeline.poisson_lambda_e=1.0 \
    pipeline.poisson_lambda_s=1.0 \
    pipeline.poisson_num_iter=50 \
    pipeline.poisson_momentum=0.1 \
    "$@"
