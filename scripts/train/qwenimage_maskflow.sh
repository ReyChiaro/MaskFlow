export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=8

PIPELINE="qwenimage_mask_flow"
BASE_MODEL="${BASE_MODEL-/data/nvme7/models/Qwen-Image-Edit-2511}"
IMAGE_ROOT="dataset/MaskEdit/scene"

torchrun \
    --nnodes=1 \
    --nproc-per-node=1 \
    --master-addr 127.0.0.1 \
    --master-port 29696 \
    finetune.py \
    --config-path=configs \
    --config-name=train \
    project.project_name="maskflow-linear_decay_1.0_0.0" \
    trainset=mask_edit \
    evalset=mask_edit \
    trainset.image_root=$IMAGE_ROOT \
    trainset.data_file=$IMAGE_ROOT/train.jsonl \
    evalset.image_root=$IMAGE_ROOT \
    evalset.data_file=${IMAGE_ROOT}/runtime_testset2.jsonl \
    pipeline=$PIPELINE \
    pipeline.pretrained_model=$BASE_MODEL \
    pipeline.mask_dilation_kernel=25 \
    pipeline.mask_blur_kernel=25 \
    pipeline.mask_blur_sigma=25.0 \
    pipeline.mask_edge_width=50 \
    pipeline.mask_loss_weight=1.0 \
    pipeline.edge_loss_weight=0 \
    pipeline.enable_vae_mask_encoding=true \
    pipeline.enable_masked_loss=true \
    pipeline.enable_local_denoise_train=false \
    pipeline.enable_local_denoise_infer=false \
    pipeline.local_denoise_steps="[0.0,1.0]" \
    pipeline.enable_pixel_blend=true \
    pipeline.enable_poisson_train=true \
    pipeline.enable_poisson_infer=true \
    pipeline.poisson_steps="[0.0,1.0]" \
    pipeline.poisson_lambda_e=1.0 \
    pipeline.poisson_lambda_s=1.0 \
    pipeline.poisson_num_iter=50 \
    pipeline.poisson_momentum=0.1 \
    adapter=lora \
    adapter.r=256 \
    adapter.lora_alpha=256 \
    adapter.adapter_name=maskflow \
    trainer=maskflow \
    trainer.enable_save_optimizer=false \
    trainer.base_seed=42 \
    trainer.cfg_scale=4.0 \
    trainer.cfg_dropout=0.1 \
    trainer.max_training_steps=5000 \
    trainer.save_steps=1000 \
    trainer.eval_steps=2500 \
    trainer.mixed_precision=bf16 \
    trainer.enable_gradient_checkpoint=true \
    trainer.gradient_accumulation_steps=1 \
    trainer.batch_size_per_process=1 \
    trainer.num_inference_steps=50 \
    trainer.fsdp_strategy=no_shard \
    trainer.prompt_sampler_cfgs.name=linear-decay \
    trainer.prompt_sampler_cfgs.p=0.5 \
    trainer.prompt_sampler_cfgs.start_p=1.0 \
    trainer.prompt_sampler_cfgs.end_p=0.0 \
