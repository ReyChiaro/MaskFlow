export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=3

PIPELINE="qwenimage_mask_flow"
BASE_MODEL="/data/nvme7/models/Qwen-Image-Edit-2511"
IMAGE_ROOT="dataset/MaskEdit/scene"

torchrun \
    --nnodes=1 \
    --nproc-per-node=1 \
    --master-addr 127.0.0.1 \
    --master-port 29800 \
    main.py \
    --config-path=configs \
    --config-name=train \
    project.project_name="_test_maskflow_lora" \
    trainset=mask_edit \
    evalset=mask_edit \
    trainset.image_root=$IMAGE_ROOT \
    trainset.data_file=outputs/data_cache/single.jsonl \
    evalset.image_root=$IMAGE_ROOT \
    evalset.data_file=outputs/data_cache/single.jsonl \
    pipeline=$PIPELINE \
    pipeline.pretrained_model=$BASE_MODEL \
    pipeline.cfg_dropout=0.1 \
    pipeline.mask_dilation_kernel=25 \
    pipeline.mask_blur_kernel=25 \
    pipeline.mask_blur_sigma=25.0 \
    pipeline.mask_edge_width=50 \
    pipeline.mask_loss_weight=0 \
    pipeline.edge_loss_weight=0 \
    pipeline.enable_vae_mask_encoding=true \
    pipeline.inpainting_denoising_steps=45 \
    adapter=lora \
    adapter.r=256 \
    adapter.lora_alpha=256 \
    adapter.adapter_name=mask_flow \
    trainer=lora \
    trainer.enable_save_optimizer=true \
    trainer.base_seed=42 \
    trainer.cfg_scale=4.0 \
    trainer.max_training_steps=10000 \
    trainer.save_steps=100 \
    trainer.eval_steps=50 \
    trainer.mixed_precision=bf16 \
    trainer.enable_gradient_checkpoint=true \
    trainer.gradient_accumulation_steps=1 \
    trainer.batch_size_per_process=1 \
    trainer.num_inference_steps=50 \
    trainer.fsdp_strategy=no_shard