export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1

PIPELINE="qwenimage_mask_flow"
BASE_MODEL="/data/nvme7/models/Qwen-Image-Edit-2511"
IMAGE_ROOT="dataset/MaskEdit/scene"

torchrun \
    --nnodes=1 \
    --nproc-per-node=2 \
    --master-addr 127.0.0.1 \
    --master-port 29555 \
    main.py \
    --config-path=configs \
    --config-name=train \
    project.project_name="maskflow_lora" \
    trainset=mask_edit \
    evalset=mask_edit \
    trainset.image_root=$IMAGE_ROOT \
    trainset.data_file=$IMAGE_ROOT/trainset.jsonl \
    evalset.image_root=$IMAGE_ROOT \
    evalset.data_file=outputs/data_cache/testset.jsonl \
    pipeline=$PIPELINE \
    pipeline.pretrained_model=$BASE_MODEL \
    pipeline.mask_dilation_kernel=25 \
    pipeline.mask_blur_kernel=25 \
    pipeline.mask_blur_sigma=25.0 \
    pipeline.mask_edge_width=50 \
    pipeline.mask_loss_weight=0.5 \
    pipeline.edge_loss_weight=0 \
    adapter=lora \
    trainer=lora \
    trainer.max_training_steps=20000 \
    trainer.save_steps=100 \
    trainer.eval_steps=50 \
    trainer.enable_gradient_checkpoint=true \
    trainer.gradient_accumulation_steps=1 \
    trainer.batch_size_per_process=1 \
    trainer.num_inference_steps=50 \
    trainer.fsdp_strategy=no_shard