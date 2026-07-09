export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=8

PIPELINE="qwenimage_edit_plus_2511"
BASE_MODEL="/data/nvme7/models/Qwen-Image-Edit-2511"
IMAGE_ROOT="dataset/MaskEdit/scene"

torchrun \
    --nnodes=1 \
    --nproc-per-node=1 \
    --master-addr 127.0.0.1 \
    --master-port 29666 \
    finetune.py \
    --config-path=configs \
    --config-name=train \
    project.project_name="qwenimage_edit_plus_lora" \
    trainset=mask_edit \
    evalset=mask_edit \
    trainset.image_root=$IMAGE_ROOT \
    trainset.data_file=$IMAGE_ROOT/train.jsonl \
    evalset.image_root=$IMAGE_ROOT \
    evalset.data_file=${IMAGE_ROOT}/runtime_testset2.jsonl \
    pipeline=$PIPELINE \
    pipeline.pretrained_model=$BASE_MODEL \
    adapter=lora \
    adapter.r=256 \
    adapter.lora_alpha=256 \
    adapter.adapter_name=default \
    trainer=lora \
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
    trainer.fsdp_strategy=no_shard