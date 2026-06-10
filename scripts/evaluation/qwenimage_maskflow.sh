export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=2

PIPELINE="qwenimage_mask_flow"
BASE_MODEL="/data/nvme7/models/Qwen-Image-Edit-2511"
IMAGE_ROOT="dataset/MaskEdit/scene"
LORA_MODEL="outputs/experiments/maskflow_lora/20260609-115651/checkpoints/step-6000/model"

torchrun \
    --nnodes=1 \
    --nproc-per-node=1 \
    --master-addr 127.0.0.1 \
    --master-port 29655 \
    evaluate.py \
    --config-path=configs \
    --config-name=evaluation \
    project.project_name="eval_maskflow_lora" \
    evalset=mask_edit \
    evalset.image_root=$IMAGE_ROOT \
    evalset.data_file=outputs/data_cache/testset.jsonl \
    pipeline=$PIPELINE \
    pipeline.pretrained_model=$BASE_MODEL \
    pipeline.mask_dilation_kernel=45 \
    pipeline.mask_blur_kernel=25 \
    pipeline.mask_blur_sigma=25.0 \
    pipeline.mask_edge_width=50 \
    pipeline.mask_loss_weight=0.5 \
    pipeline.edge_loss_weight=0 \
    adapter=lora \
    base_seed=42 \
    resume_from=$LORA_MODEL \
    +is_fsdp_checkpoint=true \
    +lora_safetensors_dir=pretrained_weights/lora_adapter \
    cfg_scale=1.0 \
    num_inference_steps=50