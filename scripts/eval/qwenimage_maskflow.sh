.venv/bin/python3 evaluate.py \
    --config-path configs \
    --config-name eval_maskflow \
    project=evaluation \
    project.project_name=eval_QwenImage-MaskFlow-r256 \
    pipeline=qwenimage_maskflow \
    pipeline.pretrained_model=/root/models/QwenImage-Edit-2511 \
    evalset=mask_edit \
    sft_adapter=qwenimage_lora \
    adapters.sft.path="outputs/experiments/QwenImage-MaskFlow-r256/20260908-144736/checkpoints/step-1250/lora_adapter/pytorch_lora_weights.safetensors" \
    base_seed=42 \
    num_inference_steps=50 \