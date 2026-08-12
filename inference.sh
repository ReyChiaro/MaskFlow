SCENE_ROOT="./dataset/MaskEdit/scene"
INFOG_ROOT="./dataset/MaskEdit/infographics_en"
DATA_ROOT=${SCENE_ROOT}

python inference.py \
    pipeline.pretrained_model=/root/models/Qwen-Image-Edit-2511 \
    checkpoint.sft_path=outputs/experiments/maskflow_maskcfg/20260811-152753/checkpoints/step-1250/lora_adapter/pytorch_lora_weights.safetensors \
    checkpoint.sft_adapter_name=maskflow \
    checkpoint.path=outputs/experiments/dmd_maskflow_TCFG4/20260812-020538/checkpoints/step-50/student_lora/pytorch_lora_weights.safetensors \
    checkpoint.adapter_name=dmd \
    input.source=${DATA_ROOT}/source/000000172637.png \
    input.mask=${DATA_ROOT}/mask/000000172637_0.png \
    'input.prompt="Replace the objects in the masked area with a row of red fire hydrants of similar size and spacing along the sidewalk."' \
    runtime.num_inference_steps=8 \

