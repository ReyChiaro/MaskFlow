LORA_MODEL_LIST=(
    "ablations/mf-baseline/20260617-052359/checkpoints/step-5000/model"
    "ablations/mf-mask_inp-0.1-1.0/20260617-052359/checkpoints/step-5000/model"
    "ablations/mf-mask_inp-0.0-0.9/20260617-052400/checkpoints/step-5000/model"
    "ablations/mf-mask_inp-0.1-0.9/20260617-052401/checkpoints/step-5000/model"
    "ablations/mf_ps-constant_p0.2/20260617-052400/checkpoints/step-5000/model"
    "ablations/mf_ps-constant_p0.5/20260617-052358/checkpoints/step-5000/model"
    "ablations/mf_ps-linear1.0-0.5/20260617-052359/checkpoints/step-5000/model"
    "ablations/mf_ps-linear1.0-0.0/20260617-052359/checkpoints/step-5000/model"
)


GPUS_CSV="0,1,2,3,4,5,6,7"
BASE_SEED="42"

PROJECT_NAME_PREFIX="eval_"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-evaluations}"

PIPELINE="qwenimage_mask_flow"
BASE_MODEL="Qwen/Qwen-Image-Edit-2511"
IMAGE_ROOT="dataset/MaskEdit/scene"
DATA_FILE="${IMAGE_ROOT}/test.jsonl"

ADAPTER_NAME="maskflow"

CFG_SCALE="4.0"
NUM_INFERENCE_STEPS=50


EVALUATION_DIR="${OUTPUT_DIR}/evaluations"
LOG_DIR="${OUTPUT_DIR}/logs"
SHARD_DIR="${OUTPUT_DIR}/data_shards"
mkdir -p "$EVALUATION_DIR" "$LOG_DIR" "$SHARD_DIR"