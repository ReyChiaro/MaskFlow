import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as T

from loguru import logger
from PIL import Image
from torchvision.utils import save_image

from pipelines.qwenimage.qwenimage_edit_plus import QwenImageEditPlus
from pipelines.qwenimage.qwenimage_mask_flow import QwenImageMaskFlow
from schedulers import MaskFlowScheduler, RectifiedFlowMatchingScheduler


PIPELINE_ALIASES = {
    "qwenimage_mask_flow": "qwenimage_mask_flow",
    "maskflow": "qwenimage_mask_flow",
    "qwenimage_edit_plus": "qwenimage_edit_plus",
    "qwenimage_edit_plus_2511": "qwenimage_edit_plus",
    "edit_plus": "qwenimage_edit_plus",
}


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_dtype(value: str) -> torch.dtype:
    dtypes = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        return dtypes[value.lower()]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(f"Unsupported dtype: {value}") from exc


def seed_everything(seed: int, device: torch.device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_rgb_tensor(path: str | Path) -> torch.Tensor:
    return T.to_tensor(Image.open(path).convert("RGB"))


def load_mask_tensor(path: str | Path, threshold: float | None = 0.5) -> torch.Tensor:
    mask = T.to_tensor(Image.open(path).convert("L"))
    if threshold is not None:
        mask = (mask >= threshold).to(mask.dtype)
    return mask.repeat(3, 1, 1)


def build_scheduler(args: argparse.Namespace) -> RectifiedFlowMatchingScheduler:
    scheduler_kwargs = dict(
        weighting_scheme=args.weighting_scheme,
        logit_normal_mean=args.logit_normal_mean,
        logit_normal_std=args.logit_normal_std,
        mode_scale=args.mode_scale,
        base_image_seq_len=args.base_image_seq_len,
        base_shift=args.base_shift,
        max_image_seq_len=args.max_image_seq_len,
        max_shift=args.max_shift,
        shift=args.shift,
        shift_power=args.shift_power,
        time_shift_type=args.time_shift_type,
        use_dynamic_shifting=args.use_dynamic_shifting,
    )
    if args.pipeline == "qwenimage_mask_flow":
        return MaskFlowScheduler(**scheduler_kwargs, unmask_with=args.unmask_with)
    return RectifiedFlowMatchingScheduler(**scheduler_kwargs)


def build_pipeline(
    args: argparse.Namespace,
    scheduler: RectifiedFlowMatchingScheduler,
    generator: torch.Generator,
    device: torch.device,
):
    common_kwargs = dict(
        pretrained_model=args.pretrained_model,
        scheduler=scheduler,
        generator=generator,
        device=device,
        dtype=args.dtype,
    )
    if args.pipeline == "qwenimage_edit_plus":
        return QwenImageEditPlus(**common_kwargs)

    return QwenImageMaskFlow(
        **common_kwargs,
        mask_dilation_kernel=args.mask_dilation_kernel,
        mask_blur_kernel=args.mask_blur_kernel,
        mask_blur_sigma=args.mask_blur_sigma,
        mask_edge_width=args.mask_edge_width,
        enable_vae_mask_encoding=args.enable_vae_mask_encoding,
        enable_masked_loss=args.enable_masked_loss,
        enable_pixel_blend=args.enable_pixel_blend,
        local_denoise_steps=[args.local_denoise_start, args.local_denoise_end],
        enable_local_denoise_train=False,
        enable_local_denoise_infer=args.enable_local_denoise_infer,
        enable_poisson_train=False,
        enable_poisson_infer=args.enable_poisson_infer,
        poisson_steps=[args.poisson_start, args.poisson_end],
        poisson_lambda_e=args.poisson_lambda_e,
        poisson_lambda_s=args.poisson_lambda_s,
        poisson_num_iter=args.poisson_num_iter,
        poisson_momentum=args.poisson_momentum,
    )


def load_lora_adapter(pipe: QwenImageEditPlus, args: argparse.Namespace):
    if args.lora_path is None:
        logger.info("No LoRA adapter provided; run with base model weights.")
        return

    lora_path = Path(args.lora_path)
    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA path does not exist: {lora_path}")

    pipe.transformer.load_lora_adapter(
        lora_path,
        prefix=None,
        adapter_name=args.adapter_name,
        use_safetensors=True,
    )
    pipe.transformer.set_adapter(args.adapter_name)
    logger.info(f"Loaded LoRA adapter '{args.adapter_name}' from {lora_path}.")


def build_batch(args: argparse.Namespace) -> dict:
    source = load_rgb_tensor(args.source)
    mask_threshold = None if args.mask_threshold < 0 else args.mask_threshold
    mask = load_mask_tensor(args.mask, mask_threshold)

    return {
        "prompt": [args.prompt],
        "negative_prompt": [args.negative_prompt],
        "target": source.unsqueeze(0),
        "conditions": {
            "source": source.unsqueeze(0),
            "mask": mask.unsqueeze(0),
        },
        "image_name": [Path(args.source).stem],
    }


def save_outputs(outputs: dict[str, torch.Tensor], output_path: str | Path, save_debug: bool):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(outputs["output"][0].detach().float().cpu().clamp(0, 1), output_path)

    if not save_debug:
        return

    debug_dir = output_path.with_suffix("")
    debug_dir.mkdir(parents=True, exist_ok=True)
    for name, tensor in outputs.items():
        save_image(tensor[0].detach().float().cpu().clamp(0, 1), debug_dir / f"{name}.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-image Qwen-Image inference without Hydra configs.")

    # Model
    parser.add_argument(
        "--pipeline",
        default="qwenimage_mask_flow",
        choices=sorted(PIPELINE_ALIASES),
        help="Pipeline to run. Defaults to qwenimage_mask_flow.",
    )
    parser.add_argument("--pretrained-model", required=True, help="Base Qwen-Image-Edit model path or HF id.")
    parser.add_argument("--lora-path", default=None, help="Optional safetensors LoRA adapter directory or file.")
    parser.add_argument("--adapter-name", default="maskflow", help="Adapter name used when loading LoRA.")

    # User inputs
    parser.add_argument("--source", required=True, help="Source image path.")
    parser.add_argument("--mask", required=True, help="Mask image path; white area is edited.")
    parser.add_argument("--prompt", required=True, help="Edit prompt.")
    parser.add_argument("--negative-prompt", default="", help="Negative prompt for CFG.")
    parser.add_argument("--output", required=True, help="Output image path.")

    # Inference args
    parser.add_argument("--device", default=None, help="Torch device, e.g. cuda:0 or cpu.")
    parser.add_argument("--dtype", type=parse_dtype, default=torch.bfloat16, help="bf16, fp16, or fp32.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--save-debug", action="store_true", help="Also save mask/edge/output tensors next to output.")

    # Mask image edit args
    parser.add_argument("--mask-dilation-kernel", type=int, default=75)
    parser.add_argument("--mask-blur-kernel", type=int, default=125)
    parser.add_argument("--mask-blur-sigma", type=float, default=25)
    parser.add_argument("--mask-edge-width", type=int, default=25)
    parser.add_argument("--enable-vae-mask-encoding", type=str2bool, default=True)
    parser.add_argument("--enable-masked-loss", type=str2bool, default=True)
    parser.add_argument("--enable-pixel-blend", type=str2bool, default=False)
    parser.add_argument("--enable-local-denoise-infer", type=str2bool, default=False)
    parser.add_argument("--local-denoise-start", type=float, default=0.0)
    parser.add_argument("--local-denoise-end", type=float, default=1.0)

    # Poisson refinement args
    parser.add_argument("--enable-poisson-infer", type=str2bool, default=True)
    parser.add_argument("--poisson-start", type=float, default=0.0)
    parser.add_argument("--poisson-end", type=float, default=1.0)
    parser.add_argument("--poisson-lambda-e", type=float, default=0.1)
    parser.add_argument("--poisson-lambda-s", type=float, default=1.0)
    parser.add_argument("--poisson-num-iter", type=int, default=50)
    parser.add_argument("--poisson-momentum", type=float, default=0.1)

    # Timestep/sigma scheduler
    parser.add_argument("--weighting-scheme", default="logit_normal", choices=["logit_normal", "mode"])
    parser.add_argument("--logit-normal-mean", type=float, default=0.0)
    parser.add_argument("--logit-normal-std", type=float, default=1.0)
    parser.add_argument("--mode-scale", type=float, default=1.29)
    parser.add_argument("--base-image-seq-len", type=int, default=256)
    parser.add_argument("--base-shift", type=float, default=0.5)
    parser.add_argument("--max-image-seq-len", type=int, default=8192)
    parser.add_argument("--max-shift", type=float, default=0.9)
    parser.add_argument("--shift", type=float, default=1.0)
    parser.add_argument("--shift-power", type=int, default=1)
    parser.add_argument("--time-shift-type", default="exponential", choices=["exponential", "linear"])
    parser.add_argument("--use-dynamic-shifting", type=str2bool, default=True)
    parser.add_argument(
        "--unmask-with",
        default="noisy_source",
        choices=["target", "source", "noisy_target", "noisy_source"],
    )

    return parser.parse_args()


def main():
    args = parse_args()
    args.pipeline = PIPELINE_ALIASES[args.pipeline]

    if args.local_denoise_start > args.local_denoise_end:
        raise ValueError(
            f"local denoise start must be <= end, got [{args.local_denoise_start}, {args.local_denoise_end}]."
        )
    if args.poisson_start > args.poisson_end:
        raise ValueError(f"poisson start must be <= end, got [{args.poisson_start}, {args.poisson_end}].")

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    seed_everything(args.seed, device)
    generator = torch.Generator(device).manual_seed(args.seed)

    scheduler = build_scheduler(args)
    pipe = build_pipeline(args, scheduler, generator, device)
    pipe.transformer.requires_grad_(False)
    pipe.transformer.eval()
    pipe.vae.eval()
    pipe.text_pipeline.text_encoder.eval()

    load_lora_adapter(pipe, args)
    pipe.transformer.requires_grad_(False)

    batch = build_batch(args)
    log_msg = (
        f"Run {args.pipeline} inference on {args.source} with {args.num_inference_steps} steps, "
        f"cfg_scale={args.cfg_scale}"
    )
    if args.pipeline == "qwenimage_mask_flow":
        log_msg += (
            f", unmask_with={args.unmask_with}, pixel_blend={args.enable_pixel_blend}, "
            f"poisson_infer={args.enable_poisson_infer}"
        )
    logger.info(log_msg + ".")

    with torch.inference_mode():
        outputs = pipe.eval_step(
            batch,
            num_inference_steps=args.num_inference_steps,
            cfg_scale=args.cfg_scale,
        )

    save_outputs(outputs, args.output, args.save_debug)
    logger.info(f"Saved output to {args.output}.")


if __name__ == "__main__":
    main()
