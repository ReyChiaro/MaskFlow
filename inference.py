import hydra
import torch
import argparse
import torchvision.transforms.functional as T

from pathlib import Path
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig
from PIL import Image

from trainer.lora_utils import load_inference_loras


def parse_args():
    parser = argparse.ArgumentParser("Pipeline Inference Interface")

    # -------- Pipeline Configs -------- #
    parser.add_argument("--model", type=str, help="Base model name or weights path.")
    parser.add_argument("--model-config", type=str, help="Config file (yaml) to init model.")

    # -------- Generation Configs -------- #
    parser.add_argument("--batched")

    return parser.parse_args()


def load_image(path: str) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    return T.to_tensor(image).unsqueeze(0)


def build_pipeline(cfg: DictConfig, device: torch.device, dtype: torch.dtype):
    generator = torch.Generator(device=device).manual_seed(cfg.runtime.seed)
    pipeline = instantiate(
        cfg.pipeline,
        device=device,
        dtype=dtype,
        generator=generator,
    )

    load_inference_loras(pipeline.transformer, cfg.checkpoint)
    pipeline.vae.eval()
    pipeline.text_pipeline.text_encoder.eval()
    pipeline.transformer.eval()
    return pipeline


def main(cfg: DictConfig) -> None:
    device = torch.device(cfg.runtime.device)
    dtype = getattr(torch, cfg.runtime.dtype)
    pipeline = build_pipeline(cfg, device, dtype)

    source = load_image(cfg.input.source)
    mask = load_image(cfg.input.mask)
    result = pipeline.generate()

    output_path = Path(cfg.output.path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = result["output"][0].float().cpu().clamp(0, 1)
    T.to_pil_image(output).save(output_path)
    logger.info(f"Inference result saved to {output_path.resolve()}")


if __name__ == "__main__":
    main()
