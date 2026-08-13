from pathlib import Path

import hydra
import torch
import torchvision.transforms.functional as TF
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig
from PIL import Image

from trainer.lora_utils import load_inference_loras


def load_image(path: str) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    return TF.to_tensor(image).unsqueeze(0)


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


@hydra.main(config_path="configs", config_name="inference", version_base="1.3")
def main(cfg: DictConfig) -> None:
    device = torch.device(cfg.runtime.device)
    dtype = getattr(torch, cfg.runtime.dtype)
    pipeline = build_pipeline(cfg, device, dtype)

    source = load_image(cfg.input.source)
    mask = load_image(cfg.input.mask)
    batch = {
        "prompt": [cfg.input.prompt],
        "negative_prompt": [cfg.input.negative_prompt],
        "target": source,
        "conditions": {"source": source, "mask": mask},
    }

    result = pipeline.eval_step(
        batch=batch,
        num_inference_steps=cfg.runtime.num_inference_steps,
        text_cfg_scale=cfg.runtime.text_cfg_scale,
        mask_cfg_scale=cfg.runtime.mask_cfg_scale,
    )

    output_path = Path(cfg.output.path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = result["output"][0].float().cpu().clamp(0, 1)
    TF.to_pil_image(output).save(output_path)
    logger.info(f"Inference result saved to {output_path.resolve()}")


if __name__ == "__main__":
    main()
