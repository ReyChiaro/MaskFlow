from pathlib import Path

import hydra
import torch
import torchvision.transforms.functional as T
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig
from PIL import Image
from torchvision.utils import save_image

from data_module.utils import preprocess_images
from pipelines.base_pipeline import BasePipeline
from trainer.lora_utils import load_inference_loras


def load_image(path: str) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    # CPU float32 RGB [C, H, W], in [0, 1].
    return T.to_tensor(image)


def build_pipeline(cfg: DictConfig, device: torch.device, dtype: torch.dtype) -> BasePipeline:
    pipeline = instantiate(
        cfg.pipeline,
        device=device,
        dtype=dtype,
    )
    pipeline.load_pretrained_weights()

    load_inference_loras(pipeline.transformer, cfg.checkpoint, cfg.checkpoint.lora_scale)
    pipeline.vae.eval()
    pipeline.text_pipeline.text_encoder.eval()
    pipeline.transformer.eval()
    return pipeline


@hydra.main(version_base=None, config_path="configs", config_name="inference")
@torch.inference_mode()
def main(cfg: DictConfig) -> None:
    device = torch.device(cfg.runtime.device)
    dtype = getattr(torch, cfg.runtime.dtype)
    pipeline = build_pipeline(cfg, device, dtype)

    conditions, _ = preprocess_images(
        {"source": load_image(cfg.input.source), "mask": load_image(cfg.input.mask)}, **cfg.preprocessing
    )
    result = pipeline.eval_step(
        {
            "prompt": [cfg.input.prompt],
            "negative_prompt": [cfg.input.negative_prompt],
            "conditions": {key: image.unsqueeze(0) for key, image in conditions.items()},
        },
        num_inference_steps=cfg.runtime.num_inference_steps,
        text_cfg_scale=cfg.runtime.text_cfg_scale,
        mask_cfg_scale=cfg.runtime.mask_cfg_scale,
        interaction_cfg_scale=cfg.runtime.interaction_cfg_scale,
        seed=cfg.runtime.seed,
    )

    output_path = Path(cfg.output.path).with_suffix(".png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(result["output"][0].float().cpu(), output_path)
    logger.info(f"Inference result saved to {output_path.resolve()}")


if __name__ == "__main__":
    main()
