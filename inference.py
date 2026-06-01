import os
import time
import torch
import hydra
import torchvision.transforms.functional as T

from PIL import Image
from hydra.utils import instantiate
from omegaconf import OmegaConf

from pipelines.qwenimage.qwenimage_mask_flow import QwenImageMaskFlow
from pipelines.qwenimage.qwenimage_edit_plus import QwenImageEditPlus

get_value = lambda cfg, key, default, cls: default if getattr(cfg, key, default) == default else cls(getattr(cfg, key))


@hydra.main(config_path="configs", config_name="inference", version_base="v1.2")
def inference(cfgs: OmegaConf):
    r"""
    Args:
        cfgs (OmegaConf): This configs will be filled automatically via hydra command line.

        Other acceptable configs:
        - prompt (default: "")
        - negative_prompt (default: None)
        - image (default: None)
        - mask (default: None)
        - height (default: 1024)
        - width (default: 1024)
        - cfg_scale (default: 0)

        - rank (default: 0)
        - num_inference_steps (default: 50)
        - lora_model (default: None)
        - lora_adapter_name (default: "default")
        - seed (default: current time)

    """
    prompt = get_value(cfgs, "prompt", "", str)
    negative_prompt = get_value(cfgs, "negative_prompt", None, str)
    image = get_value(cfgs, "image", None, list)
    mask = get_value(cfgs, "mask", None, str)

    height = get_value(cfgs, "height", None, int)
    width = get_value(cfgs, "width", None, int)
    cfg_scale = get_value(cfgs, "cfg_scale", 0.0, float)

    rank = get_value(cfgs, "rank", 0, int)
    num_inference_steps = get_value(cfgs, "num_inference_steps", 50, int)
    lora_model = get_value(cfgs, "lora_model", None, str)
    lora_adapter_name = get_value(cfgs, "lora_adapter_name", "default", str)
    seed = get_value(cfgs, "seed", int(time.time()), int)

    device = torch.device(f"cuda:{rank}")
    generator = torch.Generator(device).manual_seed(seed)
    dtype = torch.bfloat16

    pipe: QwenImageMaskFlow | QwenImageEditPlus = instantiate(
        cfgs.pipeline, generator=generator, device=device, dtype=dtype
    )

    if lora_model is not None and os.path.exists(lora_model):
        pipe.transformer.load_lora_adapter(lora_model, prefix=None, adapter_name=lora_adapter_name)
        pipe.transformer.set_adapter(lora_adapter_name)
        pipe.transformer.requires_grad_(False)

    if image is not None:
        image = [Image.open(i).convert("RGB") for i in image]
    if mask is not None:
        mask = Image.open(mask).convert("RGB")

        mask = T.to_tensor(mask)
        mask = torch.where(mask > 0.3, 1.0, 0.0)
        mask = T.to_pil_image(mask)

    output: Image.Image = pipe.generate(
        prompt=prompt,
        image=image,
        mask=mask,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        cfg_scale=cfg_scale,
        height=height,
        width=width,
    )
    output.save("output.png")


if __name__ == "__main__":
    inference()
