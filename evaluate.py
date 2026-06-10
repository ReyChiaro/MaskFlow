import torch
import hydra
import torchvision.transforms.functional as T

from pathlib import Path
from PIL import Image
from tqdm import tqdm
from hydra.utils import instantiate
from omegaconf import OmegaConf
from loguru import logger

from pipelines.qwenimage.qwenimage_mask_flow import QwenImageMaskFlow
from data_module.mask_edit_dataset import MaskEditDataset


@hydra.main(config_path="configs", config_name="evaluation", version_base="v1.2")
def evaluate(cfgs: OmegaConf):
    r"""
    Args: cfgs can include following options
        +is_fsdp_checkpoint (bool)
    """

    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    seed = cfgs.base_seed
    generator = torch.Generator(device).manual_seed(seed)
    evaluate_dir = Path(cfgs.project.evaluation_dir)
    evaluate_dir.mkdir(exist_ok=True, parents=True)

    pipe: QwenImageMaskFlow = instantiate(cfgs.pipe_configs, device=device, generator=generator, dtype=dtype)
    pipe.transformer.requires_grad_(False)

    if getattr(cfgs, "is_fsdp_checkpoint", False):
        import peft
        import torch.distributed.checkpoint as DCP
        from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict, StateDictOptions

        lora_configs = peft.LoraConfig(
            r=cfgs.adapter.r,
            lora_alpha=cfgs.adapter.lora_alpha,
            lora_dropout=cfgs.adapter.lora_dropout,
            bias="none",
            target_modules=list(cfgs.adapter.target_modules),
        )
        pipe.transformer.add_adapter(lora_configs, adapter_name=cfgs.adapter.adapter_name)
        pipe.transformer.set_adapter(cfgs.adapter.adapter_name)

        for n, p in pipe.transformer.named_parameters():
            if cfgs.adapter.adapter_name in n and "lora_" in n:
                p.requires_grad_(True)

        transformer_states = get_model_state_dict(
            pipe.transformer,
            options=StateDictOptions(full_state_dict=False, ignore_frozen_params=True),
        )
        DCP.load({"model": transformer_states}, checkpoint_id=str(cfgs.resume_from))
        set_model_state_dict(
            pipe.transformer,
            transformer_states,
            options=StateDictOptions(full_state_dict=False, ignore_frozen_params=True, strict=False),
        )

        safetensors_dir = Path(getattr(cfgs, "lora_safetensors_dir", evaluate_dir / "lora_adapter"))
        safetensors_dir.mkdir(exist_ok=True, parents=True)
        pipe.transformer.save_lora_adapter(
            safetensors_dir,
            adapter_name=cfgs.adapter.adapter_name,
            safe_serialization=True,
        )
        pipe.transformer.delete_adapters(cfgs.adapter.adapter_name)
        pipe.transformer.load_lora_adapter(safetensors_dir, prefix=None, adapter_name=cfgs.adapter.adapter_name)
        pipe.transformer.set_adapter(cfgs.adapter.adapter_name)
        pipe.transformer.requires_grad_(False)
        logger.info(f"Converted FSDP checkpoint {cfgs.resume_from} to LoRA safetensors at {safetensors_dir}.")
    else:
        # safetensors
        pipe.transformer.load_lora_adapter(cfgs.resume_from, prefix=None, adapter_name=cfgs.adapter.adapter_name)
        pipe.transformer.set_adapter(cfgs.adapter.adapter_name)

    dataset: MaskEditDataset = instantiate(cfgs.evalset)

    for i, sample in tqdm(enumerate(dataset), desc="Eval"):
        prompt = sample.get("prompt", "")
        negative_prompt = sample.get("negative_prompt", " ")
        conditions = sample.get("conditions", None)
        target = sample.get("target", None)
        image_name = sample.get("image_name", "eval-image")

        if conditions is None or not isinstance(conditions, (list, tuple)) or len(conditions) != 2:
            logger.warning(f"Eval [{i+1}/{len(dataset)}] Conditions not contain [source, mask]")
            continue

        source = T.to_pil_image(conditions[0].float())
        mask = T.to_pil_image(conditions[1].float())

        output: Image.Image = pipe.generate(
            prompt=prompt,
            image=source,
            negative_prompt=negative_prompt,
            mask=mask,
            height=source.height,
            width=source.width,
            num_inference_steps=cfgs.num_inference_steps,
            cfg_scale=cfgs.cfg_scale,
        )

        output.save(evaluate_dir / f"{image_name}.png")


if __name__ == "__main__":
    evaluate()
