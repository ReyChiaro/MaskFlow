import torch
import hydra
import torchvision.transforms.functional as T

from torchvision.utils import save_image
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
        +lora_safetensors_dir (str): If FSDP checkpoint is given,
            it will be converted into safetensors for the first time
    """
    cfg_contents = "\n" + " Configs ".center(50, "=")
    cfg_contents += "\n" + OmegaConf.to_yaml(cfgs)
    cfg_contents += "\n" + "=" * 50
    logger.info(cfg_contents)

    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    seed = cfgs.base_seed
    generator = torch.Generator(device).manual_seed(seed)
    evaluate_dir = Path(cfgs.project.evaluation_dir)
    evaluate_dir.mkdir(exist_ok=True, parents=True)

    pipe: QwenImageMaskFlow = instantiate(cfgs.pipe_configs, device=device, generator=generator, dtype=dtype)
    pipe.transformer.requires_grad_(False)

    if cfgs.resume_from is not None and Path(cfgs.resume_from).exists():
        safetensors_dir = cfgs.resume_from
        if getattr(cfgs, "is_fsdp_checkpoint", False):
            import peft
            import torch.distributed.checkpoint as DCP
            from torch.distributed.checkpoint.state_dict import (
                get_model_state_dict,
                set_model_state_dict,
                StateDictOptions,
            )

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

            safetensors_dir = Path(getattr(cfgs, "lora_safetensors_dir", f"{cfgs.resume_from}/lora_adapter"))
            safetensors_dir.mkdir(exist_ok=True, parents=True)
            pipe.transformer.save_lora_adapter(
                safetensors_dir,
                adapter_name=cfgs.adapter.adapter_name,
                safe_serialization=True,
            )
            pipe.transformer.delete_adapters(cfgs.adapter.adapter_name)
            logger.info(f"Converted FSDP checkpoint {cfgs.resume_from} to LoRA safetensors at {safetensors_dir}.")

        pipe.transformer.load_lora_adapter(
            safetensors_dir,
            prefix=None,
            adapter_name=cfgs.adapter.adapter_name,
            use_safetensors=True,
        )
        pipe.transformer.set_adapter(cfgs.adapter.adapter_name)
    pipe.transformer.requires_grad_(False)

    dataset: MaskEditDataset = instantiate(cfgs.evalset)

    prefix = "c_"
    for i, sample in tqdm(enumerate(dataset), desc="Eval", total=len(dataset)):
        if i >= len(dataset):
            logger.info(f"Evaluation Finished.")
            break
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

        output_dict: dict[str, Image.Image] = pipe.generate(
            prompt=prompt,
            image=source,
            negative_prompt=negative_prompt,
            mask=mask,
            height=source.height,
            width=source.width,
            num_inference_steps=cfgs.num_inference_steps,
            cfg_scale=cfgs.cfg_scale,
        )

        output = output_dict["output"]
        output.save(evaluate_dir / f"{image_name}.png")

        # Concat and Save
        if target is None:
            logger.warning(f"Eval [{i+1}/{len(dataset)}] Target image not found, concat image will not be saved.")
            continue

        processed_mask = output_dict["processed_mask"]
        source = T.to_tensor(source)
        processed_mask = T.to_tensor(processed_mask)
        mask = T.to_tensor(mask)
        output = T.to_tensor(output)

        row1 = torch.cat([source, mask, processed_mask, target, output], dim=-1)

        colorred_mask1 = mask.clone()
        if colorred_mask1.shape[0] == 1:
            colorred_mask1 = colorred_mask1.repeat(3, 1, 1)
        colorred_mask1[0] = torch.where(processed_mask[0] == 1.0, 1.0, 0.0)
        colorred_mask1[1] = torch.where(processed_mask[1] == 1.0, 0.0, 0.0)
        colorred_mask1[2] = torch.where(processed_mask[2] == 1.0, 0.0, 0.0)

        colorred_mask2 = colorred_mask1.clone()
        colorred_mask2[0] = torch.where((processed_mask[0] > 0) & (processed_mask[0] < 1), 0.0, 0.0)
        colorred_mask2[1] = torch.where((processed_mask[1] > 0) & (processed_mask[1] < 1), 0.0, 0.0)
        colorred_mask2[2] = torch.where((processed_mask[2] > 0) & (processed_mask[2] < 1), 1.0, 0.0)

        # Draw the hard mask
        colorred_source = torch.where(processed_mask == 1, 0.3 * colorred_mask1 + 0.7 * source, source)
        colorred_target = torch.where(processed_mask == 1, 0.3 * colorred_mask1 + 0.7 * target, target)
        colorred_output = torch.where(processed_mask == 1, 0.3 * colorred_mask1 + 0.7 * output, output)

        # Draw the smooth mask
        colorred_source = torch.where(
            (processed_mask > 0) & (processed_mask < 1),
            0.3 * colorred_mask2 + 0.7 * source,
            colorred_source,
        )
        colorred_target = torch.where(
            (processed_mask > 0) & (processed_mask < 1),
            0.3 * colorred_mask2 + 0.7 * target,
            colorred_target,
        )
        colorred_output = torch.where(
            (processed_mask > 0) & (processed_mask < 1),
            0.3 * colorred_mask2 + 0.7 * output,
            colorred_output,
        )

        row2 = torch.cat([colorred_source, mask, processed_mask, colorred_target, colorred_output], dim=-1)

        save_image(torch.cat([row1, row2], dim=1), evaluate_dir / f"{prefix}{image_name}.png")


if __name__ == "__main__":
    evaluate()
