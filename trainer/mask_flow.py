import os
import torch

from accelerate import Accelerator

from diffusers.loaders.peft import PeftAdapterMixin

from loguru import logger
from peft import LoraConfig
from typing import Union
from safetensors.torch import load_file

from pipelines.pipeline_manager import DenoiserInputs, DenoiserOutputs
from trainer.trainer import Trainer


class MaskFlowLoRATrainer(Trainer):

    def _init_adapter(self, accelerator: Accelerator):
        # Adapter configs is pairs of component -> list of target_modules
        cfgs = self.adapter_configs
        components = self.pipeline.components

        for c in components:
            lora_cfgs = getattr(cfgs, c, None)
            if lora_cfgs is None:
                continue
            lora_configs = LoraConfig(
                r=lora_cfgs.r,
                lora_alpha=lora_cfgs.lora_alpha,
                lora_dropout=lora_cfgs.lora_dropout,
                bias="none",
                target_modules=list(lora_cfgs.target_modules),
            )
            module: Union[PeftAdapterMixin | torch.nn.Module] = getattr(self.pipeline, c)
            module.requires_grad_(False)
            module.add_adapter(lora_configs, adapter_name=lora_cfgs.adapter_name)
            module.set_adapter(lora_cfgs.adapter_name)
            for n, p in module.named_parameters():
                if lora_cfgs.adapter_name in n:
                    p.requires_grad_(True)
            setattr(self.pipeline, c, module)
            logger.info(f"Add LoRA adapter to {c}.")
        logger.info(f"Adapter initialized.")

    def save_checkpoints(self, global_step: int):
        trainable_modules = self.pipeline.trainable_modules
        align_w = len(f"{self.training_steps_per_process}")
        checkpoint_dir = os.path.join(self.checkpoint_dir, f"step-{global_step:{align_w}d}")
        os.makedirs(checkpoint_dir)
        for m in trainable_modules:
            lora_cfgs = getattr(self.adapter_configs, m)
            unwrap_m: PeftAdapterMixin = self.unwrap_model(getattr(self.pipeline, m)).to(torch.float32)
            unwrap_m.save_lora_adapter(
                os.path.join(checkpoint_dir, f"{m}_lora"),
                adapter_name=lora_cfgs.adapter_name,
                upcast_before_saving=True,
                safe_serialization=True,
                weight_name=f"{lora_cfgs.adapter_name}.safetensors",
            )
            logger.info(f"Adapter in {m} saved to {checkpoint_dir}")

    def load_checkpoints(
        self,
        module_name: str,
        checkpoint_path: str,
        adapter_name: str = "default",
    ):
        state_dict = load_file(checkpoint_path)
        module: Union[PeftAdapterMixin | torch.nn.Module] = getattr(self.pipeline, module_name)
        module.load_lora_adapter(
            state_dict,
            prefix=None,
            adapter_name=adapter_name,
            use_safetensors=True,
        )
        module.set_adapter(adapter_name)
        setattr(self.pipeline, module)

    @torch.no_grad()
    def eval_step(self, data_loader, context=None):
        return super().eval_step(data_loader, context)

    def forwrad_step(self, accelerator, batch, context=None) -> torch.Tensor:
        with torch.no_grad():
            batch = self.pipeline.preprocess_everything(batch, context["device"], context["train_dtype"])
            # Encode everything
            prompt_embeds, prompt_embeds_mask = self.pipeline.encode_prompt(
                prompt=batch["prompt"],
                image_conditions=batch["conditions_to_text_encoder"],
                device=context["device"] or accelerator.device,
            )
            conditions_latents = [
                self.pipeline.encode_image(
                    image=cond_img,
                    generator=context["generator"],
                    sample_mode="sample",
                    device=context["device"],
                    dtype=context["train_dtype"],
                )
                for cond_img in batch["conditions"]
            ]
            target_latents = self.pipeline.encode_image(
                image=batch["target"],
                generator=context["generator"],
                sample_mode="sample",
                device=context["device"],
                dtype=context["train_dtype"],
            )

        denoiser_inputs: DenoiserInputs = self.pipeline.preprocess_denoiser_inputs(
            prompt_embeds=prompt_embeds,
            target_latents=target_latents,
            prompt_embeds_mask=prompt_embeds_mask,
            image_condition_latents=conditions_latents,
            generator=context["generator"],
            device=context["device"],
            dtype=context["train_dtype"],
        )
        denoiser_outputs: DenoiserOutputs = self.pipeline.denoise(denoiser_inputs)
        return denoiser_outputs.loss

    @torch.inference_mode()
    def generate(self):
        return super().generate()

    def __init__(
        self,
        data_loader_workers,
        batch_size_per_process,
        training_steps_per_process,
        save_steps_per_process,
        eval_steps_per_process,
        pipeline_configs,
        optimizer_configs,
        train_data_configs,
        output_dir="outputs",
        project_name="outputs/project-train",
        backup_dir=None,
        checkpoint_dir="outputs/checkpoints",
        evaluation_dir="outputs/evaluations",
        log_dir="logs",
        random_seed=0,
        num_epochs=None,
        num_warmup_steps_per_process=None,
        max_grad_norm=1,
        eval_data_configs=None,
        adapter_configs=None,
        lr_scheduler_configs=None,
        cudnn_deterministic: bool = False,
        cudnn_benchmark: bool = True,
    ):
        super().__init__(
            data_loader_workers,
            batch_size_per_process,
            training_steps_per_process,
            save_steps_per_process,
            eval_steps_per_process,
            pipeline_configs,
            optimizer_configs,
            train_data_configs,
            output_dir,
            project_name,
            backup_dir,
            checkpoint_dir,
            evaluation_dir,
            log_dir,
            random_seed,
            num_epochs,
            num_warmup_steps_per_process,
            max_grad_norm,
            eval_data_configs,
            adapter_configs,
            lr_scheduler_configs,
            cudnn_benchmark,
            cudnn_deterministic,
        )
