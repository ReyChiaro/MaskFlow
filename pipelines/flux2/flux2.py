import dataclasses
import math

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLFlux2, Flux2Transformer2DModel
from diffusers.pipelines.flux2.pipeline_flux2 import Flux2Pipeline
from PIL import Image
from torchvision.transforms.functional import to_pil_image, to_tensor
from tqdm import tqdm

from pipelines.base_pipeline import BasePipeline, ForwardOutput, PreprocessOutput
from schedulers.flux2_flow_matching import Flux2FlowMatchingScheduler


@dataclasses.dataclass
class Flux2ForwardOutput(ForwardOutput):
    text_ids: torch.Tensor | None = None
    image_ids: torch.Tensor | None = None
    latent_ids: torch.Tensor | None = None
    negative_text_ids: torch.Tensor | None = None


@dataclasses.dataclass
class Flux2(BasePipeline):
    pretrained_model: str | None = None
    scheduler: Flux2FlowMatchingScheduler | None = None
    generator: torch.Generator | None = None
    device: torch.device | None = None
    dtype: torch.dtype | None = None

    max_area: int | None = None
    max_condition_area: int | None = None
    condition_keys: list[str] = dataclasses.field(default_factory=lambda: ["source"])
    max_sequence_length: int = 512
    text_encoder_out_layers: list[int] = dataclasses.field(default_factory=lambda: [10, 20, 30])
    guidance_scale: float = 4.0
    train_sample_mode: str = "argmax"
    rescale_cfg: bool = False

    vae: AutoencoderKLFlux2 = dataclasses.field(init=False, default=None)
    transformer: Flux2Transformer2DModel = dataclasses.field(init=False, default=None)
    text_pipeline: Flux2Pipeline = dataclasses.field(init=False, default=None)

    def __post_init__(self) -> None:
        self.initialize_pipeline(Flux2Pipeline)
        if self.scheduler.native_scheduler is None:
            self.scheduler.native_scheduler = self.text_pipeline.scheduler

    @property
    def vae_scale_factor(self):
        return self.text_pipeline.vae_scale_factor

    @property
    def latent_patch_size(self):
        return self.vae.config.patch_size

    def image_size(self, image, max_area=None):
        height, width = image.shape[-2:]
        scale = min(1.0, math.sqrt(max_area / (height * width))) if max_area else 1.0
        multiple_h = self.vae_scale_factor * self.latent_patch_size[0]
        multiple_w = self.vae_scale_factor * self.latent_patch_size[1]
        return max(multiple_h, int(height * scale) // multiple_h * multiple_h), max(
            multiple_w,
            int(width * scale) // multiple_w * multiple_w,
        )

    def resize_image(self, image, size, is_mask=False):
        image = image.to(self.device, dtype=self.dtype)
        if image.shape[-2:] != tuple(size):
            if is_mask:
                image = F.interpolate(image.float(), size=size, mode="nearest")
            else:
                image = F.interpolate(image.float(), size=size, mode="bicubic", align_corners=False, antialias=True)
        return image.clamp(0, 1).to(dtype=self.dtype)

    def preprocess_inputs(self, batch):
        target = batch.get("target")
        conditions = batch.get("conditions") or {}
        if not isinstance(conditions, dict):
            raise ValueError("FLUX.2 conditions must be a dictionary of batched images.")
        reference = target if target is not None else conditions.get("source")
        if reference is None:
            raise ValueError("Evaluation requires a target or conditions.source for output dimensions.")
        height, width = self.image_size(reference, self.max_area)
        if target is not None:
            target = self.resize_image(target, (height, width))
            target = self.image_processor.preprocess(target, height=height, width=width)
        dit_conditions = {}
        for key in self.condition_keys:
            if key in conditions:
                dit_conditions[key] = self.preprocess_condition(conditions[key])
        return PreprocessOutput(
            prompt=batch["prompt"],
            negative_prompt=batch.get("negative_prompt"),
            target=target,
            height=height,
            width=width,
            dit_conditions=dit_conditions,
        )

    def preprocess_condition(self, images):
        # Match native reference-image Lanczos downscaling and center cropping exactly.
        processed = []
        for image in images:
            image = to_pil_image(image.detach().float().cpu())
            if self.max_condition_area and image.width * image.height > self.max_condition_area:
                image = self.image_processor._resize_to_target_area(image, self.max_condition_area)
            height, width = self.image_processor.get_default_height_width(image)
            processed.append(self.image_processor.preprocess(image, height=height, width=width, resize_mode="crop"))
        return torch.cat(processed).to(self.device, dtype=self.dtype)

    @torch.no_grad()
    def encode_prompt(self, prompt):
        return self.text_pipeline.encode_prompt(
            prompt=prompt,
            device=self.device,
            max_sequence_length=self.max_sequence_length,
            text_encoder_out_layers=tuple(self.text_encoder_out_layers),
        )

    def latent_statistics(self, latents):
        mean = self.vae.bn.running_mean.to(latents).view(1, -1, 1, 1)
        std = (self.vae.bn.running_var.to(latents).view(1, -1, 1, 1) + self.vae.config.batch_norm_eps).sqrt()
        return mean, std

    @torch.no_grad()
    def encode_image(self, image, sample_mode="argmax"):
        distribution = self.vae.encode(image).latent_dist
        if sample_mode == "argmax":
            latents = distribution.mode()
        elif sample_mode == "sample":
            latents = distribution.sample(generator=self.generator)
        else:
            raise ValueError(f"Unsupported VAE sample mode: {sample_mode}")
        latents = Flux2Pipeline._patchify_latents(latents)
        mean, std = self.latent_statistics(latents)
        return (latents - mean) / std

    @torch.no_grad()
    def decode_image(self, latents):
        mean, std = self.latent_statistics(latents)
        latents = Flux2Pipeline._unpatchify_latents(latents * std + mean)
        decoded = self.vae.decode(latents, return_dict=False)[0]
        return self.image_processor.postprocess(decoded, output_type="pt")

    def encode_conditions(self, data, sample_mode="argmax"):
        """Encode reference images with the same VAE sampling mode as the target."""
        latents = [self.encode_image(image, sample_mode) for image in data.dit_conditions.values()]
        packed = [Flux2Pipeline._pack_latents(image) for image in latents]
        ids = None
        if latents:
            # Native helper accepts one sample's reference images, not a batch of unrelated images.
            ids = torch.cat(
                [
                    Flux2Pipeline._prepare_image_ids([image[index : index + 1] for image in latents])
                    for index in range(latents[0].shape[0])
                ]
            ).to(self.device)
        return packed, ids

    @torch.no_grad()
    def _prepare_inputs(
        self,
        data: PreprocessOutput,
        sample_mode: str = "argmax",
        seed: int | list[int] | None = None,
    ) -> Flux2ForwardOutput:
        """Prepare shared text, reference latents, noise and coordinates for train/eval."""
        prompt_embeds, text_ids = self.encode_prompt(data.prompt)
        height, width = data.target.shape[-2:] if data.target is not None else (data.height, data.width)
        generators = self.generator
        if seed is not None:
            seeds = [seed] * len(data.prompt) if isinstance(seed, int) else seed
            generators = [torch.Generator(self.device).manual_seed(value) for value in seeds]
        noise, latent_ids = self.text_pipeline.prepare_latents(
            batch_size=len(data.prompt),
            num_latents_channels=self.vae.config.latent_channels,
            height=height,
            width=width,
            dtype=self.dtype,
            device=self.device,
            generator=generators,
        )
        conditions, condition_ids = self.encode_conditions(data, sample_mode)
        image_ids = torch.cat([latent_ids, condition_ids], dim=1) if condition_ids is not None else latent_ids
        result = Flux2ForwardOutput(
            prompt_embeds=prompt_embeds,
            text_ids=text_ids,
            image_ids=image_ids,
            latent_ids=latent_ids,
            conditions=conditions,
            noise=noise,
            height=height,
            width=width,
        )
        return result

    @torch.no_grad()
    def prepare_eval_inputs(
        self,
        data: PreprocessOutput,
        text_cfg_scale: float = 1.0,
        seed: int | list[int] | None = None,
    ) -> Flux2ForwardOutput:
        """Use deterministic VAE encodings and optionally encode negative text for CFG."""
        result = self._prepare_inputs(data, seed=seed)
        if text_cfg_scale > 1:
            negative_prompt = data.negative_prompt or [""] * len(data.prompt)
            result.negative_prompt_embeds, result.negative_text_ids = self.encode_prompt(negative_prompt)
        return result

    @torch.no_grad()
    def prepare_forward_inputs(self, data):
        if data.target is None:
            raise ValueError("Training requires target images.")
        result = self.prepare_eval_inputs(data)
        target = Flux2Pipeline._pack_latents(self.encode_image(data.target, self.train_sample_mode))
        ts = self.scheduler.sample_timesteps(target.shape[0], self.generator, self.device)
        result.sigmas = self.scheduler.get_sigmas(ts, target.shape[1])
        result.timesteps = result.sigmas
        result.noised_target = self.scheduler.add_noise_by_sigmas(result.noise, target, result.sigmas)
        result.ground_truth = self.scheduler.get_velocity(result.noise, target)
        return result

    def denoise(self, xt, timestep, inputs, negative=False):
        hidden_states = torch.cat([xt] + inputs.conditions, dim=1)
        guidance = None
        if self.transformer.config.guidance_embeds:
            guidance = torch.full((xt.shape[0],), self.guidance_scale, device=xt.device, dtype=torch.float32)
        prediction = self.transformer(
            hidden_states=hidden_states,
            timestep=timestep.to(xt).expand(xt.shape[0]),
            guidance=guidance,
            encoder_hidden_states=inputs.negative_prompt_embeds if negative else inputs.prompt_embeds,
            txt_ids=inputs.negative_text_ids if negative else inputs.text_ids,
            img_ids=inputs.image_ids,
            return_dict=False,
        )[0]
        return prediction[:, : xt.shape[1]].to(xt)

    def compute_loss(self, prediction, inputs):
        return {"loss": F.mse_loss(prediction.float(), inputs.ground_truth.float())}

    def forward_step(self, batch):
        data = self.preprocess_inputs(batch)
        inputs = self.prepare_forward_inputs(data)
        prediction = self.denoise(inputs.noised_target, inputs.timesteps, inputs)
        return self.compute_loss(prediction, inputs)

    def predict_velocity(self, xt, timestep, inputs, text_cfg_scale):
        prediction = self.denoise(xt, timestep, inputs)
        if inputs.negative_prompt_embeds is not None:
            negative = self.denoise(xt, timestep, inputs, negative=True)
            combined = negative + text_cfg_scale * (prediction - negative)
            if self.rescale_cfg:
                combined = combined * (
                    prediction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                    / combined.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                )
            prediction = combined
        return prediction

    def inference_step(self, xt, prediction, sigma, next_sigma, derivative, inputs):
        return self.scheduler.step(xt, prediction, sigma, next_sigma, derivative)

    def postprocess_output(self, output, data):
        return {"output": output}

    @torch.inference_mode()
    def eval_step(
        self,
        batch: dict,
        num_inference_steps: int = 50,
        text_cfg_scale: float = 1.0,
        seed: int | list[int] = 42,
        **cfg_kwargs,
    ) -> dict[str, torch.Tensor]:
        data = self.preprocess_inputs(batch)
        inputs = self.prepare_eval_inputs(data, text_cfg_scale, seed=seed, **cfg_kwargs)
        xt = inputs.noised_target if inputs.noised_target is not None else inputs.noise
        with self.scheduler.inference(num_inference_steps, xt.shape[1]) as inferencer:
            for timestep, sigma, next_sigma, derivative in tqdm(inferencer, total=num_inference_steps):
                timestep, sigma, next_sigma, derivative = [
                    value.to(self.device) for value in (timestep, sigma, next_sigma, derivative)
                ]
                prediction = self.predict_velocity(xt, timestep, inputs, text_cfg_scale, **cfg_kwargs)
                xt = self.inference_step(xt, prediction, sigma, next_sigma, derivative, inputs)
        output = self.decode_image(Flux2Pipeline._unpack_latents_with_ids(xt, inputs.latent_ids))
        return self.postprocess_output(output, data)

    @torch.inference_mode()
    def generate(
        self,
        prompt,
        source_image: Image.Image | None = None,
        mask_image: Image.Image | None = None,
        negative_prompt="",
        height=None,
        width=None,
        num_inference_steps=50,
        text_cfg_scale=1.0,
        **cfg_kwargs,
    ):
        conditions = {}
        for key, image in (("source", source_image), ("mask", mask_image)):
            if image is not None:
                conditions[key] = to_tensor(image.convert("RGB")).unsqueeze(0)
        if height is None or width is None:
            if source_image is None:
                raise ValueError("height and width are required when no source image is supplied.")
            width = width or source_image.width
            height = height or source_image.height
        batch = {
            "prompt": [prompt],
            "negative_prompt": [negative_prompt],
            "conditions": conditions,
            "target": torch.zeros(1, self.vae.config.in_channels, height, width),
        }
        return self.eval_step(batch, num_inference_steps, text_cfg_scale, **cfg_kwargs)
