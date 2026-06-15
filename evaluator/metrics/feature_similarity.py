import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel, CLIPModel

from evaluator.register import REGISTER_METRIC
from .mask_utils import mask_region_pair

CLIP_MODEL_ID = "openai/clip-vit-large-patch14-336"
DINO_MODEL_ID = "facebook/dinov2-large"

_MODEL_CACHE: dict[tuple[str, str], tuple[object, torch.nn.Module]] = {}


def _load_model(model_id: str, device: torch.device) -> tuple[object, torch.nn.Module]:
    cache_key = (model_id, str(device))
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    processor = AutoImageProcessor.from_pretrained(model_id)
    if model_id == CLIP_MODEL_ID:
        model = CLIPModel.from_pretrained(model_id).to(device).eval()
    else:
        model = AutoModel.from_pretrained(model_id).to(device).eval()

    for param in model.parameters():
        param.requires_grad_(False)
    _MODEL_CACHE[cache_key] = (processor, model)
    return processor, model


def _preprocess_images(processor: object, images: torch.Tensor, device: torch.device) -> torch.Tensor:
    inputs = processor(
        images=images.detach().cpu().float(),
        return_tensors="pt",
        do_rescale=False,
    )
    return inputs["pixel_values"].to(device)


def _image_features(
    images: torch.Tensor,
    model_id: str,
    batch_size: int = 8,
) -> torch.Tensor:
    processor, model = _load_model(model_id, images.device)
    features = []

    for start in range(0, len(images), batch_size):
        pixel_values = _preprocess_images(processor, images[start : start + batch_size], images.device)
        if model_id == CLIP_MODEL_ID:
            batch_features = model.get_image_features(pixel_values=pixel_values)
        else:
            outputs = model(pixel_values=pixel_values)
            batch_features = getattr(outputs, "pooler_output", None)
            if batch_features is None:
                batch_features = outputs.last_hidden_state[:, 0]
        features.append(F.normalize(batch_features.float(), dim=-1))

    return torch.cat(features, dim=0)


def _feature_similarity(source: torch.Tensor, target: torch.Tensor, model_id: str, batch_size: int = 8) -> float:
    source_features = _image_features(source, model_id, batch_size=batch_size)
    target_features = _image_features(target, model_id, batch_size=batch_size)
    return (source_features * target_features).sum(dim=-1).mean().item()


@REGISTER_METRIC("CLIP")
def CLIP(source: torch.Tensor, target: torch.Tensor, clip_batch_size: int = 8, **kwargs):
    with torch.inference_mode():
        return _feature_similarity(source, target, CLIP_MODEL_ID, batch_size=clip_batch_size)


@REGISTER_METRIC("CLIP-FG")
def CLIP_FG(source: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, clip_batch_size: int = 8, **kwargs):
    with torch.inference_mode():
        masked_source, masked_target = mask_region_pair(source, target, mask, foreground=True)
        return _feature_similarity(masked_source, masked_target, CLIP_MODEL_ID, batch_size=clip_batch_size)


@REGISTER_METRIC("CLIP-BG")
def CLIP_BG(source: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, clip_batch_size: int = 8, **kwargs):
    with torch.inference_mode():
        masked_source, masked_target = mask_region_pair(source, target, mask, foreground=False)
        return _feature_similarity(masked_source, masked_target, CLIP_MODEL_ID, batch_size=clip_batch_size)


@REGISTER_METRIC("DINO")
def DINO(source: torch.Tensor, target: torch.Tensor, dino_batch_size: int = 8, **kwargs):
    with torch.inference_mode():
        return _feature_similarity(source, target, DINO_MODEL_ID, batch_size=dino_batch_size)


@REGISTER_METRIC("DINO-FG")
def DINO_FG(source: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, dino_batch_size: int = 8, **kwargs):
    with torch.inference_mode():
        masked_source, masked_target = mask_region_pair(source, target, mask, foreground=True)
        return _feature_similarity(masked_source, masked_target, DINO_MODEL_ID, batch_size=dino_batch_size)


@REGISTER_METRIC("DINO-BG")
def DINO_BG(source: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, dino_batch_size: int = 8, **kwargs):
    with torch.inference_mode():
        masked_source, masked_target = mask_region_pair(source, target, mask, foreground=False)
        return _feature_similarity(masked_source, masked_target, DINO_MODEL_ID, batch_size=dino_batch_size)
