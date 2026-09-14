import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel, CLIPModel

from evaluator.register import REGISTER_METRIC
from evaluator.metrics.mask_utils import iter_image_pairs, mask_region_pair, mean_metric

CLIP_MODEL_ID = "/root/intern-xr/models/clip-vit-large-patch14-336"
DINO_MODEL_ID = "/root/intern-xr/models/dinov2-large"

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
            outputs = model.get_image_features(pixel_values=pixel_values)
        else:
            outputs = model(pixel_values=pixel_values)
        batch_features = getattr(outputs, "pooler_output", None)
        if batch_features is None:
            batch_features = outputs.last_hidden_state[:, 0]
        features.append(F.normalize(batch_features.float(), dim=-1))

    return torch.cat(features, dim=0)


def _feature_similarity_pair(source: torch.Tensor, target: torch.Tensor, model_id: str, batch_size: int = 8) -> float:
    source_features = _image_features(source, model_id, batch_size=batch_size)
    target_features = _image_features(target, model_id, batch_size=batch_size)
    return (source_features * target_features).sum(dim=-1).mean().item()


def _feature_similarity(source: torch.Tensor, target: torch.Tensor, model_id: str, batch_size: int = 8) -> float:
    return mean_metric(
        [
            _feature_similarity_pair(source_image, target_image, model_id, batch_size=batch_size)
            for source_image, target_image, _ in iter_image_pairs(source, target)
        ]
    )


@REGISTER_METRIC("CLIP")
def CLIP(source: torch.Tensor, target: torch.Tensor, clip_batch_size: int = 8, **kwargs):
    with torch.inference_mode():
        return _feature_similarity(source, target, CLIP_MODEL_ID, batch_size=clip_batch_size)


@REGISTER_METRIC("CLIP-FG")
def CLIP_FG(source: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, clip_batch_size: int = 8, **kwargs):
    with torch.inference_mode():
        values = []
        for source_image, target_image, mask_image in iter_image_pairs(source, target, mask):
            masked_source, masked_target = mask_region_pair(source_image, target_image, mask_image, foreground=True)
            values.append(_feature_similarity_pair(masked_source, masked_target, CLIP_MODEL_ID, batch_size=clip_batch_size))
        return mean_metric(values)


@REGISTER_METRIC("CLIP-BG")
def CLIP_BG(source: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, clip_batch_size: int = 8, **kwargs):
    with torch.inference_mode():
        values = []
        for source_image, target_image, mask_image in iter_image_pairs(source, target, mask):
            masked_source, masked_target = mask_region_pair(source_image, target_image, mask_image, foreground=False)
            values.append(_feature_similarity_pair(masked_source, masked_target, CLIP_MODEL_ID, batch_size=clip_batch_size))
        return mean_metric(values)


@REGISTER_METRIC("DINO")
def DINO(source: torch.Tensor, target: torch.Tensor, dino_batch_size: int = 8, **kwargs):
    with torch.inference_mode():
        return _feature_similarity(source, target, DINO_MODEL_ID, batch_size=dino_batch_size)


@REGISTER_METRIC("DINO-FG")
def DINO_FG(source: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, dino_batch_size: int = 8, **kwargs):
    with torch.inference_mode():
        values = []
        for source_image, target_image, mask_image in iter_image_pairs(source, target, mask):
            masked_source, masked_target = mask_region_pair(source_image, target_image, mask_image, foreground=True)
            values.append(_feature_similarity_pair(masked_source, masked_target, DINO_MODEL_ID, batch_size=dino_batch_size))
        return mean_metric(values)


@REGISTER_METRIC("DINO-BG")
def DINO_BG(source: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, dino_batch_size: int = 8, **kwargs):
    with torch.inference_mode():
        values = []
        for source_image, target_image, mask_image in iter_image_pairs(source, target, mask):
            masked_source, masked_target = mask_region_pair(source_image, target_image, mask_image, foreground=False)
            values.append(_feature_similarity_pair(masked_source, masked_target, DINO_MODEL_ID, batch_size=dino_batch_size))
        return mean_metric(values)
