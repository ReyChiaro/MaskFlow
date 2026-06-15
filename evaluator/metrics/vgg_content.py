import torch
import torch.nn.functional as F
from torchvision.models import VGG16_Weights, vgg16

from evaluator.register import REGISTER_METRIC
from .mask_utils import mask_region_pair

_VGG_CONTENT_MODELS: dict[str, torch.nn.Module] = {}


def _vgg_content_features(device: torch.device) -> torch.nn.Module:
    device_key = str(device)
    if device_key in _VGG_CONTENT_MODELS:
        return _VGG_CONTENT_MODELS[device_key]

    model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16].to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    _VGG_CONTENT_MODELS[device_key] = model
    return model


def _imagenet_normalize(images: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    return (images - mean) / std


def _vgg_content_loss(source: torch.Tensor, target: torch.Tensor) -> float:
    model = _vgg_content_features(source.device)
    source_features = model(_imagenet_normalize(source))
    target_features = model(_imagenet_normalize(target))
    return F.mse_loss(source_features, target_features).item()


@REGISTER_METRIC("VGG-CONTENT")
def VGG_CONTENT(source: torch.Tensor, target: torch.Tensor, **kwargs):
    with torch.inference_mode():
        return _vgg_content_loss(source, target)


@REGISTER_METRIC("VGG-CONTENT-FG")
def VGG_CONTENT_FG(source: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, **kwargs):
    with torch.inference_mode():
        masked_source, masked_target = mask_region_pair(source, target, mask, foreground=True)
        return _vgg_content_loss(masked_source, masked_target)


@REGISTER_METRIC("VGG-CONTENT-BG")
def VGG_CONTENT_BG(source: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, **kwargs):
    with torch.inference_mode():
        masked_source, masked_target = mask_region_pair(source, target, mask, foreground=False)
        return _vgg_content_loss(masked_source, masked_target)
