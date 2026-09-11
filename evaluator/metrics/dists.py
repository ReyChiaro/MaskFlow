import torch
from torchmetrics.functional.image.dists import DISTSNetwork

from evaluator.register import REGISTER_METRIC
from .mask_utils import RegionalImages, image_device, iter_image_pairs, mean_metric

_MODEL_CACHE = {}


def _load_model(device):
    key = str(device)
    if key not in _MODEL_CACHE:
        # Cache TorchMetrics' pretrained network: its functional helper rebuilds VGG per call.
        _MODEL_CACHE[key] = DISTSNetwork().to(device).eval().requires_grad_(False)
    return _MODEL_CACHE[key]


@REGISTER_METRIC("DISTS")
@torch.inference_mode()
def DISTS(source, target, **kwargs):
    """Mean full-image DISTS distance for RGB images in [0, 1]; lower is better."""
    model = _load_model(image_device(source))
    return mean_metric([
        model(prediction.float(), reference.float()).mean().item()
        for prediction, reference, _ in iter_image_pairs(source, target)
    ])


@REGISTER_METRIC("DISTS-FG")
@torch.inference_mode()
def DISTS_FG(source, target, mask=None, **kwargs):
    """Foreground DISTS with the background zeroed in both images."""
    return DISTS(RegionalImages(source, mask, True), RegionalImages(target, mask, True), **kwargs)


@REGISTER_METRIC("DISTS-BG")
@torch.inference_mode()
def DISTS_BG(source, target, mask=None, **kwargs):
    """Background DISTS with the foreground zeroed in both images."""
    return DISTS(RegionalImages(source, mask, False), RegionalImages(target, mask, False), **kwargs)
