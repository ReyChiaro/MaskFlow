"""Boundary preservation and continuity on RGB images in [0, 1]."""

import math

import torch
import torch.nn.functional as F

from evaluator.register import REGISTER_METRIC

from .mask_utils import image_list, iter_image_pairs, mean_metric, region_mask

BOUNDARY_METRICS = frozenset({"SEAM", "SEAM_GRAD"})


def validate_dilation_size(dilation_size):
    """Size is the square morphology kernel width, not its radius."""
    if isinstance(dilation_size, bool) or not isinstance(dilation_size, int) or dilation_size < 3 or dilation_size % 2 == 0:
        raise ValueError("dilation_size must be an odd integer >= 3.")


def boundary_mask(mask, image, dilation_size=3):
    """Return dilation(mask > 0) minus erosion(mask > 0), on the image grid.

    Replicated padding avoids inventing a seam at the image frame. Full/empty
    masks have no foreground/background interface and are therefore undefined.
    The returned mask has one channel and includes both sides of the interface.
    """
    validate_dilation_size(dilation_size)
    if mask is None:
        raise ValueError("Boundary metrics require a mask.")
    binary = region_mask(mask, image, foreground=True)[:, :1].float()
    radius = dilation_size // 2
    padded = F.pad(binary, (radius, radius, radius, radius), mode="replicate")
    dilated = F.max_pool2d(padded, dilation_size, stride=1)
    eroded = -F.max_pool2d(-padded, dilation_size, stride=1)
    boundary = dilated > eroded
    if not boundary.flatten(1).any(dim=1).all():
        raise ValueError("Empty boundary: mask must contain both foreground and background.")
    return boundary


@REGISTER_METRIC("SEAM")
@torch.inference_mode()
def SEAM(source, target=None, mask=None, dilation_size=3, originals=None, **kwargs):
    """Mean per-image boundary MSE against originals (positional target alias).

    Despite the common metric signature, the reference must be the original
    source image, never the desired edited target. Lower is better.
    """
    reference = originals if originals is not None else target
    if reference is None:
        raise ValueError("SEAM requires originals (source images).")
    values = []
    for prediction, original, sample_mask in iter_image_pairs(source, reference, mask):
        boundary = boundary_mask(sample_mask, prediction, dilation_size).expand_as(prediction)
        error = (prediction.float() - original.float()).square()
        values.append(error.masked_select(boundary).mean().item())
    return mean_metric(values)


@REGISTER_METRIC("SEAM_GRAD")
@torch.inference_mode()
def SEAM_GRAD(source, target=None, mask=None, dilation_size=3, **kwargs):
    """Mean absolute 8-neighbor directional derivative at boundary pixels.

    Compute differences on the unmasked prediction. For each boundary pixel,
    average |I(p+d)-I(p)| / ||d|| over RGB and valid in-frame neighbors, then
    average boundary pixels and images equally. No reference is used. Lower
    means smoother, but natural edges/textures also increase this score.
    """
    images = image_list(source)
    values = []
    for prediction, _, sample_mask in iter_image_pairs(images, images, mask):
        boundary = boundary_mask(sample_mask, prediction, dilation_size)
        prediction = prediction.float()
        height, width = prediction.shape[-2:]
        magnitude = torch.zeros_like(prediction[:, :1])
        count = torch.zeros_like(magnitude)
        for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
            y0, y1 = max(0, -dy), min(height, height - dy)
            x0, x1 = max(0, -dx), min(width, width - dx)
            if y0 >= y1 or x0 >= x1:
                continue
            center = prediction[..., y0:y1, x0:x1]
            neighbor = prediction[..., y0 + dy:y1 + dy, x0 + dx:x1 + dx]
            derivative = (neighbor - center).abs().mean(dim=1, keepdim=True) / math.hypot(dy, dx)
            magnitude[..., y0:y1, x0:x1] += derivative
            count[..., y0:y1, x0:x1] += 1
        values.append((magnitude / count.clamp_min(1)).masked_select(boundary).mean().item())
    return mean_metric(values)
