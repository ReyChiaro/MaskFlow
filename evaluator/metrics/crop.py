"""Foreground bounding-box views for the existing whole-image metrics."""

from collections.abc import Sequence
from functools import wraps

import torch
import torch.nn.functional as F

from .mask_utils import image_list


def foreground_bbox(mask: torch.Tensor) -> tuple[int, int, int, int]:
    """Return exclusive (top, left, bottom, right) bounds of mask > 0.

    Use the first channel, as other regional metrics do. Empty foregrounds have
    no bounding box and are reported rather than silently scoring the full image.
    """
    rows, columns = torch.where(mask[0] > 0)
    if rows.numel() == 0:
        raise ValueError("CROP metrics require a nonempty foreground mask.")
    return rows.min().item(), columns.min().item(), rows.max().item() + 1, columns.max().item() + 1


class ForegroundCrops(Sequence):
    """Lazily align images to shared mask grids and crop shared bounding boxes.

    Each region contains the mask's (height, width) and exclusive bbox. Both
    comparison images use these same regions, preserving identical crop sizes
    and aspect ratios. Crops retain all pixels inside the box, including holes.
    """

    def __init__(self, images, regions):
        self.images, self.regions = image_list(images), regions
        if len(self.images) != len(regions):
            raise ValueError("CROP metrics require one mask per image.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        size, (top, left, bottom, right) = self.regions[index]
        image = self.images[index]
        if image.shape[-2:] != size:
            image = F.interpolate(
                image.unsqueeze(0).float(), size=size, mode="bilinear", align_corners=False, antialias=True
            ).squeeze(0)
        return image[:, top:bottom, left:right]


def crop_metric(metric):
    """Wrap a whole-image metric without changing its aggregation or model loading.

    In particular, FID still receives the complete pair of image distributions,
    and CLIP-TEXT retains the original prompt order without requiring targets.
    """
    @wraps(metric)
    def compute(source, target=None, mask=None, **kwargs):
        if mask is None:
            raise ValueError("CROP metrics require a mask.")
        regions = [(image.shape[-2:], foreground_bbox(image)) for image in image_list(mask)]
        source = ForegroundCrops(source, regions)
        target = ForegroundCrops(target, regions) if target is not None else None
        return metric(source, target, **kwargs)

    return compute
