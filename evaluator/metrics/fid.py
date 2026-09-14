import torch
from torchmetrics.image.fid import FrechetInceptionDistance

from evaluator.register import REGISTER_METRIC

from .mask_utils import image_device, iter_image_pairs


@REGISTER_METRIC("FID")
def FID(source: torch.Tensor, target: torch.Tensor, **kwargs):
    fid = FrechetInceptionDistance(normalize=True).to(image_device(source))
    with torch.inference_mode():
        for source_image, target_image, _ in iter_image_pairs(source, target):
            fid.update(target_image, real=True)
            fid.update(source_image, real=False)
        return fid.compute().item()
