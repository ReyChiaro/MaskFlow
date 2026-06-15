import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure

from evaluator.register import REGISTER_METRIC
from .mask_utils import mask_region_pair


@REGISTER_METRIC("SSIM")
def SSIM(source: torch.Tensor, target: torch.Tensor, **kwargs):
    ssim = StructuralSimilarityIndexMeasure(data_range=(0.0, 1.0)).to(source.device)
    with torch.inference_mode():
        return ssim(source, target).item()


@REGISTER_METRIC("SSIM-FG")
def SSIM_FG(source: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, **kwargs):
    ssim = StructuralSimilarityIndexMeasure(data_range=(0.0, 1.0)).to(source.device)
    with torch.inference_mode():
        masked_source, masked_target = mask_region_pair(source, target, mask, foreground=True)
        return ssim(masked_source, masked_target).item()


@REGISTER_METRIC("SSIM-BG")
def SSIM_BG(source: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, **kwargs):
    ssim = StructuralSimilarityIndexMeasure(data_range=(0.0, 1.0)).to(source.device)
    with torch.inference_mode():
        masked_source, masked_target = mask_region_pair(source, target, mask, foreground=False)
        return ssim(masked_source, masked_target).item()
