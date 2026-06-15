import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure

from evaluator.register import REGISTER_METRIC


@REGISTER_METRIC("SSIM")
def SSIM(source: torch.Tensor, target: torch.Tensor):
    ssim = StructuralSimilarityIndexMeasure()
    return ssim(source, target).item()
