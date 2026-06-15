import torch
from torchmetrics.image import PeakSignalNoiseRatio

from evaluator.register import REGISTER_METRIC


@REGISTER_METRIC("PSNR")
def PSNR(source: torch.Tensor, target: torch.Tensor):
    psnr = PeakSignalNoiseRatio(data_range=(0.0, 1.0))
    return psnr(source, target).item()
