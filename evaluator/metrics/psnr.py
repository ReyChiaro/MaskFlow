import torch
from torchmetrics.image import PeakSignalNoiseRatio

from evaluator.register import REGISTER_METRIC
from .mask_utils import mask_region_pair


@REGISTER_METRIC("PSNR")
def PSNR(source: torch.Tensor, target: torch.Tensor, **kwargs):
    psnr = PeakSignalNoiseRatio(data_range=(0.0, 1.0)).to(source.device)
    with torch.inference_mode():
        return psnr(source, target).item()


@REGISTER_METRIC("PSNR-FG")
def PSNR_FG(source: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, **kwargs):
    psnr = PeakSignalNoiseRatio(data_range=(0.0, 1.0)).to(source.device)
    with torch.inference_mode():
        masked_source, masked_target = mask_region_pair(source, target, mask, foreground=True)
        return psnr(masked_source, masked_target).item()


@REGISTER_METRIC("PSNR-BG")
def PSNR_BG(source: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, **kwargs):
    psnr = PeakSignalNoiseRatio(data_range=(0.0, 1.0)).to(source.device)
    with torch.inference_mode():
        masked_source, masked_target = mask_region_pair(source, target, mask, foreground=False)
        return psnr(masked_source, masked_target).item()
