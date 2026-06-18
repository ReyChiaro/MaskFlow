import torch
from torchmetrics.image import PeakSignalNoiseRatio

from evaluator.register import REGISTER_METRIC
from .mask_utils import image_device, iter_image_pairs, mask_region_pair, mean_metric


@REGISTER_METRIC("PSNR")
def PSNR(source: torch.Tensor, target: torch.Tensor, **kwargs):
    psnr = PeakSignalNoiseRatio(data_range=(0.0, 1.0)).to(image_device(source))
    with torch.inference_mode():
        return mean_metric(
            [psnr(source_image, target_image).item() for source_image, target_image, _ in iter_image_pairs(source, target)]
        )


@REGISTER_METRIC("PSNR-FG")
def PSNR_FG(source: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, **kwargs):
    psnr = PeakSignalNoiseRatio(data_range=(0.0, 1.0)).to(image_device(source))
    with torch.inference_mode():
        values = []
        for source_image, target_image, mask_image in iter_image_pairs(source, target, mask):
            masked_source, masked_target = mask_region_pair(source_image, target_image, mask_image, foreground=True)
            values.append(psnr(masked_source, masked_target).item())
        return mean_metric(values)


@REGISTER_METRIC("PSNR-BG")
def PSNR_BG(source: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, **kwargs):
    psnr = PeakSignalNoiseRatio(data_range=(0.0, 1.0)).to(image_device(source))
    with torch.inference_mode():
        values = []
        for source_image, target_image, mask_image in iter_image_pairs(source, target, mask):
            masked_source, masked_target = mask_region_pair(source_image, target_image, mask_image, foreground=False)
            values.append(psnr(masked_source, masked_target).item())
        return mean_metric(values)
