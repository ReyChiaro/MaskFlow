import torch
import torch.nn.functional as F

from evaluator.register import REGISTER_METRIC
from .mask_utils import iter_image_pairs, mask_region_pair, mean_metric


@REGISTER_METRIC("MSE")
def MSE(source: torch.Tensor, target: torch.Tensor, **kwargs):
    with torch.inference_mode():
        return mean_metric(
            [F.mse_loss(source_image, target_image).item() for source_image, target_image, _ in iter_image_pairs(source, target)]
        )


@REGISTER_METRIC("MSE-FG")
def MSE_FG(source: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, **kwargs):
    with torch.inference_mode():
        values = []
        for source_image, target_image, mask_image in iter_image_pairs(source, target, mask):
            masked_source, masked_target = mask_region_pair(source_image, target_image, mask_image, foreground=True)
            values.append(F.mse_loss(masked_source, masked_target).item())
        return mean_metric(values)


@REGISTER_METRIC("MSE-BG")
def MSE_BG(source: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, **kwargs):
    with torch.inference_mode():
        values = []
        for source_image, target_image, mask_image in iter_image_pairs(source, target, mask):
            masked_source, masked_target = mask_region_pair(source_image, target_image, mask_image, foreground=False)
            values.append(F.mse_loss(masked_source, masked_target).item())
        return mean_metric(values)
