import torch
import torch.nn.functional as F

from evaluator.register import REGISTER_METRIC
from .mask_utils import mask_region_pair


@REGISTER_METRIC("MSE")
def MSE(source: torch.Tensor, target: torch.Tensor, **kwargs):
    with torch.inference_mode():
        return F.mse_loss(source, target).item()


@REGISTER_METRIC("MSE-FG")
def MSE_FG(source: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, **kwargs):
    with torch.inference_mode():
        masked_source, masked_target = mask_region_pair(source, target, mask, foreground=True)
        return F.mse_loss(masked_source, masked_target).item()


@REGISTER_METRIC("MSE-BG")
def MSE_BG(source: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, **kwargs):
    with torch.inference_mode():
        masked_source, masked_target = mask_region_pair(source, target, mask, foreground=False)
        return F.mse_loss(masked_source, masked_target).item()
