import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from evaluator.register import REGISTER_METRIC
from .mask_utils import mask_region_pair


@REGISTER_METRIC("LPIPS")
def LPIPS(source: torch.Tensor, target: torch.Tensor, **kwargs):
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(source.device)
    with torch.inference_mode():
        return lpips(source, target).item()


@REGISTER_METRIC("LPIPS-FG")
def LPIPS_FG(source: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, **kwargs):
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(source.device)
    with torch.inference_mode():
        masked_source, masked_target = mask_region_pair(source, target, mask, foreground=True)
        return lpips(masked_source, masked_target).item()


@REGISTER_METRIC("LPIPS-BG")
def LPIPS_BG(source: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, **kwargs):
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(source.device)
    with torch.inference_mode():
        masked_source, masked_target = mask_region_pair(source, target, mask, foreground=False)
        return lpips(masked_source, masked_target).item()
