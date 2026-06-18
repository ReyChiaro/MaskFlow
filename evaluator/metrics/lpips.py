import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from evaluator.register import REGISTER_METRIC
from .mask_utils import image_device, iter_image_pairs, mask_region_pair, mean_metric


@REGISTER_METRIC("LPIPS")
def LPIPS(source: torch.Tensor, target: torch.Tensor, **kwargs):
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(image_device(source))
    with torch.inference_mode():
        return mean_metric(
            [lpips(source_image, target_image).item() for source_image, target_image, _ in iter_image_pairs(source, target)]
        )


@REGISTER_METRIC("LPIPS-FG")
def LPIPS_FG(source: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, **kwargs):
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(image_device(source))
    with torch.inference_mode():
        values = []
        for source_image, target_image, mask_image in iter_image_pairs(source, target, mask):
            masked_source, masked_target = mask_region_pair(source_image, target_image, mask_image, foreground=True)
            values.append(lpips(masked_source, masked_target).item())
        return mean_metric(values)


@REGISTER_METRIC("LPIPS-BG")
def LPIPS_BG(source: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, **kwargs):
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(image_device(source))
    with torch.inference_mode():
        values = []
        for source_image, target_image, mask_image in iter_image_pairs(source, target, mask):
            masked_source, masked_target = mask_region_pair(source_image, target_image, mask_image, foreground=False)
            values.append(lpips(masked_source, masked_target).item())
        return mean_metric(values)
