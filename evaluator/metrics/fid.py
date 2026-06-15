import torch
from torchmetrics.image.fid import FrechetInceptionDistance

from evaluator.register import REGISTER_METRIC


@REGISTER_METRIC("FID")
def FID(source: torch.Tensor, target: torch.Tensor, **kwargs):
    fid = FrechetInceptionDistance(normalize=True).to(source.device)
    with torch.inference_mode():
        fid.update(target, real=True)
        fid.update(source, real=False)
        return fid.compute().item()
