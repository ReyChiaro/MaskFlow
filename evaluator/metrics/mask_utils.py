import torch


def region_mask(mask: torch.Tensor, source: torch.Tensor, foreground: bool) -> torch.Tensor:
    mask = mask.to(device=source.device)
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.ndim != 4:
        raise ValueError(f"Expected mask with shape [N, H, W] or [N, 1, H, W], got {tuple(mask.shape)}.")
    if mask.shape[1] != 1:
        mask = mask[:, :1]
    if mask.shape[0] != source.shape[0] or mask.shape[-2:] != source.shape[-2:]:
        raise ValueError(f"Mask shape {tuple(mask.shape)} does not match image shape {tuple(source.shape)}.")

    mask = mask > 0.5
    if not foreground:
        mask = ~mask
    return mask.expand_as(source)


def mask_region_pair(
    source: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    foreground: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = region_mask(mask, source, foreground).to(dtype=source.dtype)
    return source * mask, target * mask
