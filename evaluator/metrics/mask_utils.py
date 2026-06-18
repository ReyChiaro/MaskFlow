import torch

ImageInput = torch.Tensor | list[torch.Tensor]


def image_list(images: ImageInput) -> list[torch.Tensor]:
    if isinstance(images, torch.Tensor):
        if images.ndim == 3:
            return [images]
        if images.ndim == 4:
            return [image for image in images]
        raise ValueError(f"Expected image tensor with shape [C, H, W] or [N, C, H, W], got {tuple(images.shape)}.")
    return images


def image_device(images: ImageInput) -> torch.device:
    images = image_list(images)
    if not images:
        return torch.device("cpu")
    return images[0].device


def optional_mask_list(mask: ImageInput | None) -> list[torch.Tensor] | None:
    if mask is None:
        return None
    return image_list(mask)


def iter_image_pairs(
    source: ImageInput,
    target: ImageInput,
    mask: ImageInput | None = None,
):
    source_list = image_list(source)
    target_list = image_list(target)
    mask_list = optional_mask_list(mask)

    if len(source_list) != len(target_list):
        raise ValueError(
            f"Given sources ({len(source_list)}) and targets ({len(target_list)}) should contain same num."
        )
    if mask_list is not None and len(source_list) != len(mask_list):
        raise ValueError(f"Given sources ({len(source_list)}) and masks ({len(mask_list)}) should contain same num.")

    for idx, (source_image, target_image) in enumerate(zip(source_list, target_list, strict=True)):
        if source_image.shape != target_image.shape:
            raise ValueError(
                f"Source and target at index {idx} should have the same shape, "
                f"got {tuple(source_image.shape)} and {tuple(target_image.shape)}."
            )

        pair_mask = None if mask_list is None else mask_list[idx]
        if pair_mask is not None and pair_mask.shape[-2:] != source_image.shape[-2:]:
            raise ValueError(
                f"Mask at index {idx} should match image spatial shape {tuple(source_image.shape[-2:])}, "
                f"got {tuple(pair_mask.shape[-2:])}."
            )

        yield source_image.unsqueeze(0), target_image.unsqueeze(0), (
            None if pair_mask is None else pair_mask.unsqueeze(0)
        )


def mean_metric(values: list[float]) -> float:
    if not values:
        return float("nan")
    return sum(values) / len(values)


def region_mask(mask: torch.Tensor, source: torch.Tensor, foreground: bool, binary_thresh: float = 0) -> torch.Tensor:
    mask = mask.to(device=source.device)
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.ndim != 4:
        raise ValueError(f"Expected mask with shape [N, H, W] or [N, 1, H, W], got {tuple(mask.shape)}.")
    if mask.shape[1] != 1:
        mask = mask[:, :1]
    if mask.shape[0] != source.shape[0] or mask.shape[-2:] != source.shape[-2:]:
        raise ValueError(f"Mask shape {tuple(mask.shape)} does not match image shape {tuple(source.shape)}.")

    mask = mask > binary_thresh
    if not foreground:
        mask = ~mask
    return mask.expand_as(source)


def mask_region_pair(
    source: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    foreground: bool,
    binary_thresh: float = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = region_mask(mask, source, foreground, binary_thresh).to(dtype=source.dtype)
    return source * mask, target * mask
