"""Shared spatial mask operations; model-specific latent codecs stay in their pipelines."""

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as T
from tqdm import tqdm


def dilate_mask(mask, kernel_size):
    kernel_size += 1 - kernel_size % 2
    return F.max_pool2d(mask, kernel_size=kernel_size, padding=kernel_size // 2, stride=1)


def blur_mask(mask, kernel_size, sigma):
    kernel_size += 1 - kernel_size % 2
    blurred = T.gaussian_blur(mask, kernel_size=kernel_size, sigma=sigma)
    blurred[mask < 1] = blurred[mask < 1] * 2
    blurred[mask >= 1] = 1
    return blurred


def neighbor_sum(x: torch.Tensor) -> torch.Tensor:
    B, C, H, W = x.shape
    kernel = x.new_tensor(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
    ).view(1, 1, 3, 3)
    kernel = kernel.repeat(C, 1, 1, 1)
    return F.conv2d(x, kernel, padding=1, groups=C)


def poisson_refine(
    g: torch.Tensor,
    x_S: torch.Tensor,
    M: torch.Tensor,
    soft_M: torch.Tensor,
    poisson_lambda_e: float,
    poisson_lambda_s: float,
    poisson_num_iter: int,
    poisson_momentum: float,
    eps: float = 1e-6,
    disable_progress_bar: bool = False,
):
    r"""
    Args
        g   (Tensor): The guidance tensor
        x_S (Tensor): The source tensor to keep background
        M   (Tensor): Binary mask
        soft_M (Optional Tensor): Soften mask for blending
    TODO: The semantic and formulas should be checked for `M` and `soft_M`
    """
    B, C, H, W = g.shape
    M = M.float().expand(B, C, H, W)
    soft_M = soft_M.float().expand(B, C, H, W)

    # Initialization
    y = M * g + (1.0 - M) * x_S

    D = neighbor_sum(torch.ones_like(g))
    nsum_soft_M = neighbor_sum(soft_M)

    weight_sum = 0.5 * soft_M * D + 0.5 * nsum_soft_M

    nsum_g = neighbor_sum(g)
    nsum_g_soft_M = neighbor_sum(soft_M * g)
    weighted_nsum_g = 0.5 * soft_M * nsum_g + 0.5 * nsum_g_soft_M

    div_g = weight_sum * g - weighted_nsum_g

    x_S_out = (1.0 - M) * x_S
    nsum_x_S = neighbor_sum(x_S_out)
    nsum_x_S_soft_M = neighbor_sum(soft_M * x_S_out)
    soft_boundry = 0.5 * soft_M * nsum_x_S + 0.5 * nsum_x_S_soft_M

    diag = weight_sum + poisson_lambda_e * soft_M + poisson_lambda_s * (1.0 - soft_M)

    b = div_g + soft_boundry + poisson_lambda_e * soft_M * g + poisson_lambda_s * (1.0 - soft_M) * x_S

    # Solve linear
    for _ in tqdm(
        range(poisson_num_iter),
        desc="Solve Poisson",
        disable=disable_progress_bar,
    ):
        y_in = M * y
        nsum_y_in = neighbor_sum(y_in)
        nsum_y_in_soft_M = neighbor_sum(soft_M * y_in)

        y_next = (0.5 * soft_M * nsum_y_in + 0.5 * nsum_y_in_soft_M + b) / diag.clamp(min=eps)

        # Update masked area only
        y_next = M * y_next + (1.0 - M) * x_S

        y = poisson_momentum * y + (1.0 - poisson_momentum) * y_next

    # if soft_M is not None:
    #     soft_M = soft_M.expand_as(g)
    #     y = soft_M * y + (1.0 - soft_M) * x_S
    return y
