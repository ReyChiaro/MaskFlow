import torch
import torch.nn.functional as F

from dataclasses import dataclass
from tqdm import tqdm


def dilate_mask(mask: torch.Tensor, ks: int = 25) -> torch.Tensor:
    padding = ks // 2
    dilated = F.max_pool2d(mask, kernel_size=ks, padding=padding, stride=1)
    return dilated


def erode_mask(mask: torch.Tensor, ks: int = 25) -> torch.Tensor:
    mask = 1 - mask
    padding = ks // 2
    eroded = F.max_pool2d(mask, kernel_size=ks, padding=padding, stride=1)
    return 1.0 - eroded


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


def neighbor_degree(x: torch.Tensor) -> torch.Tensor:
    ones = torch.ones_like(x)
    return neighbor_sum(ones)


@dataclass
class PoissonJacobiSolver:

    num_iters: int = 30
    lambda_c: float = 0.1
    momentum: float = 1.0

    def solve(
        self,
        g: torch.Tensor,
        x_S: torch.Tensor,
        M: torch.Tensor,
        soft_M: torch.Tensor | None = None,
    ):
        r"""
        Args
            g   (Tensor): The guidance tensor
            x_S (Tensor): The source tensor to keep background
            M   (Tensor): Mask
            soft_M (Optional Tensor): Soften mask for blending
        """
        B, C, H, W = g.shape
        M = M.float().expand(B, C, H, W)

        # Initialization
        y = M * g + (1.0 - M) * x_S
        D = neighbor_degree(g)
        nsum_g = neighbor_sum(g)
        div_g = D * g - nsum_g

        x_S_out = (1.0 - M) * x_S
        nsum_x_S = neighbor_sum(x_S_out)

        b = div_g + self.lambda_c * g + nsum_x_S

        diag = D + self.lambda_c

        # Solve linear
        for _ in tqdm(range(self.num_iters), desc="Solve Poisson"):
            y_in = M * y
            nsum_y_in = neighbor_sum(y_in)

            y_next = (nsum_y_in + b) / diag.clamp(min=1e-6)

            # Update masked area only
            y_next = M * y_next + (1.0 - M) * x_S

            y = self.momentum * y + (1.0 - self.momentum) * y_next

        if soft_M is not None:
            soft_M = soft_M.expand_as(g)
            y = soft_M * y + (1.0 - soft_M) * x_S
        return y


if __name__ == "__main__":
    import torchvision.transforms.functional as T
    from PIL import Image
    from torchvision.utils import save_image

    bg_path = "demo/source.png"
    cn_path = "demo/target.png"
    mask_path = "demo/mask.png"

    bg = T.to_tensor(Image.open(bg_path).convert("RGB")).unsqueeze(0)
    cn = T.to_tensor(Image.open(cn_path).convert("RGB")).unsqueeze(0)
    M = T.to_tensor(Image.open(mask_path).convert("L")).unsqueeze(0)

    cn = T.resize(cn, [*bg.shape[-2:]])
    M = T.resize(M, [*bg.shape[-2:]])
    soft_M = T.gaussian_blur(M, kernel_size=25, sigma=25)
    # B = dilate_mask(M, ks=25) - erode_mask(M, ks=25)

    solver = PoissonJacobiSolver(num_iters=20, lambda_c=0.1, momentum=0.9)
    blend = solver.solve(cn, bg, M, soft_M=soft_M)
    save_image(blend, "demo/blend.png")
