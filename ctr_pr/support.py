import torch
import torch.nn.functional as F
import numpy as np

def make_support_mask(z: torch.Tensor, half_width: float) -> torch.Tensor:
    """
    Simple support: rho(z) = 0 outside |z| <= half_width.
    """
    return (z.abs() <= half_width).float()

def make_soft_support(z: torch.Tensor, half_width: float) -> torch.Tensor:
    """
    Smooth (raised-cosine) support window:
        w(z) = 0.5 * (1 + cos(pi * z / half_width))  for |z| <= half_width
             = 0                                     otherwise
    So w = 1 at z=0, smoothly to 0 at |z| = half_width.
    """
    w = torch.zeros_like(z)
    inside = torch.abs(z) <= half_width
    w[inside] = 0.5 * (1.0 + torch.cos(np.pi * z[inside] / half_width))
    return w

def make_smooth_init(z: torch.Tensor, half_width: float) -> torch.Tensor:
    w = make_soft_support(z, half_width)
    # avoid zero everywhere
    w = w + 1e-6
    dz = z[1] - z[0]
    w = w / (w.sum() * dz)  # normalize integral of rho dz = 1
    return w


def make_hann_support(z: torch.Tensor, half_width: float) -> torch.Tensor:
    """
    Hann-like support over |z| <= half_width.

    w(z) = 0.5 * (1 - cos(2π * t)),  t in [0,1],
    mapped from z in [-half_width, +half_width].

    Outside |z| > half_width, w = 0.
    """
    w = torch.zeros_like(z)
    inside = torch.abs(z) <= half_width

    # map z from [-half_width, half_width] to t in [0,1]
    t = (z[inside] + half_width) / (2.0 * half_width)
    w[inside] = 0.5 * (1.0 - torch.cos(2.0 * np.pi * t))
    return w


def make_tukey_support(z: torch.Tensor, half_width: float, alpha: float = 0.5) -> torch.Tensor:
    """
    Tukey window over |z| <= half_width.

    alpha in [0,1]:
        0   -> rectangular (hard edges)
        1   -> Hann (fully tapered)
        mid -> flat center with cosine tapers at edges
    """
    alpha = float(alpha)
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError("alpha must be in [0,1]")

    w = torch.zeros_like(z)
    inside = torch.abs(z) <= half_width
    if not inside.any():
        return w

    # normalized coordinate in [0,1] over the support
    x = (z[inside] + half_width) / (2.0 * half_width)

    if alpha == 0.0:
        w[inside] = 1.0
        return w
    if alpha == 1.0:
        # falls back to Hann over entire support
        w[inside] = 0.5 * (1.0 - torch.cos(2.0 * np.pi * x))
        return w

    # 0 <= x < alpha/2 : rising cosine
    # alpha/2 <= x <= 1 - alpha/2 : flat (1)
    # 1 - alpha/2 < x <= 1 : falling cosine
    w_local = torch.zeros_like(x)

    # rising edge
    idx1 = x < alpha / 2.0
    w_local[idx1] = 0.5 * (1.0 + torch.cos(
        np.pi * (2.0 * x[idx1] / alpha - 1.0)
    ))

    # flat region
    idx2 = (x >= alpha / 2.0) & (x <= 1.0 - alpha / 2.0)
    w_local[idx2] = 1.0

    # falling edge
    idx3 = x > 1.0 - alpha / 2.0
    w_local[idx3] = 0.5 * (1.0 + torch.cos(
        np.pi * (2.0 * x[idx3] / alpha - 2.0 / alpha + 1.0)
    ))

    w[inside] = w_local
    return w



def make_smooth_init_from_support(z, support_mask):
    dz = z[1] - z[0]
    rho0 = support_mask.clone()
    rho0 = rho0 + 1e-6          # avoid all-zero
    rho0 = rho0 / (rho0.sum() * dz)
    return rho0

