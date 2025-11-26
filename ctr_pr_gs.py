import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
from utils import *
from support import *
from propagator import *
import numpy as np


## start: torch
def _smooth_1d(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    Convolve 1D tensor x with a 1D kernel (both on same device).
    Reflect padding to avoid edge artifacts.
    """
    x = x.view(1, 1, -1)  # [B,C,L]
    k = kernel.view(1, 1, -1)
    pad = (k.shape[-1] - 1) // 2
    x_pad = F.pad(x, (pad, pad), mode="reflect")
    y = F.conv1d(x_pad, k)
    return y.view(-1)

def get_smoothing_kernel(device="cpu"):
    k = torch.tensor([1., 4., 6., 4., 1.], device=device)
    k = k / k.sum()
    return k

    
def gerchberg_saxton_1d_torch(
    I_meas: torch.Tensor,
    n_iters: int = 500,
    support_mask: torch.Tensor = None,
    smooth: bool = True,
    device: str = "cpu",
) -> torch.Tensor:
    """
    1D Gerchberg–Saxton phase retrieval. 
    Optionally applies a mild smoothing after each iteration.

    I_meas: measured spectrum (|FFT[rho_true]|^2), shape [N]
    support_mask: optional 0/1 mask in z-domain, shape [N]
    
    Returns
    --------
    rho_recon(z)
    """
    device = torch.device(device)
    I_meas = I_meas.to(device)
    N = I_meas.numel()

    # magnitude in frequency domain
    mag_meas = torch.sqrt(torch.clamp(I_meas, min=0.0))

    # random positive real initial guess in z
    rho = torch.rand(N, device=device)
    rho = rho / rho.sum()
    
    kernel = get_smoothing_kernel(device) if smooth else None

    for _ in range(n_iters):
        # forward: z -> k
        F = fft.fft(rho)

        # impose measured magnitude, keep current phase
        phase = torch.exp(1j * torch.angle(F))
        F_new = mag_meas * phase

        # inverse: k -> z
        rho_new = fft.ifft(F_new).real

        # real-space constraints: positivity + optional support
        rho_new = torch.clamp(rho_new, min=0.0)
        if support_mask is not None:
            rho_new = rho_new * support_mask.to(device)
        
        # smoothing (optional)
        if kernel is not None:
            rho_new = _smooth_1d(rho_new, kernel)

        # renormalize total charge
        s = rho_new.sum()
        if s > 0:
            rho_new = rho_new / s

        rho = rho_new

    return rho


# def gerchberg_saxton_multistart(
#     I_meas: torch.Tensor,
#     n_iters: int,
#     support_mask: torch.Tensor = None,
#     n_restarts: int = 8,
#     # gamma: int = 100,
#     # method: str = "fft", # or schroeder
#     device: str = "cpu",
# ) -> torch.Tensor:
#     """
#     Run GS multiple times with different random initializations and
#     return the reconstruction whose spectrum best matches I_meas.
#     """
#     device = torch.device(device)
#     best_rho = None
#     best_loss = float("inf")

#     for r in range(n_restarts):
#         # different seed per restart for reproducibility
#         torch.manual_seed(r)
#         rho = gerchberg_saxton_1d(
#             I_meas=I_meas,
#             n_iters=n_iters,
#             support_mask=support_mask,
#             smooth=True,
#             device=device,
#         )

#         I_rec, _ = forward_spectrum_fft(rho)
#         # if method == "fft":
#         #     I_rec, _ = forward_spectrum_fft(rho)
#         # elif method == "schroeder":
#         #     k_grid, omega_grid = make_k_and_omega_grids(z, gamma)
#         #     I_rec, F_omega_true = forward_spectrum_schroeder_1d(
#         #         z=z,
#         #         rho=rho,
#         #         omega=omega_grid,
#         #         gamma=gamma,
#         #         normalize=True,   # same behavior as before
#         #     )
            
#         # log-intensity loss, same as GD
#         loss = torch.mean(
#             (torch.log(I_rec + 1e-12) - torch.log(I_meas.to(device) + 1e-12))**2
#         )
#         if loss.item() < best_loss:
#             best_loss = float(loss.item())
#             best_rho = rho.detach()

#     print(f"[GS] best spectral loss over {n_restarts} restarts: {best_loss:.3e}")
#     return best_rho
##-- end: torch


##-- start: numpy
def get_smoothing_kernel_numpy():
    k = np.array([1., 4., 6., 4., 1.], dtype=np.float64)
    k = k / k.sum()
    return k


def _smooth_1d_numpy(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    1D convolution with reflect padding, analogous to _smooth_1d (Torch).
    """
    pad = (kernel.size - 1) // 2
    x_pad = np.pad(x, pad_width=pad, mode="reflect")
    y = np.convolve(x_pad, kernel, mode="valid")
    return y

def gerchberg_saxton_1d_numpy(
    I_meas_np: np.ndarray,
    n_iters: int = 500,
    support_mask_np: np.ndarray | None = None,
    smooth: bool = True,
    seed: int | None = None,
) -> np.ndarray:
    """
    1D Gerchberg–Saxton phase retrieval (NumPy backend).
    Same logic as gerchberg_saxton_1d_torch but in NumPy.
    """
    I_meas_np = np.asarray(I_meas_np, dtype=np.float64)
    N = I_meas_np.size
    mag_meas = np.sqrt(np.clip(I_meas_np, 0.0, None))

    if support_mask_np is not None:
        support_mask_np = np.asarray(support_mask_np, dtype=np.float64)

    rng = np.random.default_rng(seed)

    # random positive real initial guess
    rho = rng.random(N)
    rho /= rho.sum()

    kernel = get_smoothing_kernel_numpy() if smooth else None

    for _ in range(n_iters):
        # forward: z -> k
        Fk = np.fft.fft(rho)

        # impose measured magnitude, keep current phase
        phase = np.exp(1j * np.angle(Fk))
        F_new = mag_meas * phase

        # inverse: k -> z
        rho_new = np.fft.ifft(F_new).real

        # constraints in real space
        rho_new = np.maximum(rho_new, 0.0)
        if support_mask_np is not None:
            rho_new *= support_mask_np

        if kernel is not None and kernel.size > 1:
            rho_new = _smooth_1d_numpy(rho_new, kernel)

        s = rho_new.sum()
        if s > 0:
            rho_new /= s

        rho = rho_new

    return rho
##--- end: numpy


## choose backend
def gerchberg_saxton_multistart(
    I_meas: torch.Tensor,
    n_iters: int,
    support_mask: torch.Tensor = None,
    n_restarts: int = 8,
    backend: str = "torch",   # "torch" or "numpy"
    device: str = "cpu",
    seed = None,
) -> torch.Tensor:
    """
    Run GS multiple times with different random initializations and
    return the reconstruction whose spectrum best matches I_meas.

    backend:
        - "torch"  : use gerchberg_saxton_1d_torch
        - "numpy"  : use gerchberg_saxton_1d_numpy
                     (converted back to torch at the end)
    """
    backend = backend.lower()
    device = torch.device(device)
    
    if seed is not None:
        torch.manual_seed(seed)

    if backend == "torch":
         # torch backend
        best_rho = None
        best_loss = float("inf")
        dev = torch.device(device)

        for r in range(n_restarts):
            # different seed per restart for reproducibility
            # torch.manual_seed(r)
            rho = gerchberg_saxton_1d_torch(
                I_meas=I_meas,
                n_iters=n_iters,
                support_mask=support_mask,
                smooth=True,
                device=device,
            )

            I_rec, _ = forward_spectrum_fft(rho)
            loss = torch.mean(
                (torch.log(I_rec + 1e-12) - torch.log(I_meas.to(device) + 1e-12))**2
            )
            if loss.item() < best_loss:
                best_loss = float(loss.item())
                best_rho = rho.detach()

        print(f"[GS-Torch-{dev.type}] best spectral loss over {n_restarts} restarts: {best_loss:.3e}")
        return best_rho.to(device)

    elif backend == "numpy":
        # numpy backend
        I_meas_np = I_meas.detach().cpu().numpy()
        support_np = None if support_mask is None else support_mask.detach().cpu().numpy()

        best_rho_np = None
        best_loss = np.inf

        for r in range(n_restarts):
            rho_np = gerchberg_saxton_1d_numpy(
                I_meas_np=I_meas_np,
                n_iters=n_iters,
                support_mask_np=support_np,
                smooth=True,
                seed=r,
            )

            # compute spectrum and log-MSE in numpy
            Fk = np.fft.fft(rho_np)
            I_rec_np = np.abs(Fk)**2
            loss = np.mean(
                (np.log(I_rec_np + 1e-12) - np.log(I_meas_np + 1e-12))**2
            )

            if loss < best_loss:
                best_loss = float(loss)
                best_rho_np = rho_np.copy()

        print(f"[GS-NumPy] best spectral loss over {n_restarts} restarts: {best_loss:.3e}")

        # convert back to torch for downstream code
        best_rho_torch = torch.from_numpy(best_rho_np).to(device, dtype=I_meas.dtype)
        return best_rho_torch

    else:
        raise ValueError("backend must be 'torch' or 'numpy'")
