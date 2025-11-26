import torch
import torch.nn as nn
import torch.fft as fft
import matplotlib.pyplot as plt
import numpy as np

c_light = 299_792_458.0  # m/s

def forward_spectrum_fft(rho_z: torch.Tensor) -> torch.Tuple[torch.Tensor, torch.Tensor]:
    """
    Toy forward model:
        bunch rho(z) -> spectrum I(k) prop.to. |FFT[rho(z)]|^2

    Returns:
        I_k: [N] intensity spectrum
        F_k: [N] field in frequency domain (complex FFT of rho)
    """
    # DFT. We don't worry about physical scaling here (only shape matters).
    F_k = fft.fft(rho_z)
    I_k = (F_k.abs() ** 2)
    return I_k, F_k


def forward_spectrum_schroeder_1d(
    z: torch.Tensor,
    rho: torch.Tensor,
    omega: torch.Tensor,
    gamma: float,
    normalize: bool = True,
):
    """
    Longitudinal CTR spectrum in the Schroeder form-factor picture.

    Parameters
    ----------
    z      : [N] tensor, longitudinal positions [m]
    rho    : [N] tensor, bunch density (shape only; will be renormalized)
    omega  : [M] tensor, angular frequencies [rad/s] at which to evaluate I(omega)
    gamma  : scalar Lorentz factor
    normalize : if True, divide spectrum by its maximum (dimensionless output)

    Returns
    -------
    I_omega : [M] tensor, spectrum prop.to. |F_parallel(omega)|^2
    F_omega : [M] complex tensor, longitudinal form factor F_parallel(omega)
    """
    device = z.device
    dtype = torch.complex64 if rho.dtype in (torch.float32, torch.complex64) else torch.complex128

    z = z.to(device)
    rho = rho.to(device)
    omega = omega.to(device)

    # normalize rho so integral of rho dz = 1 (shape only)
    dz = z[1] - z[0]
    rho = rho / (rho.sum() * dz)

    beta = np.sqrt(1.0 - 1.0 / (gamma**2))

    # phase factor: shape [M, N]
    phase = torch.exp(-1j * omega[:, None] * z[None, :] / (beta * c_light)).to(dtype)

    # discrete integral over z
    F_omega = torch.sum(rho[None, :] * phase, dim=1) * dz

    I_omega = (F_omega.real**2 + F_omega.imag**2)

    if normalize:
        I_omega = I_omega / (I_omega.max() + 1e-20)

    return I_omega, F_omega
