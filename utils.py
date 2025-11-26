import torch
import torch.nn as nn
import torch.fft as fft
import numpy as np
import time
from propagator import *
from support import *
from ctr_pr_gs import gerchberg_saxton_multistart

c_light = 299_792_458.0  # m/s

def make_z_grid(N=1024, Lz=100e-6, device="cuda"):
    """
    Uniform longitudinal grid from -Lz/2 to Lz/2 (meters).
    """
    return torch.linspace(-Lz/2, Lz/2, N, device=device)


def make_true_bunch(z: torch.Tensor) -> torch.Tensor:
    """
    Example 'true' bunch profile rho(z): sum of two Gaussians in z.
    rho is returned normalized so integral of rho(z) dz = 1 (shape only).
    """
    device = z.device
    # centers (meters), sigmas (meters), relative weights
    centers = torch.tensor([-15e-6, 20e-6], device=device)
    sigmas  = torch.tensor([6e-6,  8e-6], device=device)
    weights = torch.tensor([0.7, 0.3], device=device)

    rho = torch.zeros_like(z)
    for z0, s, w in zip(centers, sigmas, weights):
        rho = rho + w * torch.exp(-0.5 * ((z - z0)/s)**2)

    # normalize to unit area (Q = 1 arbitrary units)
    # like a probability density which makes numerical behavior nicer
    # GS and GD don’t have to chase a trivial scaling degree of freedom;
    # we just reconstruct the shape
    rho = rho / rho.sum()
    return rho

def make_k_and_omega_grids(z: torch.Tensor, gamma: float):
    """
    Build k- and omega-grids consistent with the FFT indexing.

    Returns
    -------
    k_grid     : [N] numpy array, spatial frequencies [1/m]
    omega_grid : [N] torch tensor, angular frequencies [rad/s]
    """
    z_np = z.detach().cpu().numpy()
    dz = z_np[1] - z_np[0]
    N = z_np.size

    freq = np.fft.fftfreq(N, d=dz)        # [1/m]
    k_grid = 2 * np.pi * freq             # [1/m]

    beta = np.sqrt(1.0 - 1.0 / (gamma**2))
    omega_grid_np = beta * c_light * k_grid   # [rad/s]

    omega_grid = torch.from_numpy(omega_grid_np).to(z.device, dtype=torch.float64)
    return k_grid, omega_grid



def corr_coef(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().cpu()
    b = b.detach().cpu()
    a = a - a.mean()
    b = b - b.mean()
    num = (a * b).sum()
    den = torch.sqrt((a * a).sum() * (b * b).sum()) + 1e-12
    return float(num / den)

def spectral_log_mse(I_pred: torch.Tensor, I_meas: torch.Tensor) -> float:
    eps = 1e-12
    I_pred = I_pred.detach().cpu()
    I_meas = I_meas.detach().cpu()
    loss = ((torch.log(I_pred + eps) - torch.log(I_meas + eps))**2).mean()
    return float(loss)


def compute_sigma_z(z: torch.Tensor, rho: torch.Tensor) -> float:
    """
    Compute rms bunch length sigma_z for a normalized rho(z).
    Returns sigma_z as a Python float (meters).
    """
    w = rho / rho.sum()
    z_mean = (w * z).sum()
    var = (w * (z - z_mean)**2).sum()
    return float(torch.sqrt(var))

def compute_mean_and_sigma_z(z: torch.Tensor, rho: torch.Tensor):
    """
    Compute center-of-mass <z> and rms bunch length sigma_z
    for a normalized rho (sum rho * dz = 1).
    """
    dz = z[1] - z[0]
    # ensure normalization (shape-only prior)
    w = rho / (rho.sum() * dz + 1e-20)

    z_mean = (w * z).sum()
    var = (w * (z - z_mean)**2).sum()
    sigma_z = torch.sqrt(var + 1e-20)
    return z_mean, sigma_z


def evaluate_method(
    label: str,
    rho_true: torch.Tensor,
    rho_rec: torch.Tensor,
    I_meas: torch.Tensor,
    z: torch.Tensor,
    gamma: float,
    forward: str = "fft",  # or "schroeder"
) -> dict:
    """
    Return metrics for a single reconstruction.
    """
    # correlation in real space
    rho_corr = corr_coef(rho_true, rho_rec)

    # spectrum from reconstructed rho
    if forward == "fft":
        I_pred, _ = forward_spectrum_fft(rho_rec)
    elif forward == "schroeder":
        _, omega_grid = make_k_and_omega_grids(z, gamma)
        I_pred, _ = forward_spectrum_schroeder_1d(
            z=z, rho=rho_rec, omega=omega_grid, gamma=gamma, normalize=True
        )
    else:
        raise ValueError(f"Unknown forward '{forward}'")

    spec_loss = spectral_log_mse(I_pred, I_meas)

    return {
        "label": label,
        "rho_corr": rho_corr,
        "spec_loss": spec_loss,
    }


def benchmark_gs_backends(
    N: int = 1024,
    Lz: float = 100e-6,
    half_width: float = 50e-6,
    n_iters: int = 800,
    n_restarts: int = 50,
    gamma: float = 100.0,
    n_trials: int = 1,         # we can also average over multiple runs
):
    """
    Benchmark GS in different backends and print a LaTeX-ready table.

    Backends:
        - NumPy GS (CPU only)
        - Torch GS on CPU
        - Torch GS on GPU (if available)
    """
    # build synthetic test problem (on cpu)
    device_cpu = torch.device("cpu")

    z = make_z_grid(N=N, Lz=Lz, device=device_cpu)
    rho_true = make_true_bunch(z)

    # Schroeder forward for measured spectrum
    _, omega_grid = make_k_and_omega_grids(z, gamma)
    I_meas, _ = forward_spectrum_schroeder_1d(
        z=z, rho=rho_true, omega=omega_grid, gamma=gamma, normalize=True
    )

    # support mask (swap this for Tukey/Hann for testing)
    support_mask = make_support_mask(z, half_width=half_width)

    # configs to test
    configs = [
        {"backend": "numpy", "device": "cpu"},
        {"backend": "torch", "device": "cpu"},
    ]

    if torch.cuda.is_available():
        configs.append({"backend": "torch", "device": "cuda"})

    results = []

    # run benchmarks
    for cfg in configs:
        backend = cfg["backend"]
        dev_str = cfg["device"]
        dev = torch.device(dev_str)

        times = []
        rho_corrs = []
        spec_losses = []

        for _ in range(n_trials):
            torch.cuda.empty_cache() if dev.type == "cuda" else None

            t0 = time.perf_counter()
            rho_gs = gerchberg_saxton_multistart(
                I_meas=I_meas,
                n_iters=n_iters,
                support_mask=support_mask,
                n_restarts=n_restarts,
                backend=backend,
                device=dev_str,
            )
            t1 = time.perf_counter()
            times.append(t1 - t0)

            # evaluate using Schroeder forward for consistency
            I_pred, _ = forward_spectrum_schroeder_1d(
                z=z.to(dev_cpu := torch.device("cpu")),  # forward expects same z
                rho=rho_gs.to(dev_cpu),
                omega=omega_grid,
                gamma=gamma,
                normalize=True,
            )

            rho_corrs.append(corr_coef(rho_true, rho_gs))
            spec_losses.append(spectral_log_mse(I_pred, I_meas))

        results.append({
            "backend": backend,
            "device": dev_str,
            "n_restarts": n_restarts,
            "time_mean": np.mean(times),
            "time_std": np.std(times),
            "rho_corr_mean": np.mean(rho_corrs),
            "rho_corr_std": np.std(rho_corrs),
            "spec_loss_mean": np.mean(spec_losses),
            "spec_loss_std": np.std(spec_losses),
        })

    # print table
    print("\n% GS timing comparison")
    print("\\begin{tabular}{llrrrr}")
    print("\\hline")
    print("Backend & Device & Restarts & Time [s] & "
          "$\\mathrm{corr}(\\rho_\\mathrm{true},\\rho)$ & "
          "Spectral log-MSE\\\\")
    print("\\hline")

    for r in results:
        print(
            f"{r['backend']:s} & "
            f"{r['device']:s} & "
            f"{r['n_restarts']:3d} & "
            f"{r['time_mean']:7.3f} & "
            f"{r['rho_corr_mean']:6.3f} & "
            f"{r['spec_loss_mean']:8.2e}\\\\"
        )

    print("\\hline")
    print("\\end{tabular}")

    return results



    
