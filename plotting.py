import numpy as np
import matplotlib.pyplot as plt
import torch
from utils import *
from support import *

def plot_bunch_profiles(z, rho_true, rho_gs, rho_gd):
    z_np        = z.detach().cpu().numpy()
    rho_true_np = rho_true.detach().cpu().numpy()
    rho_gs_np   = rho_gs.detach().cpu().numpy()
    rho_gd_np   = rho_gd.detach().cpu().numpy()

    z_um = z_np * 1e6

    plt.figure(figsize=(7, 4))
    plt.plot(z_um, rho_true_np, color="C0", linestyle="-",  linewidth=2,
             label=r"$\rho_{\mathrm{true}}(z)$")
    plt.plot(z_um, rho_gs_np,   color="C1", linestyle="--", linewidth=2,
             label=r"$\rho_{\mathrm{GS}}(z)$")
    plt.plot(z_um, rho_gd_np,   color="C2", linestyle=":",  linewidth=2,
             label=r"$\rho_{\mathrm{GD}}(z)$")

    plt.xlabel(r"$z$ [$\mu\mathrm{m}$]")
    plt.ylabel(r"Normalized $\rho(z)$")
    # plt.title("CTR phase retrieval: bunch profiles")
    plt.legend(frameon=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_spectrum(I_meas, I_gd=None, I_gs=None,
                  z=None, sigma_z=None, title_suffix="", xlim_frac=0.01):
    """
    Plot normalized spectral intensity vs either k or k sigma_z.

    Parameters:
        I_meas : tensor [N]
        I_gd   : optional tensor [N]
        I_gs   : optional tensor [N]
        z      : tensor [N] (used to build k-axis)
        sigma_z: optional rms bunch length [m]. If given, x-axis is k*sigma_z.
    """
    I_meas_np = I_meas.detach().cpu().numpy()
    N = len(I_meas_np)

    # x-axis (use k if z provided, otherwise DFT index)
    if z is None:
        freq = np.fft.fftshift(np.fft.fftfreq(N))
        k = freq
        # xlabel = "Frequency index (DFT units)"
    else:
        z_np = z.detach().cpu().numpy()
        dz = z_np[1] - z_np[0]
        freq = np.fft.fftshift(np.fft.fftfreq(N, d=dz))
        k = 2 * np.pi * freq          # k [1/m]
        xlabel = r"$k$ [m$^{-1}$]"

    # shift spectra
    I_meas_shift = np.fft.fftshift(I_meas_np)
    I_meas_norm  = I_meas_shift / I_meas_shift.max()
    
    # choose x-axis representation
    if sigma_z is not None:
        x = k * sigma_z
        xlabel = r"$k \sigma_z$"
    else:
        x = k / 1e5
        xlabel = r"$k \,[10^{5}\,\mathrm{m}^{-1}]$"


    plt.figure(figsize=(7, 4))
    x_max = xlim_frac * np.max(np.abs(x))
    mask = (x >= -x_max) & (x <= x_max)

    plt.plot(x[mask], I_meas_norm[mask],
             color="C0", linestyle="-", linewidth=2,
             label=r"$|F_{\mathrm{true}}(k)|^{2}$")

    if I_gs is not None:
        I_gs_np    = I_gs.detach().cpu().numpy()
        I_gs_shift = np.fft.fftshift(I_gs_np)
        plt.plot(x[mask], (I_gs_shift / I_meas_shift.max())[mask],
                 color="C1", linestyle="--", linewidth=2,
                 label=r"$|F_{\mathrm{GS}}(k)|^{2}$")

    if I_gd is not None:
        I_gd_np    = I_gd.detach().cpu().numpy()
        I_gd_shift = np.fft.fftshift(I_gd_np)
        plt.plot(x[mask], (I_gd_shift / I_meas_shift.max())[mask],
                 color="C2", linestyle=":", linewidth=2,
                 label=r"$|F_{\mathrm{GD}}(k)|^{2}$")

    plt.xlabel(xlabel)
    # plt.ylabel(r"$\frac{|F(k)|^{2}}{\max_{k}|F(k)|^{2}}$")
    plt.ylabel(r"Normalized $|F(k)|^{2}$")
    # plt.title(f"CTR-like spectrum {title_suffix}")
    plt.legend(frameon=True)
    plt.grid(True, alpha=0.3)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    

def plot_compare_profiles_and_spectra(
    z: torch.Tensor,
    rho_true: torch.Tensor,
    rho_gs: torch.Tensor,
    rho_gd_gs: torch.Tensor,
    rho_gd_rand: torch.Tensor,
    I_meas: torch.Tensor,
    gamma: float,
):

    z_np = z.detach().cpu().numpy()
    z_um = z_np * 1e6

    # spectra (Schroeder forward for all three)
    _, omega_grid = make_k_and_omega_grids(z, gamma)
    I_true, _ = forward_spectrum_schroeder_1d(z, rho_true, omega_grid, gamma, normalize=True)
    I_gs, _   = forward_spectrum_schroeder_1d(z, rho_gs, omega_grid, gamma, normalize=True)
    I_gd1, _  = forward_spectrum_schroeder_1d(z, rho_gd_gs, omega_grid, gamma, normalize=True)
    I_gd2, _  = forward_spectrum_schroeder_1d(z, rho_gd_rand, omega_grid, gamma, normalize=True)

    I_true_np = I_true.detach().cpu().numpy()
    I_gs_np   = I_gs.detach().cpu().numpy()
    I_gd1_np  = I_gd1.detach().cpu().numpy()
    I_gd2_np  = I_gd2.detach().cpu().numpy()

    # k*sigma_z axis
    dz = z_np[1] - z_np[0]
    freq = np.fft.fftfreq(z_np.size, d=dz)
    k = 2 * np.pi * freq
    sigma_z = float(compute_sigma_z(z, rho_true))
    x = np.fft.fftshift(k * sigma_z)

    I_true_shift = np.fft.fftshift(I_true_np)
    I_gs_shift   = np.fft.fftshift(I_gs_np)
    I_gd1_shift  = np.fft.fftshift(I_gd1_np)
    I_gd2_shift  = np.fft.fftshift(I_gd2_np)

    fig, axes = plt.subplots(2, 1, figsize=(6, 7), sharex=False)

    ax = axes[0]
    ax.plot(z_um, rho_true.detach().cpu().numpy(), "C0-",  lw=2, label=r"$\rho_{\mathrm{true}}(z)$")
    ax.plot(z_um, rho_gs.detach().cpu().numpy(),   "C1--", lw=2, label=r"$\rho_{\mathrm{GS}}(z)$")
    ax.plot(z_um, rho_gd_gs.detach().cpu().numpy(),"C2-.", lw=2, label=r"$\rho_{\mathrm{GD,GS}}(z)$")
    ax.plot(z_um, rho_gd_rand.detach().cpu().numpy(),"C3:", lw=2, label=r"$\rho_{\mathrm{GD,rand}}(z)$")
    ax.set_ylabel("Normalized $\\rho(z)$")
    ax.set_xlabel("$z$ [$\\mu$m]")
    ax.set_xlim([-100,100])
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.3)

    
    ax = axes[1]
    ax.plot(x, I_true_shift, "C0-",  lw=2, label=r"$|F_{\mathrm{true}}(k)|^{2}$")
    ax.plot(x, I_gs_shift,   "C1--", lw=2, label=r"$|F_{\mathrm{GS}}(k)|^{2}$")
    ax.plot(x, I_gd1_shift,  "C2-.", lw=2, label=r"$|F_{\mathrm{GD,GS}}(k)|^{2}$")
    ax.plot(x, I_gd2_shift,  "C3:",  lw=2, label=r"$|F_{\mathrm{GD,rand}}(k)|^{2}$")
    ax.set_xlabel("$k \\sigma_{z}$")
    ax.set_ylabel("Normalized $|F(k)|^{2}$")
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-10,10])
    

    fig.tight_layout()
    plt.show()


def plot_loss_vs_iterations(losses_gs_init, losses_rand_init,
                            label_gs="GD from GS", label_rand="GD from random",
                            logy=True):
    L1 = np.array([float(x) for x in losses_gs_init])
    L2 = np.array([float(x) for x in losses_rand_init])
    it1 = np.arange(len(L1))
    it2 = np.arange(len(L2))

    plt.figure(figsize=(6,4))
    plt.plot(it1, L1, label=label_gs, linewidth=2)
    plt.plot(it2, L2, label=label_rand, linewidth=2, linestyle="--")

    plt.xlabel("iterations")
    plt.ylabel("loss")

    if logy:
        plt.yscale("log")

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    
# for GS clusgter (“Spaghetti + best” plot in z–space like O. Zarini IEEE's Fig. 4)
def plot_gs_cluster_profiles(z, rho_true, rho_stack, loss_arr, K=10):
    z_np = z.detach().cpu().numpy()
    rho_true_np = rho_true.detach().cpu().numpy()
    R, N = rho_stack.shape

    # sort candidates by spectral loss
    idx = np.argsort(loss_arr)
    rho_sorted = rho_stack[idx].numpy()
    loss_sorted = loss_arr[idx]

    plt.figure(figsize=(6,4))
    # all candidates in light grey
    for r in range(R):
        plt.plot(z_np*1e6, rho_sorted[r], color="0.8", linewidth=0.5)

    # best K in slightly darker grey
    for r in range(min(K, R)):
        plt.plot(z_np*1e6, rho_sorted[r], color="0.5", linewidth=0.8)

    # best candidate (smallest loss)
    plt.plot(z_np*1e6, rho_sorted[0], color="C1", linewidth=2,
             label=r"best GS candidate")

    # true bunch (only in synthetic tests)
    plt.plot(z_np*1e6, rho_true_np, color="C0", linewidth=2,
             label=r"$\rho_\mathrm{true}(z)$")

    plt.xlabel(r"$z\,[\mu\mathrm{m}]$")
    plt.ylabel(r"Normalized $\rho(z)$")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
  
    
# Histogram / scatter of spectral loss (show cluster)
def plot_gs_loss_histogram(loss_arr):
    plt.figure(figsize=(4,3))
    plt.hist(loss_arr, bins=15, color="C0", alpha=0.7)
    plt.xlabel("Spectral log-MSE")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    
# synthetic data: scatter correlation vs loss
def plot_loss_vs_corr(z, rho_true, rho_stack, loss_arr):
    rho_true_np = rho_true.detach().cpu().numpy()
    def corr(a,b):
        a = a - a.mean()
        b = b - b.mean()
        return np.dot(a,b) / np.sqrt(np.dot(a,a)*np.dot(b,b))

    corrs = []
    for r in range(rho_stack.shape[0]):
        corrs.append(corr(rho_true_np, rho_stack[r].numpy()))

    plt.figure(figsize=(4,3))
    plt.scatter(loss_arr, corrs, s=20)
    plt.xlabel("Spectral log-MSE")
    plt.ylabel(r"$\mathrm{corr}(\rho_\mathrm{true}, \rho)$")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
