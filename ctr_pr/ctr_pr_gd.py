import torch
import torch.nn as nn
import torch.fft as fft
from utils import *
from support import *
from propagator import *


class BunchModel(nn.Module):
    def __init__(self, N: int, 
                 support_mask: torch.Tensor = None,
                 init_profile: torch.Tensor = None, 
                 device: str = "cpu"):        
        """
        Trainable bunch rho(z):
          - positivity via exp(log_rho)
          - optional support mask in z
          - normalization: sum_j rho_j = 1
          - can be initialized from a given profile (e.g. GS result)
        """
        super().__init__()
        device = torch.device(device)
        
        if init_profile is None:
            # random positive shape if nothing is provided
            rho0 = torch.rand(N, device=device)
            rho0 = rho0 / rho0.sum()
        else:
            rho0 = init_profile.to(device).clamp(min=1e-12)
            rho0 = rho0 / rho0.sum()
            
        # log-parameterization: rho = exp(log_rho)
        self.log_rho = nn.Parameter(torch.log(rho0))

        
        if support_mask is None:
            support_mask = torch.ones(N, device=device)
        self.register_buffer("support_mask", support_mask.to(device))
    
    def forward(self) -> torch.Tensor:
        # enforce positivity via exp, then support + normalization
        rho = torch.exp(self.log_rho) * self.support_mask
        s = rho.sum()
        if s > 0:
            rho = rho / s
        return rho

    # def forward(self) -> torch.Tensor:
    #     # enforce positivity + normalization
    #     rho = torch.nn.functional.softplus(self.raw)
    #     rho = rho * self.support_mask
    #     s = rho.sum()
    #     if s > 0:
    #         rho = rho / s
    #     return rho
    
    
def gradient_descent(
    I_meas: torch.Tensor,
    z: torch.Tensor,
    support_mask: torch.Tensor = None,
    init_profile: torch.Tensor = None,
    n_steps: int = 4000,
    lr: float = 5e-3,
    gamma: float = 100.0,
    method: str = "fft", # or schroeder
    lambda_smooth: float = 1e-2,
    # physicall motivated priors (if applicable)
    z0_target: float | None = None,        # target center-of-mass [m]
    sigma_z_target: float | None = None,   # target rms length [m]
    lambda_com: float = 0.0,                # weight for center-of-mass prior
    lambda_sigma: float = 0.0,             # weight for sigma_z prior
    device: str = "cpu",
) -> torch.Tensor:
    """
    Fully PyTorch phase retrieval:
        - rho(z) is a trainable parameter (see BunchModel)
        - forward model is FFT-based CTR spectrum
        - loss compares log-intensities (for dynamic range robustness)
        - data term: log-intensity mismatch
        - regularization: lambda_smooth * ||parital²rho/partial z^2||^2
        - optional initialization from a given profile (e.g. GS result or smooth initial support)
        - optional physics regularization terms:
            loss = data_loss
               + lambda_smooth * ||partial²rho/partial z^2||^2
               + lambda_com    * (<z> - z0_target)^2
               + lambda_sigma  * (sigma_z - sigma_z_target)^2
    
    Params:
        I_meas       : measured intensity spectrum, shape [N]
        support_mask : optional support in z
    """
    device = torch.device(device)
    I_meas = I_meas.to(device)
    z = z.to(device)
    N = I_meas.numel()
    
    # normalize measured spectrum (if not done already)
    I_meas = I_meas / (I_meas.max() + 1e-12)
    
    if method == "schroeder":
        _, omega_grid = make_k_and_omega_grids(z, gamma)
        omega_grid = omega_grid.to(device)

    model = BunchModel(
        N,
        support_mask=support_mask,
        init_profile=init_profile,
        device=device,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    # for Bi: add lr scheduler here (some decay maybe)???
    eps = 1e-12

    if method == "schroeder":
        k_grid, omega_grid = make_k_and_omega_grids(z, gamma)
        # k_grid = k_grid.to(device)
        # omega_grid = omega_grid.to(device)
    
    losses = []
    for step in range(n_steps):
        opt.zero_grad()

        rho = model()
        
        if method == "fft":
            I_pred, _ = forward_spectrum_fft(rho)
        elif method == "schroeder":
            I_pred, F_omega_true = forward_spectrum_schroeder_1d(
                z=z,
                rho=rho,
                omega=omega_grid,
                gamma=gamma,
                normalize=True,   # same behavior as before
            )            
        
        # We can further exploit this! What is our data loss function?
        
        # # regular MSE
        # data_loss = torch.mean(
        #     ((I_pred) - (I_meas))**2)
        
        data_loss = torch.mean(((I_pred) - (I_meas))**2 / (I_meas + eps)**2)
        
        # log-intensity loss to reduce emphasis on bright peaks
        # more stable for large dynamic range
        # data_loss = torch.mean(
        #     (torch.log(I_pred + eps) - torch.log(I_meas + eps))**2
        # )

        lap = second_derivative(rho)
        smooth_loss = (lap**2).mean()
        
        # NEW: center-of-mass and sigma_z priors
        com_loss = rho.new_tensor(0.0)
        sigma_loss = rho.new_tensor(0.0)
        
        if (z0_target is not None) or (sigma_z_target is not None):
            z_mean, sigma_z = compute_mean_and_sigma_z(z, rho)

            if z0_target is not None and lambda_com > 0.0:
                com_loss = (z_mean - z0_target)**2

            if sigma_z_target is not None and lambda_sigma > 0.0:
                sigma_loss = (sigma_z - sigma_z_target)**2

        # loss = data_loss + lambda_smooth*smooth_loss
                
        loss = (
            data_loss
            + lambda_smooth * smooth_loss
            # Max says we can't get this from experiments
            + lambda_com * com_loss
            + lambda_sigma * sigma_loss
        )
        

        loss.backward()
        opt.step()

        losses.append(float(loss.detach()))

        if step % 500 == 0 or step == n_steps - 1:
            # print(f"[GD] step {step:4d}, "
            #       f"data={data_loss.item():.3e}, "
            #       f"smooth={smooth_loss.item():.3e}, "
            #       f"total={loss.item():.3e}")

            print(
                f"[GD] step {step:4d}, "
                f"data={data_loss.item():.3e}, "
                f"smooth={smooth_loss.item():.3e}, "
                f"com={com_loss.item():.3e}, "
                f"sigma={sigma_loss.item():.3e}, "
                f"total={loss.item():.3e}"
            )

    # final rho(z)
    rho_final = model().detach()
    return rho_final, losses


def second_derivative(rho: torch.Tensor) -> torch.Tensor:
    """
    Discrete 2nd derivative (Laplacian) with unit grid spacing.
    We don't need physical dz here; we're just enforcing smoothness.
    """
    return rho[2:] - 2 * rho[1:-1] + rho[:-2]



def make_smooth_init(z: torch.Tensor, half_width: float) -> torch.Tensor:
    w = make_soft_support(z, half_width)
    # avoid zero everywhere
    w = w + 1e-6
    dz = z[1] - z[0]
    w = w / (w.sum() * dz)  # normalize summation of rho dz = 1
    return w