import torch
from typing import Tuple
from pykeops.torch import LazyTensor as KLT, Vi, Vj
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.xpu import device
# ===============================
# Lattice helpers (KeOps-friendly)
# ===============================
def differentiable_round(x: torch.Tensor) -> torch.Tensor:
    return x + (torch.round(x) - x).detach()

def _to_frac(x: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
    x   = torch.as_tensor(x)
    cel = torch.as_tensor(cell, dtype=x.dtype, device=x.device)
    fT  = torch.linalg.solve(cel.transpose(-2, -1), x.transpose(-2, -1))
    return fT.transpose(-2, -1)

def _from_frac(f: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
    return torch.matmul(f, cell)

def wrap_positions(cart: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
    f = _to_frac(cart, cell)
    return _from_frac(f - torch.floor(f), cell)

# ==========================
# PW grid + residual (unchanged)
# ==========================

class PWGrid(nn.Module):
    def __init__(self, grid_shape, device="cpu", dtype=torch.float32):
        super().__init__()
        self.nx, self.ny, self.nz = grid_shape
        self.nzr = self.nz // 2 + 1

        self.register_buffer("cell", torch.eye(3, dtype=dtype, device=device))
        self.register_buffer("B", torch.eye(3, dtype=dtype, device=device))

        # ensure correct dtype/device early
        hx = torch.fft.fftfreq(self.nx, d=1.0).to(dtype=dtype, device=device) * self.nx
        hy = torch.fft.fftfreq(self.ny, d=1.0).to(dtype=dtype, device=device) * self.ny
        hz = torch.fft.rfftfreq(self.nz, d=1.0).to(dtype=dtype, device=device) * self.nz

        H, K, L = torch.meshgrid(hx, hy, hz, indexing="ij")
        # materialize (break zero-stride) and keep them out of the state_dict
        self.register_buffer("H", H.clone().contiguous(), persistent=False)
        self.register_buffer("K", K.clone().contiguous(), persistent=False)
        self.register_buffer("L", L.clone().contiguous(), persistent=False)

        self.rfftn_shape = (self.nx, self.ny, self.nzr)
        self.real_shape  = (self.nx, self.ny, self.nz)

        self.register_buffer("Gx", torch.zeros(self.rfftn_shape, dtype=dtype, device=device), persistent=False)
        self.register_buffer("Gy", torch.zeros_like(self.Gx), persistent=False)
        self.register_buffer("Gz", torch.zeros_like(self.Gx), persistent=False)
        self.register_buffer("Gnorm", torch.zeros_like(self.Gx), persistent=False)
        self.register_buffer("phi6", torch.zeros(*self.rfftn_shape, 6, dtype=dtype, device=device), persistent=False)
        self.register_buffer("lp_mask", torch.ones(self.rfftn_shape, dtype=torch.bool, device=device), persistent=False)
        cond_h = (self.H == 0)
        cond_k = (self.K == 0)
        cond_l = (self.L == 0)

        if (self.nx % 2) == 0:
            cond_h = cond_h | (self.H == -self.nx / 2)
        if (self.ny % 2) == 0:
            cond_k = cond_k | (self.K == -self.ny / 2)
        if (self.nz % 2) == 0:
            cond_l = cond_l | (self.L == self.nz / 2)  # rfftfreq is nonnegative on last axis

        real_only_mask = cond_h & cond_k & cond_l
        self.register_buffer("real_only_mask", real_only_mask, persistent=False)


    @torch.no_grad()
    def set_cell(self, cell_cart: torch.Tensor):
        dev = self.H.device
        cell_cart = cell_cart.to(dev)
        if self.cell.device != dev: self.cell = self.cell.to(dev)
        if self.B.device    != dev: self.B    = self.B.to(dev)
        self.cell.copy_(cell_cart)
        self.B.copy_(2 * math.pi * torch.linalg.inv(self.cell).mT)

        HKL = torch.stack([self.H, self.K, self.L], dim=-1)  # (..., 3)
        G = torch.einsum('...j,jk->...k', HKL, self.B)  # ✅ G = HKL @ B
        self.Gx.copy_(G[..., 0]);
        self.Gy.copy_(G[..., 1]);
        self.Gz.copy_(G[..., 2])
        G2 = (G ** 2).sum(dim=-1)
        self.Gnorm.copy_(torch.sqrt(torch.clamp_min(G2, 1e-24)))
        Gx, Gy, Gz = self.Gx, self.Gy, self.Gz

        self.phi6[...,0] = Gx*Gx; self.phi6[...,1] = Gy*Gy; self.phi6[...,2] = Gz*Gz
        self.phi6[...,3] = 2*Gx*Gy; self.phi6[...,4] = 2*Gx*Gz; self.phi6[...,5] = 2*Gy*Gz

    @property
    def volume(self):
        return torch.abs(torch.linalg.det(self.cell))

    @torch.no_grad()
    def compute_lowpass_indices(self, Gmax=None, max_modes=None):
        if Gmax is None:
            flat = torch.ones_like(self.Gnorm.reshape(-1), dtype=torch.bool)
        else:
            flat = (self.Gnorm.reshape(-1) <= Gmax)
        idx = torch.nonzero(flat, as_tuple=False).squeeze(-1)
        if (max_modes is not None) and (idx.numel() > max_modes):
            g = self.Gnorm.reshape(-1)
            idx = idx[g[idx].argsort()[:max_modes]]
        return idx

    # PWGrid helper
    @torch.no_grad()
    def compute_band_indices(self, Gmin=None, Gmax=None, max_modes=None):
        g = self.Gnorm.reshape(-1)
        mask = torch.ones_like(g, dtype=torch.bool)
        if Gmin is not None: mask &= (g >= Gmin)
        if Gmax is not None: mask &= (g <= Gmax)
        idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        if (max_modes is not None) and (idx.numel() > max_modes):
            idx = idx[g[idx].argsort()[:max_modes]]
        return idx


def struc_from_gaussians_keops(pw: PWGrid, mu_cart, Sigma_cart, w):
    N = pw.Gx.numel()
    nx, ny, nzr = pw.Gx.shape
    G  = torch.stack([pw.Gx, pw.Gy, pw.Gz], dim=-1).reshape(N, 3)
    phi = pw.phi6.reshape(N, 6)

    coeff = torch.stack([Sigma_cart[:,0,0], Sigma_cart[:,1,1], Sigma_cart[:,2,2],
                         Sigma_cart[:,0,1], Sigma_cart[:,0,2], Sigma_cart[:,1,2]], dim=1)

    G_i   = KLT(G[:,None,:])
    phi_i = KLT(phi[:,None,:])
    mu_j  = KLT(mu_cart[None,:,:])
    a_j   = KLT(coeff[None,:,:])
    w_j   = KLT(w[None,:,None])

    q_ij  = (phi_i * a_j).sum(-1)
    amp   = (-0.5*q_ij).exp()

    dot   = (G_i * mu_j).sum(-1)
    c_ij  = (dot).cos()
    s_ij  = (dot).sin()

    realN = (amp * c_ij * w_j).sum(dim=1).view(N)
    imagN = -(amp * s_ij * w_j).sum(dim=1).view(N)
    cG    = torch.complex(realN, imagN).reshape(nx, ny, nzr)
    return cG
