# <PROJECT NAME>
# Copyright (C) <2025>  <Yutaro Hara>
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License
# as published by the Free Software Foundation; either version 2.1
# of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this library. If not, see <https://www.gnu.org/licenses/>.

"""
Unified Compute Kernels Module

Provides a unified interface for both CPU and GPU computation.
Automatically selects the best backend based on available hardware.
"""

import numpy as np
from numba import jit, prange

# Try to import CUDA and device backend
try:
    from device_backend import get_backend
    from cuda_kernels import (
        cuda_update_vel_order2, cuda_update_vel_order2_fw,
        cuda_update_str_order2, cuda_update_str_order2_fw,
        cuda_apply_absorbing_coeff, cuda_apply_absorbing_coeff_stress
    )
    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False
    get_backend = None

# CPU Kernels (Numba JIT-compiled)
@jit(nopython=True, parallel=True, fastmath=True, cache=True, inline='always')
def cpu_update_vel_order2(u, w, v, sxx, szz, sxz, syx, syz, inv_dx, inv_dz, dt, dt_rho_u, dt_rho_w, dt_rho):
    """CPU JIT kernel for backward velocity update"""
    nx, nz = u.shape
    for i in prange(1, nx - 1):
        for j in range(1, nz - 1):
            sxx_i1 = sxx[i+1, j]
            sxx_i = sxx[i, j]
            szz_j1 = szz[i, j+1]
            szz_j = szz[i, j]
            sxz_i1 = sxz[i+1, j]
            sxz_i = sxz[i, j]
            sxz_j1 = sxz[i, j+1]
            
            sxx_x = (sxx_i1 - sxx_i) * inv_dx
            szz_z = (szz_j1 - szz_j) * inv_dz
            sxz_x = (sxz_i1 - sxz_i) * inv_dx
            sxz_z = (sxz_j1 - sxz_i) * inv_dz
            
            u[i, j] -= (sxx_x + sxz_z) * dt_rho_u[i, j]
            w[i, j] -= (sxz_x + szz_z) * dt_rho_w[i, j]
            
            syx_x = (syx[i+1, j] - syx[i, j]) * inv_dx
            syz_z = (syz[i, j+1] - syz[i, j]) * inv_dz
            v[i, j] -= (syx_x + syz_z) * dt_rho[i, j]

@jit(nopython=True, parallel=True, fastmath=True, cache=True, inline='always')
def cpu_update_vel_order2_fw(u, w, v, sxx, szz, sxz, syx, syz, inv_dx, inv_dz, dt, dt_rho_u, dt_rho_w, dt_rho):
    """CPU JIT kernel for forward velocity update"""
    nx, nz = u.shape
    for i in prange(1, nx - 1):
        for j in range(1, nz - 1):
            sxx_i = sxx[i, j]
            sxx_im1 = sxx[i-1, j]
            szz_j = szz[i, j]
            szz_jm1 = szz[i, j-1]
            sxz_i = sxz[i, j]
            sxz_im1 = sxz[i-1, j]
            sxz_jm1 = sxz[i, j-1]
            
            sxx_x = (sxx_i - sxx_im1) * inv_dx
            szz_z = (szz_j - szz_jm1) * inv_dz
            sxz_x = (sxz_i - sxz_im1) * inv_dx
            sxz_z = (sxz_i - sxz_jm1) * inv_dz
            
            u[i, j] += (sxx_x + sxz_z) * dt_rho_u[i, j]
            w[i, j] += (sxz_x + szz_z) * dt_rho_w[i, j]
            
            syx_x = (syx[i, j] - syx[i-1, j]) * inv_dx
            syz_z = (syz[i, j] - syz[i, j-1]) * inv_dz
            v[i, j] += (syx_x + syz_z) * dt_rho[i, j]

@jit(nopython=True, parallel=True, fastmath=True, cache=True, inline='always')
def cpu_update_str_order2(u, w, v, sxx, szz, sxz, syx, syz, inv_dx, inv_dz, neg_dt, lam, mu, mxz, myx, myz):
    """CPU JIT kernel for backward stress update"""
    nx, nz = u.shape
    for i in prange(1, nx - 1):
        for j in range(1, nz - 1):
            u_ij = u[i, j]
            w_ij = w[i, j]
            v_ij = v[i, j]
            
            u_x = (u_ij - u[i-1, j]) * inv_dx
            u_z = (u_ij - u[i, j-1]) * inv_dz
            w_x = (w_ij - w[i-1, j]) * inv_dx
            w_z = (w_ij - w[i, j-1]) * inv_dz
            
            div_vel = u_x + w_z
            lam_div = lam[i, j] * div_vel
            
            sxx[i, j] += neg_dt * (lam_div + 2.0 * mu[i, j] * u_x)
            szz[i, j] += neg_dt * (lam_div + 2.0 * mu[i, j] * w_z)
            sxz[i, j] += neg_dt * mxz[i, j] * (u_z + w_x)
            
            v_x = (v_ij - v[i-1, j]) * inv_dx
            v_z = (v_ij - v[i, j-1]) * inv_dz
            syx[i, j] += neg_dt * myx[i, j] * v_x
            syz[i, j] += neg_dt * myz[i, j] * v_z

@jit(nopython=True, parallel=True, fastmath=True, cache=True, inline='always')
def cpu_update_str_order2_fw(u, w, v, sxx, szz, sxz, syx, syz, inv_dx, inv_dz, dt, lam, mu, mxz, myx, myz):
    """CPU JIT kernel for forward stress update"""
    nx, nz = u.shape
    for i in prange(1, nx - 1):
        for j in range(1, nz - 1):
            u_ij = u[i, j]
            w_ij = w[i, j]
            v_ij = v[i, j]
            
            u_x = (u[i+1, j] - u_ij) * inv_dx
            u_z = (u[i, j+1] - u_ij) * inv_dz
            w_x = (w[i+1, j] - w_ij) * inv_dx
            w_z = (w[i, j+1] - w_ij) * inv_dz
            
            div_vel = u_x + w_z
            lam_div = lam[i, j] * div_vel
            
            sxx[i, j] += dt * (lam_div + 2.0 * mu[i, j] * u_x)
            szz[i, j] += dt * (lam_div + 2.0 * mu[i, j] * w_z)
            sxz[i, j] += dt * mxz[i, j] * (u_z + w_x)
            
            v_x = (v[i+1, j] - v_ij) * inv_dx
            v_z = (v[i, j+1] - v_ij) * inv_dz
            syx[i, j] += dt * myx[i, j] * v_x
            syz[i, j] += dt * myz[i, j] * v_z

@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def cpu_apply_absorbing_coeff(u, v, w, absorb_coeff):
    """CPU JIT kernel for applying absorbing coefficients"""
    nx, nz = u.shape
    for i in prange(nx):
        for j in range(nz):
            coeff = absorb_coeff[i, j]
            u[i, j] *= coeff
            v[i, j] *= coeff
            w[i, j] *= coeff

@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def cpu_apply_absorbing_coeff_stress(sxx, szz, sxz, syx, syz, absorb_coeff):
    """CPU JIT kernel for applying absorbing coefficients to stress"""
    nx, nz = sxx.shape
    for i in prange(nx):
        for j in range(nz):
            coeff = absorb_coeff[i, j]
            sxx[i, j] *= coeff
            szz[i, j] *= coeff
            sxz[i, j] *= coeff
            syx[i, j] *= coeff
            syz[i, j] *= coeff

# Unified interface class
class ComputeKernels:
    """
    Unified interface for compute kernels.
    Automatically uses GPU or CPU based on backend.
    """
    
    def __init__(self, force_cpu=False):
        """Initialize compute kernels with backend selection"""
        self.backend = None
        self.use_gpu = False
        
        if BACKEND_AVAILABLE and not force_cpu:
            self.backend = get_backend(force_cpu=force_cpu)
            self.use_gpu = self.backend.is_cuda()
        
        if self.use_gpu:
            print(f"[Compute] Using GPU acceleration: {self.backend.device_name}")
        else:
            print("[Compute] Using CPU with Numba JIT parallelization")
    
    def update_vel_order2(self, u, w, v, sxx, szz, sxz, syx, syz, inv_dx, inv_dz, dt, dt_rho_u, dt_rho_w, dt_rho):
        """Backward velocity update"""
        if self.use_gpu:
            cuda_update_vel_order2(u, w, v, sxx, szz, sxz, syx, syz, inv_dx, inv_dz, dt_rho_u, dt_rho_w, dt_rho)
        else:
            cpu_update_vel_order2(u, w, v, sxx, szz, sxz, syx, syz, inv_dx, inv_dz, dt, dt_rho_u, dt_rho_w, dt_rho)
    
    def update_vel_order2_fw(self, u, w, v, sxx, szz, sxz, syx, syz, inv_dx, inv_dz, dt, dt_rho_u, dt_rho_w, dt_rho):
        """Forward velocity update"""
        if self.use_gpu:
            cuda_update_vel_order2_fw(u, w, v, sxx, szz, sxz, syx, syz, inv_dx, inv_dz, dt_rho_u, dt_rho_w, dt_rho)
        else:
            cpu_update_vel_order2_fw(u, w, v, sxx, szz, sxz, syx, syz, inv_dx, inv_dz, dt, dt_rho_u, dt_rho_w, dt_rho)
    
    def update_str_order2(self, u, w, v, sxx, szz, sxz, syx, syz, inv_dx, inv_dz, neg_dt, lam, mu, mxz, myx, myz):
        """Backward stress update"""
        if self.use_gpu:
            cuda_update_str_order2(u, w, v, sxx, szz, sxz, syx, syz, inv_dx, inv_dz, neg_dt, lam, mu, mxz, myx, myz)
        else:
            cpu_update_str_order2(u, w, v, sxx, szz, sxz, syx, syz, inv_dx, inv_dz, neg_dt, lam, mu, mxz, myx, myz)
    
    def update_str_order2_fw(self, u, w, v, sxx, szz, sxz, syx, syz, inv_dx, inv_dz, dt, lam, mu, mxz, myx, myz):
        """Forward stress update"""
        if self.use_gpu:
            cuda_update_str_order2_fw(u, w, v, sxx, szz, sxz, syx, syz, inv_dx, inv_dz, dt, lam, mu, mxz, myx, myz)
        else:
            cpu_update_str_order2_fw(u, w, v, sxx, szz, sxz, syx, syz, inv_dx, inv_dz, dt, lam, mu, mxz, myx, myz)
    
    def apply_absorbing_coeff(self, u, v, w, absorb_coeff):
        """Apply absorbing coefficients to velocity"""
        if self.use_gpu:
            cuda_apply_absorbing_coeff(u, v, w, absorb_coeff)
        else:
            cpu_apply_absorbing_coeff(u, v, w, absorb_coeff)
    
    def apply_absorbing_coeff_stress(self, sxx, szz, sxz, syx, syz, absorb_coeff):
        """Apply absorbing coefficients to stress"""
        if self.use_gpu:
            cuda_apply_absorbing_coeff_stress(sxx, szz, sxz, syx, syz, absorb_coeff)
        else:
            cpu_apply_absorbing_coeff_stress(sxx, szz, sxz, syx, syz, absorb_coeff)
