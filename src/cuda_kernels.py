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
CUDA GPU Kernels for Seismic Wave Propagation

High-performance GPU kernels for finite difference wave propagation.
"""

import numpy as np
from numba import cuda
import math

@cuda.jit
def update_vel_order2_cuda(u, w, v, sxx, szz, sxz, syx, syz, inv_dx, inv_dz, dt_rho_u, dt_rho_w, dt_rho):
    """
    CUDA kernel for velocity update (order 2).
    Each thread handles one grid point.
    """
    i, j = cuda.grid(2)
    nx, nz = u.shape
    
    if 0 < i < nx - 1 and 0 < j < nz - 1:
        # P-SV wave
        sxx_x = (sxx[i+1, j] - sxx[i, j]) * inv_dx
        szz_z = (szz[i, j+1] - szz[i, j]) * inv_dz
        sxz_x = (sxz[i+1, j] - sxz[i, j]) * inv_dx
        sxz_z = (sxz[i, j+1] - sxz[i, j]) * inv_dz
        
        u[i, j] -= (sxx_x + sxz_z) * dt_rho_u[i, j]
        w[i, j] -= (sxz_x + szz_z) * dt_rho_w[i, j]
        
        # SH wave
        syx_x = (syx[i+1, j] - syx[i, j]) * inv_dx
        syz_z = (syz[i, j+1] - syz[i, j]) * inv_dz
        v[i, j] -= (syx_x + syz_z) * dt_rho[i, j]

@cuda.jit
def update_vel_order2_fw_cuda(u, w, v, sxx, szz, sxz, syx, syz, inv_dx, inv_dz, dt_rho_u, dt_rho_w, dt_rho):
    """
    CUDA kernel for forward velocity update (order 2).
    """
    i, j = cuda.grid(2)
    nx, nz = u.shape
    
    if 0 < i < nx - 1 and 0 < j < nz - 1:
        # P-SV wave
        sxx_x = (sxx[i, j] - sxx[i-1, j]) * inv_dx
        szz_z = (szz[i, j] - szz[i, j-1]) * inv_dz
        sxz_x = (sxz[i, j] - sxz[i-1, j]) * inv_dx
        sxz_z = (sxz[i, j] - sxz[i, j-1]) * inv_dz
        
        u[i, j] += (sxx_x + sxz_z) * dt_rho_u[i, j]
        w[i, j] += (sxz_x + szz_z) * dt_rho_w[i, j]
        
        # SH wave
        syx_x = (syx[i, j] - syx[i-1, j]) * inv_dx
        syz_z = (syz[i, j] - syz[i, j-1]) * inv_dz
        v[i, j] += (syx_x + syz_z) * dt_rho[i, j]

@cuda.jit
def update_str_order2_cuda(u, w, v, sxx, szz, sxz, syx, syz, inv_dx, inv_dz, neg_dt, lam, mu, mxz, myx, myz):
    """
    CUDA kernel for stress update (order 2) - backward propagation.
    """
    i, j = cuda.grid(2)
    nx, nz = u.shape
    
    if 0 < i < nx - 1 and 0 < j < nz - 1:
        # Compute derivatives
        u_x = (u[i, j] - u[i-1, j]) * inv_dx
        u_z = (u[i, j] - u[i, j-1]) * inv_dz
        w_x = (w[i, j] - w[i-1, j]) * inv_dx
        w_z = (w[i, j] - w[i, j-1]) * inv_dz
        
        # P-SV wave
        div_vel = u_x + w_z
        lam_div = lam[i, j] * div_vel
        
        sxx[i, j] += neg_dt * (lam_div + 2.0 * mu[i, j] * u_x)
        szz[i, j] += neg_dt * (lam_div + 2.0 * mu[i, j] * w_z)
        sxz[i, j] += neg_dt * mxz[i, j] * (u_z + w_x)
        
        # SH wave
        v_x = (v[i, j] - v[i-1, j]) * inv_dx
        v_z = (v[i, j] - v[i, j-1]) * inv_dz
        syx[i, j] += neg_dt * myx[i, j] * v_x
        syz[i, j] += neg_dt * myz[i, j] * v_z

@cuda.jit
def update_str_order2_fw_cuda(u, w, v, sxx, szz, sxz, syx, syz, inv_dx, inv_dz, dt, lam, mu, mxz, myx, myz):
    """
    CUDA kernel for stress update (order 2) - forward propagation.
    """
    i, j = cuda.grid(2)
    nx, nz = u.shape
    
    if 0 < i < nx - 1 and 0 < j < nz - 1:
        # Compute derivatives
        u_x = (u[i+1, j] - u[i, j]) * inv_dx
        u_z = (u[i, j+1] - u[i, j]) * inv_dz
        w_x = (w[i+1, j] - w[i, j]) * inv_dx
        w_z = (w[i, j+1] - w[i, j]) * inv_dz
        
        # P-SV wave
        div_vel = u_x + w_z
        lam_div = lam[i, j] * div_vel
        
        sxx[i, j] += dt * (lam_div + 2.0 * mu[i, j] * u_x)
        szz[i, j] += dt * (lam_div + 2.0 * mu[i, j] * w_z)
        sxz[i, j] += dt * mxz[i, j] * (u_z + w_x)
        
        # SH wave
        v_x = (v[i+1, j] - v[i, j]) * inv_dx
        v_z = (v[i, j+1] - v[i, j]) * inv_dz
        syx[i, j] += dt * myx[i, j] * v_x
        syz[i, j] += dt * myz[i, j] * v_z

@cuda.jit
def apply_absorbing_coeff_cuda(u, v, w, absorb_coeff):
    """
    CUDA kernel for applying absorbing coefficients to velocity fields.
    """
    i, j = cuda.grid(2)
    nx, nz = u.shape
    
    if i < nx and j < nz:
        coeff = absorb_coeff[i, j]
        u[i, j] *= coeff
        v[i, j] *= coeff
        w[i, j] *= coeff

@cuda.jit
def apply_absorbing_coeff_stress_cuda(sxx, szz, sxz, syx, syz, absorb_coeff):
    """
    CUDA kernel for applying absorbing coefficients to stress fields.
    """
    i, j = cuda.grid(2)
    nx, nz = sxx.shape
    
    if i < nx and j < nz:
        coeff = absorb_coeff[i, j]
        sxx[i, j] *= coeff
        szz[i, j] *= coeff
        sxz[i, j] *= coeff
        syx[i, j] *= coeff
        syz[i, j] *= coeff

def get_cuda_grid_config(nx, nz, threads_per_block=(16, 16)):
    """
    Calculate optimal CUDA grid configuration.
    
    Args:
        nx, nz: Grid dimensions
        threads_per_block: Threads per block (tuple)
        
    Returns:
        (threads_per_block, blocks_per_grid) tuples
    """
    threads_x, threads_z = threads_per_block
    blocks_x = (nx + threads_x - 1) // threads_x
    blocks_z = (nz + threads_z - 1) // threads_z
    blocks_per_grid = (blocks_x, blocks_z)
    
    return threads_per_block, blocks_per_grid

# Wrapper functions for easier use
def cuda_update_vel_order2(u, w, v, sxx, szz, sxz, syx, syz, inv_dx, inv_dz, dt_rho_u, dt_rho_w, dt_rho):
    """Wrapper for CUDA velocity update (backward)"""
    nx, nz = u.shape
    threads_per_block, blocks_per_grid = get_cuda_grid_config(nx, nz)
    update_vel_order2_cuda[blocks_per_grid, threads_per_block](
        u, w, v, sxx, szz, sxz, syx, syz, inv_dx, inv_dz, dt_rho_u, dt_rho_w, dt_rho
    )
    cuda.synchronize()

def cuda_update_vel_order2_fw(u, w, v, sxx, szz, sxz, syx, syz, inv_dx, inv_dz, dt_rho_u, dt_rho_w, dt_rho):
    """Wrapper for CUDA velocity update (forward)"""
    nx, nz = u.shape
    threads_per_block, blocks_per_grid = get_cuda_grid_config(nx, nz)
    update_vel_order2_fw_cuda[blocks_per_grid, threads_per_block](
        u, w, v, sxx, szz, sxz, syx, syz, inv_dx, inv_dz, dt_rho_u, dt_rho_w, dt_rho
    )
    cuda.synchronize()

def cuda_update_str_order2(u, w, v, sxx, szz, sxz, syx, syz, inv_dx, inv_dz, neg_dt, lam, mu, mxz, myx, myz):
    """Wrapper for CUDA stress update (backward)"""
    nx, nz = u.shape
    threads_per_block, blocks_per_grid = get_cuda_grid_config(nx, nz)
    update_str_order2_cuda[blocks_per_grid, threads_per_block](
        u, w, v, sxx, szz, sxz, syx, syz, inv_dx, inv_dz, neg_dt, lam, mu, mxz, myx, myz
    )
    cuda.synchronize()

def cuda_update_str_order2_fw(u, w, v, sxx, szz, sxz, syx, syz, inv_dx, inv_dz, dt, lam, mu, mxz, myx, myz):
    """Wrapper for CUDA stress update (forward)"""
    nx, nz = u.shape
    threads_per_block, blocks_per_grid = get_cuda_grid_config(nx, nz)
    update_str_order2_fw_cuda[blocks_per_grid, threads_per_block](
        u, w, v, sxx, szz, sxz, syx, syz, inv_dx, inv_dz, dt, lam, mu, mxz, myx, myz
    )
    cuda.synchronize()

def cuda_apply_absorbing_coeff(u, v, w, absorb_coeff):
    """Wrapper for CUDA absorbing coefficient application"""
    nx, nz = u.shape
    threads_per_block, blocks_per_grid = get_cuda_grid_config(nx, nz)
    apply_absorbing_coeff_cuda[blocks_per_grid, threads_per_block](u, v, w, absorb_coeff)
    cuda.synchronize()

def cuda_apply_absorbing_coeff_stress(sxx, szz, sxz, syx, syz, absorb_coeff):
    """Wrapper for CUDA absorbing coefficient application to stress"""
    nx, nz = sxx.shape
    threads_per_block, blocks_per_grid = get_cuda_grid_config(nx, nz)
    apply_absorbing_coeff_stress_cuda[blocks_per_grid, threads_per_block](
        sxx, szz, sxz, syx, syz, absorb_coeff
    )
    cuda.synchronize()
