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
RTM Utilities Module

Common JIT-compiled utility functions for Reverse Time Migration.
All functions are optimized with Numba for maximum performance and cache efficiency.
"""

import numpy as np
from numba import jit, prange

# ============================================================================
# Material Property Computations (Harmonic Averaging and Staggered Grids)
# ============================================================================

@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def compute_shear_avg_SH(mu):
    """
    Optimized harmonic averaging for SH wave shear modulus.
    Uses parallel loops and fused computation for cache efficiency.
    
    Args:
        mu: Shear modulus array (nx, nz)
        
    Returns:
        mux, muz: Harmonic averages in x and z directions
    """
    nx, nz = mu.shape
    mux = mu.copy()
    muz = mu.copy()
    
    for i in prange(1, nx - 1):
        for j in range(1, nz - 1):
            # Harmonic mean in x-direction: 2 / (1/mu[i,j] + 1/mu[i+1,j])
            mux[i, j] = 2.0 / (1.0 / mu[i, j] + 1.0 / mu[i+1, j])
            # Harmonic mean in z-direction: 2 / (1/mu[i,j] + 1/mu[i,j+1])
            muz[i, j] = 2.0 / (1.0 / mu[i, j] + 1.0 / mu[i, j+1])
    
    return mux, muz

@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def compute_shear_avg_PSV(mu):
    """
    Optimized harmonic averaging for P-SV wave shear modulus.
    4-point harmonic mean computed in parallel for cache efficiency.
    
    Args:
        mu: Shear modulus array (nx, nz)
        
    Returns:
        muxz: 4-point harmonic average
    """
    nx, nz = mu.shape
    muxz = mu.copy()
    
    for i in prange(1, nx - 1):
        for j in range(1, nz - 1):
            # 4-point harmonic mean
            inv_sum = (1.0 / mu[i, j] + 1.0 / mu[i+1, j] + 
                      1.0 / mu[i, j+1] + 1.0 / mu[i+1, j+1])
            muxz[i, j] = 4.0 / inv_sum
    
    return muxz

@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def compute_rho_staggered(rho):
    """
    Optimized staggered grid density computation for both u and w.
    Returns both in one pass for better cache utilization.
    
    Args:
        rho: Density array (nx, nz)
        
    Returns:
        rho_u, rho_w: Staggered grid densities
    """
    nx, nz = rho.shape
    rho_u = rho.copy()
    rho_w = rho.copy()
    
    for i in prange(1, nx - 1):
        for j in range(1, nz - 1):
            # Average for u-component (staggered in x)
            rho_u[i, j] = 0.5 * (rho[i, j] + rho[i+1, j])
            # Average for w-component (staggered in z)
            rho_w[i, j] = 0.5 * (rho[i, j] + rho[i, j+1])
    
    return rho_u, rho_w

@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def precompute_dt_over_rho(dt, rho_u, rho_w, rho):
    """
    Pre-compute dt/rho arrays to avoid division in inner loops.
    This is a critical optimization for the time-stepping loop.
    
    Args:
        dt: Time step
        rho_u, rho_w, rho: Density arrays
        
    Returns:
        dt_rho_u, dt_rho_w, dt_rho: Pre-computed dt/rho arrays
    """
    nx, nz = rho.shape
    dt_rho_u = np.empty_like(rho_u)
    dt_rho_w = np.empty_like(rho_w)
    dt_rho = np.empty_like(rho)
    
    for i in prange(nx):
        for j in range(nz):
            dt_rho_u[i, j] = dt / rho_u[i, j]
            dt_rho_w[i, j] = dt / rho_w[i, j]
            dt_rho[i, j] = dt / rho[i, j]
    
    return dt_rho_u, dt_rho_w, dt_rho

@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def compute_elastic_parameters(rho, vs, vp):
    """
    Compute elastic parameters (mu, lambda) from density and velocities.
    Fused computation for cache efficiency.
    
    Args:
        rho: Density array
        vs: S-wave velocity array
        vp: P-wave velocity array
        
    Returns:
        mu, lam: Shear modulus and Lamé parameter
    """
    nx, nz = rho.shape
    mu = np.empty_like(rho)
    lam = np.empty_like(rho)
    
    for i in prange(nx):
        for j in range(nz):
            mu[i, j] = rho[i, j] * vs[i, j] * vs[i, j]
            vp_vs_ratio = vp[i, j] / vs[i, j]
            lam[i, j] = (vp_vs_ratio * vp_vs_ratio - 2.0) * mu[i, j]
    
    return mu, lam

# ============================================================================
# Absorbing Boundary Coefficients
# ============================================================================

@jit(nopython=True, fastmath=True, cache=True)
def compute_absorbing_coeff(nx, nz, FW, a=0.0053):
    """
    Optimized absorbing coefficient computation using vectorized operations.
    Pre-compute exponential decay coefficients for Cerjan boundary conditions.
    
    Args:
        nx, nz: Grid dimensions
        FW: Frame width for absorbing boundary
        a: Damping coefficient
        
    Returns:
        absorb_coeff: Absorbing coefficient array
    """
    # Pre-compute all decay coefficients
    coeff = np.exp(-a**2 * np.arange(FW, 0, -1)**2)
    absorb_coeff = np.ones((nx, nz), dtype=np.float64)
    
    # Apply to boundaries using vectorized operations where possible
    # Left boundary (x-direction)
    for i in range(FW):
        ze = nz - i - 1
        absorb_coeff[i, :ze] = coeff[i]
    
    # Right boundary (x-direction)
    for i in range(FW):
        ii = nx - i - 1
        ze = nz - i - 1
        absorb_coeff[ii, :ze] = coeff[i]
    
    # Bottom boundary (z-direction)
    for j in range(FW):
        jj = nz - j - 1
        xb = j
        xe = nx - j
        absorb_coeff[xb:xe, jj] = coeff[j]
    
    return absorb_coeff

@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def apply_absorbing_coeff(u, v, w, absorb_coeff):
    """
    Apply absorbing coefficients to velocity fields.
    Optimized with parallel execution and cache-friendly access.
    
    Args:
        u, v, w: Velocity field arrays (modified in-place)
        absorb_coeff: Absorbing coefficient array
    """
    nx, nz = u.shape
    for i in prange(nx):
        for j in range(nz):
            coeff = absorb_coeff[i, j]
            u[i, j] *= coeff
            v[i, j] *= coeff
            w[i, j] *= coeff

@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def apply_absorbing_coeff_stress(sxx, szz, sxz, syx, syz, absorb_coeff):
    """
    Apply absorbing coefficients to stress fields.
    Optimized with single coefficient load for multiple arrays.
    
    Args:
        sxx, szz, sxz, syx, syz: Stress field arrays (modified in-place)
        absorb_coeff: Absorbing coefficient array
    """
    nx, nz = sxx.shape
    for i in prange(nx):
        for j in range(nz):
            coeff = absorb_coeff[i, j]
            sxx[i, j] *= coeff
            szz[i, j] *= coeff
            sxz[i, j] *= coeff
            syx[i, j] *= coeff
            syz[i, j] *= coeff

# ============================================================================
# Array Validation and Utilities
# ============================================================================

@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def check_array_finite(arr):
    """
    Check if array contains only finite values (no NaN or Inf).
    Optimized JIT version with parallel loops for fast validation.
    
    Uses aggressive optimizations:
    - parallel=True: Multi-threaded checking across rows
    - fastmath=True: Fast floating-point comparison
    
    Args:
        arr: Array to check (nx, nz)
        
    Returns:
        bool: True if all values are finite, False otherwise
    """
    nx, nz = arr.shape
    has_invalid = False
    for i in prange(nx):
        if has_invalid:
            break
        for j in range(nz):
            val = arr[i, j]
            if not (val == val and val != np.inf and val != -np.inf):
                has_invalid = True
                break
    return not has_invalid

@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def initialize_wavefield_arrays(nx, nz, dtype=np.float64):
    """
    Initialize wavefield arrays (velocity and stress).
    Returns all arrays initialized to zero for better cache locality.
    
    Args:
        nx, nz: Grid dimensions
        dtype: Data type for arrays
        
    Returns:
        u, v, w, sxx, szz, sxz, syx, syz: Initialized arrays
    """
    u = np.zeros((nx, nz), dtype=dtype)
    v = np.zeros((nx, nz), dtype=dtype)
    w = np.zeros((nx, nz), dtype=dtype)
    sxx = np.zeros((nx, nz), dtype=dtype)
    szz = np.zeros((nx, nz), dtype=dtype)
    sxz = np.zeros((nx, nz), dtype=dtype)
    syx = np.zeros((nx, nz), dtype=dtype)
    syz = np.zeros((nx, nz), dtype=dtype)
    
    return u, v, w, sxx, szz, sxz, syx, syz

@jit(nopython=True, fastmath=True, cache=True)
def gaussian_source_wavelet(nt, dt, f0):
    """
    Generate Gaussian derivative source wavelet.
    Optimized JIT version for fast wavelet generation.
    
    Args:
        nt: Number of time steps
        dt: Time step
        f0: Central frequency
        
    Returns:
        src: Source wavelet array
    """
    time = np.arange(nt, dtype=np.float64) * dt
    t0 = 3.0 / f0
    src = -2.0 * (time - t0) * (f0 * f0) * np.exp(-(f0 * f0) * (time - t0) * (time - t0))
    return src

# ============================================================================
# Matrix Operations for Imaging Conditions
# ============================================================================

@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def compute_cross_correlation(fw_u, fw_v, fw_w, bw_u, bw_v, bw_w, result_u, result_v, result_w):
    """
    Compute cross-correlation imaging condition.
    Fused operation for all three components to maximize cache efficiency.
    
    Args:
        fw_u, fw_v, fw_w: Forward propagated wavefields
        bw_u, bw_v, bw_w: Backward propagated wavefields
        result_u, result_v, result_w: Result arrays (accumulated in-place)
    """
    nx, nz = fw_u.shape
    for i in prange(nx):
        for j in range(nz):
            result_u[i, j] += fw_u[i, j] * bw_u[i, j]
            result_v[i, j] += fw_v[i, j] * bw_v[i, j]
            result_w[i, j] += fw_w[i, j] * bw_w[i, j]

@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def normalize_image(image, mean_subtraction=True):
    """
    Normalize imaging result.
    Optionally subtract mean and scale to [-1, 1] range.
    
    Args:
        image: Image array to normalize
        mean_subtraction: Whether to subtract mean
        
    Returns:
        Normalized image array
    """
    nx, nz = image.shape
    result = image.copy()
    
    if mean_subtraction:
        # Compute mean
        total = 0.0
        count = nx * nz
        for i in range(nx):
            for j in range(nz):
                total += result[i, j]
        mean_val = total / count
        
        # Subtract mean
        for i in prange(nx):
            for j in range(nz):
                result[i, j] -= mean_val
    
    # Find max absolute value for scaling
    max_abs = 0.0
    for i in range(nx):
        for j in range(nz):
            abs_val = abs(result[i, j])
            if abs_val > max_abs:
                max_abs = abs_val
    
    # Scale to [-1, 1] if max_abs > 0
    if max_abs > 1e-10:
        scale = 1.0 / max_abs
        for i in prange(nx):
            for j in range(nz):
                result[i, j] *= scale
    
    return result

@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def stack_images(image_list):
    """
    Stack multiple imaging results.
    Optimized parallel summation.
    
    Args:
        image_list: List of image arrays to stack
        
    Returns:
        Stacked image
    """
    n_images = len(image_list)
    if n_images == 0:
        raise ValueError("Empty image list")
    
    nx, nz = image_list[0].shape
    result = np.zeros((nx, nz), dtype=np.float64)
    
    for img_idx in range(n_images):
        img = image_list[img_idx]
        for i in prange(nx):
            for j in range(nz):
                result[i, j] += img[i, j]
    
    return result

# ============================================================================
# Velocity Model Utilities
# ============================================================================

@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def create_constant_velocity_model(nx, nz, vs, vp, rho):
    """
    Create constant velocity and density model.
    Optimized initialization for homogeneous models.
    
    Args:
        nx, nz: Grid dimensions
        vs, vp, rho: Constant values for S-wave, P-wave velocity and density
        
    Returns:
        vs_arr, vp_arr, rho_arr: Initialized arrays
    """
    vs_arr = np.empty((nx, nz), dtype=np.float64)
    vp_arr = np.empty((nx, nz), dtype=np.float64)
    rho_arr = np.empty((nx, nz), dtype=np.float64)
    
    for i in prange(nx):
        for j in range(nz):
            vs_arr[i, j] = vs
            vp_arr[i, j] = vp
            rho_arr[i, j] = rho
    
    return vs_arr, vp_arr, rho_arr

@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def apply_surface_boundary(field, surface_matrix):
    """
    Apply surface boundary condition by zeroing out values above surface.
    
    Optimized with parallel execution for large grids.
    Uses fastmath for efficient multiplication.
    
    Args:
        field: Wavefield array (nx, nz) - modified in-place
        surface_matrix: Surface matrix (nx, nz) - 0 above surface, 1 below
    """
    nx, nz = field.shape
    for i in prange(nx):
        for j in range(nz):
            field[i, j] *= surface_matrix[i, j]

# ============================================================================
# Performance Utilities
# ============================================================================

def get_optimal_dtype(precision='double'):
    """
    Get optimal data type for given precision.
    
    Args:
        precision: 'single' or 'double'
        
    Returns:
        NumPy dtype
    """
    if precision == 'single':
        return np.float32
    elif precision == 'double':
        return np.float64
    else:
        raise ValueError(f"Unknown precision: {precision}")

def estimate_memory_usage(nx, nz, nt, n_snapshots, dtype=np.float64):
    """
    Estimate memory usage for RTM computation.
    
    Args:
        nx, nz: Spatial grid dimensions
        nt: Number of time steps
        n_snapshots: Number of snapshots to store
        dtype: Data type
        
    Returns:
        Memory usage in MB
    """
    bytes_per_value = np.dtype(dtype).itemsize
    
    # Wavefield arrays (8 arrays: u,v,w,sxx,szz,sxz,syx,syz)
    current_fields = 8 * nx * nz * bytes_per_value
    
    # Snapshot storage (3 components: u,v,w)
    snapshots = 3 * nx * nz * n_snapshots * bytes_per_value
    
    # Material properties (several arrays)
    material_props = 10 * nx * nz * bytes_per_value
    
    # Seismogram storage
    # Assuming reasonable number of receivers
    seismogram = 3 * 100 * nt * bytes_per_value
    
    total_bytes = current_fields + snapshots + material_props + seismogram
    return total_bytes / (1024 * 1024)  # Convert to MB
