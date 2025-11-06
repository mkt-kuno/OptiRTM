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

# 1. 順方向モデリング
import numpy as np
from numba import jit, prange
import os
from icecream import ic

# Matplotlib is optional - only import if needed for visualization
_plt = None
_matplotlib = None
def _ensure_matplotlib():
    """Lazy import matplotlib only when needed"""
    global _plt, _matplotlib
    if _plt is None:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend for environments without display
        import matplotlib.pyplot as plt
        _matplotlib = matplotlib
        _plt = plt
    return _plt
# Import common RTM utilities
from rtm_utils import (
    compute_shear_avg_SH, compute_shear_avg_PSV, compute_rho_staggered,
    precompute_dt_over_rho, compute_absorbing_coeff, apply_absorbing_coeff,
    apply_absorbing_coeff_stress, check_array_finite, gaussian_source_wavelet
)

# Configure icecream for better logging
ic.configureOutput(prefix='[Forward] ', includeContext=True)

# Highly optimized JIT functions for forward modeling
@jit(nopython=True, parallel=True, fastmath=True, cache=True, inline='always')
def update_vel_order2_fw_jit(u, w, v, sxx, szz, sxz, syx, syz, inv_dx, inv_dz, dt, dt_rho_u, dt_rho_w, dt_rho):
    """
    Aggressively optimized velocity update for forward modeling with:
    - Pre-computed reciprocals (inv_dx, inv_dz, dt/rho)
    - Fused P-SV and SH wave updates
    - Cache-friendly access patterns
    - Reduced memory loads through register reuse
    """
    nx, nz = u.shape
    for i in prange(1, nx - 1):
        for j in range(1, nz - 1):
            # Load stress values once
            sxx_i = sxx[i, j]
            sxx_im1 = sxx[i-1, j]
            szz_j = szz[i, j]
            szz_jm1 = szz[i, j-1]
            sxz_i = sxz[i, j]
            sxz_im1 = sxz[i-1, j]
            sxz_jm1 = sxz[i, j-1]
            
            # P-SV wave with pre-computed reciprocals
            sxx_x = (sxx_i - sxx_im1) * inv_dx
            szz_z = (szz_j - szz_jm1) * inv_dz
            sxz_x = (sxz_i - sxz_im1) * inv_dx
            sxz_z = (sxz_i - sxz_jm1) * inv_dz
            
            # Fused multiply-add operations
            u[i, j] += (sxx_x + sxz_z) * dt_rho_u[i, j]
            w[i, j] += (sxz_x + szz_z) * dt_rho_w[i, j]
            
            # SH wave inline
            syx_x = (syx[i, j] - syx[i-1, j]) * inv_dx
            syz_z = (syz[i, j] - syz[i, j-1]) * inv_dz
            v[i, j] += (syx_x + syz_z) * dt_rho[i, j]

@jit(nopython=True, parallel=True, fastmath=True, cache=True, inline='always')
def update_str_order2_fw_jit(u, w, v, sxx, szz, sxz, syx, syz, inv_dx, inv_dz, dt, lam, mu, mxz, myx, myz):
    """
    Highly optimized stress update for forward modeling with:
    - Pre-computed reciprocals
    - Reduced redundant calculations
    - Register reuse for velocity values
    """
    nx, nz = u.shape
    for i in prange(1, nx - 1):
        for j in range(1, nz - 1):
            # Load and reuse velocity values
            u_ij = u[i, j]
            w_ij = w[i, j]
            v_ij = v[i, j]
            
            # Compute derivatives with pre-computed reciprocals
            u_x = (u[i+1, j] - u_ij) * inv_dx
            u_z = (u[i, j+1] - u_ij) * inv_dz
            w_x = (w[i+1, j] - w_ij) * inv_dx
            w_z = (w[i, j+1] - w_ij) * inv_dz
            
            # Compute common term once
            div_vel = u_x + w_z
            lam_div = lam[i, j] * div_vel
            
            # P-SV wave with fused operations
            sxx[i, j] += dt * (lam_div + 2.0 * mu[i, j] * u_x)
            szz[i, j] += dt * (lam_div + 2.0 * mu[i, j] * w_z)
            sxz[i, j] += dt * mxz[i, j] * (u_z + w_x)
            
            # SH wave inline
            v_x = (v[i+1, j] - v_ij) * inv_dx
            v_z = (v[i, j+1] - v_ij) * inv_dz
            syx[i, j] += dt * myx[i, j] * v_x
            syz[i, j] += dt * myz[i, j] * v_z

# Removed duplicate functions - now using shared utilities from rtm_utils.py:
# - apply_absorbing_coeff, apply_absorbing_coeff_stress
# - compute_absorbing_coeff
# - compute_shear_avg_SH, compute_shear_avg_PSV
# - compute_rho_staggered
# - precompute_dt_over_rho

"""
    i,j               i+1,j
u,w:o---------o--------o
 rho|sxx               |
mulm|szz               |
    |                  |
  v:o        mxz       o
syz |        sxz       |
myz |                  |
    |                  |
    o------------------o
i,j+1                  i+1,j+1
"""
class forward_modeling:
    """
    kwargs:
    nx:int   x方向のグリッド数
    nz:int   z方向のグリッド数
    dx:float x方向のグリッド間隔
    dz:float z方向のグリッド間隔
    nt:int   シミュレーション時間ステップ数
    fs:float サンプリング周波数
    vs:np.array S波速度
    rho:np.array 密度
    absorbing_frame:int 吸収境界の幅
    src_loc:list 震源の位置  [[i1,j1],[i2,j2],...]
    wavelet:np.array 震源波形
    receiver_loc:list 受信機の位置 [[i1,j1],[i2,j2],...]

    isnap:int 途中経過の表示ステップ数 default:10
    order:int 空間微分のオーダー(2 or 3) dedault:2
    """
    
    def __init__(self, **kwargs):
        self.nx = kwargs['nx']
        self.nz = kwargs['nz']
        self.dx = kwargs['dx']
        self.dz = kwargs['dz']
        self.nt = kwargs['nt']
        self.fs = kwargs['fs']
        self.vs = kwargs['vs'] if 'vs' in kwargs else np.ones((self.nx,self.nz), dtype =np.float32)*200
        self.vp = kwargs['vp'] if 'vp' in kwargs else self.vs*np.sqrt(6) # at least root2 times larger than vs, poisson ratio = 0.25, vp/vs = 1.7320508, 
        self.rho= kwargs['rho']if 'rho'in kwargs else np.ones((self.nx,self.nz), dtype =np.float32)*1800
        self.absorbing_frame = kwargs['absorbing_frame'] if 'absorbing_frame' in kwargs else 60
        self.src_loc = kwargs['src_loc']if 'src_loc'in kwargs else [self.nx // 2,0] #source location, (i,j)
        self.wavelet_u = kwargs['wavelet_u']if 'wavelet_u'in kwargs else None
        self.wavelet_v = kwargs['wavelet_v']if 'wavelet_v'in kwargs else None
        self.wavelet_w = kwargs['wavelet_w']if 'wavelet_w'in kwargs else None
        self.f0 = kwargs['f0'] if 'f0' in kwargs else None
        self.receiver_loc = kwargs['receiver_loc'] #receiver location, (i,j)
        self.isnap = kwargs['isnap']if 'isnap'in kwargs else 10
        self.order = kwargs['order']if 'order'in kwargs else 2
        self.receivers_height = kwargs['receivers_height'] if 'receivers_height' in kwargs else None ##
        self.surface_matrix = kwargs['surface_matrix'] if 'surface_matrix' in kwargs else None
        self.steepness_array = kwargs['steepness_array'] if 'steepness_array' in kwargs else None
        # Visualization control parameters
        self.enable_matplotlib = kwargs.get('enable_matplotlib', True)  # Enable matplotlib by default for backward compatibility
        self.plot_callback = kwargs.get('plot_callback', None)  # Optional callback for custom visualization

    def initialize(self):
        """
        Initialize forward modeling with optimized pre-computations.
        
        This method:
        1. Initializes seismogram arrays for recording at receivers
        2. Computes elastic parameters (mu, lambda) from velocity model
        3. Initializes wavefield arrays (velocity and stress)
        4. Applies harmonic averaging for staggered grid
        5. Pre-computes reciprocals (inv_dx, inv_dz, dt/rho) for optimal performance
        6. Generates absorbing boundary coefficients
        7. Initializes source wavelets
        
        All array operations use JIT-compiled functions with parallel=True and fastmath=True.
        """
        ic("Initializing forward modeling")
        ic(self.nx, self.nz, self.nt)
        
        self.seismogram_u = np.zeros((len(self.receiver_loc), self.nt), dtype=np.float32)
        self.seismogram_v = np.zeros((len(self.receiver_loc), self.nt), dtype=np.float32)
        self.seismogram_w = np.zeros((len(self.receiver_loc), self.nt), dtype=np.float32)

        # Compute elastic parameters
        self.mu = self.rho*self.vs**2
        self.lam = ((self.vp/self.vs)**2 - 2)*self.mu
        self.dt = 1 / self.fs
        ic("Elastic parameters computed")
        ic(self.dt)

        # Initialize wavefield arrays (velocity and stress)
        self.sxx = np.zeros((self.nx, self.nz), dtype=np.float32)
        self.sxz = np.zeros((self.nx, self.nz), dtype=np.float32)
        self.szz = np.zeros((self.nx, self.nz), dtype=np.float32)
        self.syx = np.zeros((self.nx, self.nz), dtype=np.float32)
        self.syz = np.zeros((self.nx, self.nz), dtype=np.float32)
        self.u = np.zeros((self.nx, self.nz), dtype=np.float32)
        self.v = np.zeros((self.nx, self.nz), dtype=np.float32)
        self.w = np.zeros((self.nx, self.nz), dtype=np.float32)

        # Apply optimized JIT functions for material property averaging
        ic("Computing harmonic averages for staggered grid")
        self.myx, self.myz = compute_shear_avg_SH(self.mu)
        self.mxz = compute_shear_avg_PSV(self.mu)
        self.rho_u, self.rho_w = compute_rho_staggered(self.rho)

        # Pre-compute reciprocals for optimal performance (eliminates divisions in inner loops)
        ic("Pre-computing reciprocals and dt/rho arrays")
        self.inv_dx = 1.0 / self.dx
        self.inv_dz = 1.0 / self.dz
        self.dt_rho_u, self.dt_rho_w, self.dt_rho = precompute_dt_over_rho(
            self.dt, self.rho_u, self.rho_w, self.rho
        )
        
        # Generate absorbing boundary coefficients
        ic("Computing absorbing boundary coefficients")
        ic(self.absorbing_frame)
        self.absorb_coeff = compute_absorbing_coeff(self.nx, self.nz, self.absorbing_frame)

        # Initialize source wavelets
        ic("Initializing source wavelets")
        self.wavelet_u = self.initialize_wavelet(self.wavelet_u)
        self.wavelet_v = self.initialize_wavelet(self.wavelet_v)
        self.wavelet_w = self.initialize_wavelet(self.wavelet_w)
        
        ic("Forward modeling initialization complete")
    
    def initialize_wavelet(self, wavelet, show=False):
        """
        Initialize source wavelet with automatic generation or validation.
        
        Args:
            wavelet: Input wavelet array or None for auto-generation
            show: Display wavelet plot if True
            
        Returns:
            wavelets: Initialized wavelet array (n_sources, nt)
        """
        if wavelet is None:
            ic("Generating Gaussian source wavelet")
            ic(self.f0, len(self.src_loc))
            wavelets = np.zeros((len(self.src_loc), self.nt), dtype = np.float32)
            for i, src in enumerate(self.src_loc):
                wavelets[i,:] = self._gaussian_src(self.f0)
                if show:
                    if self.plot_callback:
                        self.plot_callback('wavelet', wavelets[i,:], title='source wavelet')
                    elif self.enable_matplotlib:
                        plt = _ensure_matplotlib()
                        plt.figure(figsize=(5, 4))
                        plt.plot(wavelets[i,:])
                        plt.title('source wavelet')
                        plt.show()
        else:
            if wavelet.shape[0] != len(self.src_loc):
                raise ValueError(f'wavelet shape {wavelet.shape[0]} does not match src_loc {len(self.src_loc)}')
            if wavelet.shape[1] != self.nt:
                ic("Adjusting wavelet duration")
                ic(wavelet.shape[1], self.nt)
                if wavelet.shape[1] > self.nt:
                    wavelets = wavelet[:, :self.nt]  # Truncate
                else:
                    wavelets = np.pad(wavelet, ((0, 0), (0, self.nt - wavelet.shape[1])), mode='constant')  # Zero pad
            else:
                wavelets = wavelet
        return wavelets
    
    def plot_wavefield(self):
        if not self.enable_matplotlib:
            return
        
        plt = _ensure_matplotlib()
        # 波動場の初期プロットを設定
        u_cpu = np.asarray(self.u).T
        v_cpu = np.asarray(self.v).T
        w_cpu = np.asarray(self.w).T
         
        # 図と軸の設定
        self.fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 7))
        extent = [0.0, float(self.nx * self.dx), float(self.nz * self.dz), 0.0]

        # 初期イメージの作成
        self.im_u = ax1.imshow(u_cpu, cmap='seismic', extent=extent, animated=True, aspect='equal')    
        ax1.set_title('U Wavefield')
        ax1.set_xlabel('x [m]')
        ax1.set_ylabel('z [m]')

        self.im_v = ax2.imshow(v_cpu, cmap='seismic', extent=extent, animated=True, aspect='equal')
        ax2.set_title('V Wavefield')
        ax2.set_xlabel('x [m]')
        ax2.set_ylabel('z [m]')

        self.im_w = ax3.imshow(w_cpu, cmap='seismic', extent=extent, animated=True, aspect='equal')
        ax3.set_title('W Wavefield')
        ax3.set_xlabel('x [m]')
        ax3.set_ylabel('z [m]')

        # source, receiverの位置をプロット
        for l, loc in enumerate(self.receiver_loc):
            ax1.scatter(float(loc[0]*self.dz), float(loc[1]*self.dx), marker = '+', color = 'y')
            ax2.scatter(float(loc[0]*self.dz), float(loc[1]*self.dx), marker = '+', color = 'y')
            ax3.scatter(float(loc[0]*self.dz), float(loc[1]*self.dx), marker = '+', color = 'y')
        for k, loc in enumerate(self.src_loc):
            ax1.scatter(float(loc[0]*self.dz), float(loc[1]*self.dx), marker = '*', color = 'g')
            ax2.scatter(float(loc[0]*self.dz), float(loc[1]*self.dx), marker = '*', color = 'g')
            ax3.scatter(float(loc[0]*self.dz), float(loc[1]*self.dx), marker = '*', color = 'g')

        plt.tight_layout()
        plt.subplots_adjust(left=0.06, right=0.98, bottom=0.02, top = 0.92, hspace= 0.023, wspace= 0.12)
        plt.ion()
        plt.show(block=False)

    def display_wavefield(self):
        if not self.enable_matplotlib:
            return
        
        # 波動場データを更新
        u_cpu = self.u
        v_cpu = self.v
        w_cpu = self.w

        # Use callback if provided
        if self.plot_callback:
            self.plot_callback('wavefield', {'u': u_cpu, 'v': v_cpu, 'w': w_cpu})
            return

        # イメージのデータを更新
        self.im_u.set_data(u_cpu.T)
        self.im_v.set_data(v_cpu.T)
        self.im_w.set_data(w_cpu.T)

        # カラーバーの範囲を更新（必要に応じて）
        u_max = np.max(u_cpu) if np.max(u_cpu) > -np.min(u_cpu) else -np.min(u_cpu)
        v_max = np.max(v_cpu) if np.max(v_cpu) > -np.min(v_cpu) else -np.min(v_cpu)
        w_max = np.max(w_cpu) if np.max(w_cpu) > -np.min(w_cpu) else -np.min(w_cpu)
        self.im_u.set_clim(-u_max, u_max)
        self.im_v.set_clim(-v_max, v_max)
        self.im_w.set_clim(-w_max, w_max)

        # プロットを更新
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        #plt.pause(0.001)
    
    def update_vel(self, order):
        if order == 2:
            update_vel_order2_fw_jit(self.u, self.w, self.v, self.sxx, self.szz, self.sxz, 
                                     self.syx, self.syz, self.dx, self.dz, self.dt, 
                                     self.rho_u, self.rho_w, self.rho)
        elif order == 3:
            self._update_vel_order3()
        else:
            raise ValueError('order must be 2 or 3')
        apply_absorbing_coeff(self.u, self.v, self.w, self.absorb_coeff)

    def update_str(self, order):
        if order == 2:
            update_str_order2_fw_jit(self.u, self.w, self.v, self.sxx, self.szz, self.sxz, 
                                     self.syx, self.syz, self.dx, self.dz, self.dt, 
                                     self.lam, self.mu, self.mxz, self.myx, self.myz)
        elif order == 3:
            self._update_str_order3()
        else:
            raise ValueError('order must be 2 or 3')
        apply_absorbing_coeff_stress(self.sxx, self.szz, self.sxz, self.syx, self.syz, self.absorb_coeff)   

    def _update_vel_order2(self):
        # P-SV wave update:
        sxx_x = (self.sxx[1:-1, 1:-1] - self.sxx[0:-2, 1:-1]) / self.dx
        szz_z = (self.szz[1:-1, 1:-1] - self.szz[1:-1, 0:-2]) / self.dz
        sxz_x = (self.sxz[1:-1, 1:-1] - self.sxz[0:-2, 1:-1]) / self.dx
        sxz_z = (self.sxz[1:-1, 1:-1] - self.sxz[1:-1, 0:-2]) / self.dz
        du = (sxx_x + sxz_z) * (self.dt / self.rho_u[1:-1, 1:-1])
        dw = (sxz_x + szz_z) * (self.dt / self.rho_w[1:-1, 1:-1])
        self.u[1:-1, 1:-1] += du
        self.w[1:-1, 1:-1] += dw
        # SH wave update:   
        syx_x = (self.syx[1:-1, 1:-1] - self.syx[0:-2, 1:-1]) / self.dx
        syz_z = (self.syz[1:-1, 1:-1] - self.syz[1:-1, 0:-2]) / self.dz
        dv = (syx_x + syz_z) * (self.dt / self.rho[1:-1, 1:-1])
        self.v[1:-1, 1:-1] += dv   
    
    def _update_vel_order3(self):
        # インデックス範囲を設定
        i_start = 3
        i_end = self.v.shape[0] - 3  # nx - 3
        j_start = 3
        j_end = self.v.shape[1] - 3  # nz - 3

        # syx_x の計算
        syx_x = (
            (1/3)  * self.sx[i_start+1:i_end+1, j_start:j_end]
          - (2/3)  * self.sx[i_start:i_end, j_start:j_end]
          +  3     * self.sx[i_start-1:i_end-1, j_start:j_end]
          - (11/6) * self.sx[i_start-2:i_end-2, j_start:j_end]
        ) / self.dx

        # syz_z の計算
        syz_z = (
            (1/3) * self.sz[i_start:i_end, j_start+1:j_end+1]
            - (2/3) * self.sz[i_start:i_end, j_start:j_end]
            + 3 * self.sz[i_start:i_end, j_start-1:j_end-1]
            - (11/6) * self.sz[i_start:i_end, j_start-2:j_end-2]
        ) / self.dz

        # 速度場の更新
        dv = (syx_x + syz_z) * (self.dt / self.rho[i_start:i_end, j_start:j_end])
        self.v[i_start:i_end, j_start:j_end] += dv

    def _update_str_order2(self):
        # P-SV wave update:
        u_x = (self.u[2:, 1:-1] - self.u[1:-1, 1:-1]) / self.dx
        u_z = (self.u[1:-1, 2:] - self.u[1:-1, 1:-1]) / self.dz
        w_x = (self.w[2:, 1:-1] - self.w[1:-1, 1:-1]) / self.dx
        w_z = (self.w[1:-1, 2:] - self.w[1:-1, 1:-1]) / self.dz
        dsxx = self.dt*(self.lam[1:-1, 1:-1] * (u_x + w_z) + 2.0*self.mu[1:-1, 1:-1] * u_x)
        dszz = self.dt*(self.lam[1:-1, 1:-1] * (u_x + w_z) + 2.0*self.mu[1:-1, 1:-1] * w_z)
        dsxz = self.dt*(self.mxz[1:-1, 1:-1] * (u_z + w_x))
        self.sxx[1:-1, 1:-1] += dsxx
        self.szz[1:-1, 1:-1] += dszz
        self.sxz[1:-1, 1:-1] += dsxz

        # SH wave update:
        v_x = (self.v[2:, 1:-1] - self.v[1:-1, 1:-1]) / self.dx
        v_z = (self.v[1:-1, 2:] - self.v[1:-1, 1:-1]) / self.dz
        dsyx = self.dt * self.myx[1:-1, 1:-1] * v_x
        dsyz = self.dt * self.myz[1:-1, 1:-1] * v_z
        self.syx[1:-1, 1:-1] += dsyx
        self.syz[1:-1, 1:-1] += dsyz

    def _update_str_order3(self):
        # インデックス範囲を設定
        i_start = 3
        i_end = self.v.shape[0] - 3  # nx - 3
        j_start = 3
        j_end = self.v.shape[1] - 3  # nz - 3
        # v_x の計算
        v_x = (
            (1/3)   * self.v[i_start+1:i_end+1, j_start:j_end]
            - (2/3) * self.v[i_start:i_end, j_start:j_end]
            + 3     * self.v[i_start-1:i_end-1, j_start:j_end]
            - (11/6) * self.v[i_start-2:i_end-2, j_start:j_end]
        ) / self.dx
        # v_z の計算
        v_z = (
            (1/3) * self.v[i_start:i_end, j_start+1:j_end+1]
            - (2/3) * self.v[i_start:i_end, j_start:j_end]
            + 3 * self.v[i_start:i_end, j_start-1:j_end-1]
            - (11/6) * self.v[i_start:i_end, j_start-2:j_end-2]
        ) / self.dz

        # 応力場の更新
        self.sx[i_start:i_end, j_start:j_end] += self.dt * self.mux[i_start:i_end, j_start:j_end] * v_x
        self.sz[i_start:i_end, j_start:j_end] += self.dt * self.muz[i_start:i_end, j_start:j_end] * v_z

    def shear_avg_SH(self):
        mux = np.copy(self.mu)
        muz = np.copy(self.mu)
        # Use vectorized operations
        mu_i_j = self.mu[1:-1, 1:-1]
        mu_ip1_j = self.mu[2:, 1:-1]
        mu_i_jp1 = self.mu[1:-1, 2:]
        mux[1:-1, 1:-1] = 2 / (1 / mu_i_j + 1 / mu_ip1_j)
        muz[1:-1, 1:-1] = 2 / (1 / mu_i_j + 1 / mu_i_jp1)
        return mux, muz
    
    def shear_avg_PSV(self):
        muxz = np.copy(self.mu)
        mu_i_j = self.mu[1:-1, 1:-1]
        mu_ip1_j = self.mu[2:, 1:-1]
        mu_i_jp1 = self.mu[1:-1, 2:]
        mu_ip1_jp1 = self.mu[2:, 2:]
        muxz[1:-1,1:-1] = 4 / (1 / mu_i_j + 1 / mu_ip1_j + 1 / mu_i_jp1 + 1 / mu_ip1_jp1) 
        # for i in range(1, self.nx - 1):
        #     for j in range(1, self.nz - 1):
        #         muxz[i, j] = 4/(1/self.mu[i,j] + 1/self.mu[i+1,j] + 1/self.mu[i,j+1] + 1/self.mu[i+1,j+1])
        return muxz
        
    def rhou(self):
        """
        for i in range(1,self.nx-1):
            for j in range(1,self.nz-1):
                self.rho_u[i,j] = 0.5*(self.rho[i,j] + self.rho[i+1,j])        
        """
        rho_u = np.copy(self.rho)
        rho_i_j = self.rho[1:-1, 1:-1]
        rho_ip1_j = self.rho[2:, 1:-1]
        rho_u[1:-1, 1:-1] = 0.5 * (rho_i_j + rho_ip1_j)
        return rho_u

    def rhow(self):
        rho_w = np.copy(self.rho)
        rho_i_j = self.rho[1:-1, 1:-1]  
        rho_i_jp1 = self.rho[1:-1, 2:]
        rho_w[1:-1, 1:-1] = 0.5 * (rho_i_j + rho_i_jp1)
        # for i in range(1,self.nx-1):
        #     for j in range(1,self.nz-1):
        #         self.rho_w[i,j] = 0.5*(self.rho[i,j] + self.rho[i,j+1])
        return rho_w

    def absorb(self):
        """
        Define simple absorbing boundary frame based on wavefield damping
        according to Cerjan et al., 1985, Geophysics, 50, 705-708
        """
        FW = self.absorbing_frame # thickness of absorbing frame (gridpoints)
        a = 0.0053
        nx = self.nx
        nz = self.nz

        coeff = np.zeros(FW)

        # define coefficients in absorbing frame
        for i in range(FW):
            coeff[i] = np.exp(-(a**2 * (FW-i)**2))

        # initialize array of absorbing coefficients
        absorb_coeff = np.ones((nx,nz))

        # compute coefficients for left grid boundaries (x-direction)
        zb=0
        for i in range(FW):
            ze = nz - i - 1
            for j in range(zb,ze):
                absorb_coeff[i,j] = coeff[i]

        # compute coefficients for right grid boundaries (x-direction)
        zb=0
        for i in range(FW):
            ii = nx - i - 1
            ze = nz - i - 1
            for j in range(zb,ze):
                absorb_coeff[ii,j] = coeff[i]

        # compute coefficients for bottom grid boundaries (z-direction)
        xb=0
        for j in range(FW):
            jj = nz - j - 1
            xb = j
            xe = nx - j
            for i in range(xb,xe):
                absorb_coeff[i,jj] = coeff[j]
        return absorb_coeff

    def _gaussian_src(self, f0=10):
        time = np.linspace(0 * self.dt, self.nt * self.dt, self.nt)
        t0 = 3 / f0
        src = -2. * (time - t0) * (f0 ** 2) * (np.exp(- (f0 ** 2) * (time - t0) ** 2))
        return src
    
    def set_boundary_conditions(self):
        if self.receivers_height is None:
            # surface: free surface boundary condition Z=0

            self.syz[:, 0] = 0
            self.sxz[:, 0] = 0
            self.szz[:, 0] = 0
        else: # receivers_height is not None
            self.syz =  self.syz * self.surface_matrix
            self.sxz =  self.sxz * self.surface_matrix
            self.szz =  self.szz * self.surface_matrix

    def run(self, show=True, save=False):
        """
        run forward modeling
        retrun:
             0: simulation was cinducted safely,
             1: u faced infinite,
             2: v faced infinite,
             3: w faced infinite.
        show: bool, default True, if True, show wavefield
        save: bool, default False, if True, save wavefield             
        """
        if save: #allocate saved u,v,w, whose sizes are (nx,nz,nt//isnap)
            self.u_save = np.zeros((self.nx, self.nz, self.nt//self.isnap), dtype=np.float32)
            self.v_save = np.zeros((self.nx, self.nz, self.nt//self.isnap), dtype=np.float32)
            self.w_save = np.zeros((self.nx, self.nz, self.nt//self.isnap), dtype=np.float32)
            self.isnaps = np.zeros(self.nt//self.isnap, dtype=np.int32)
        
        #print('start forward modeling')
        self.initialize()
        if show:
            self.plot_wavefield()
        for it in range(self.nt):

            self.set_boundary_conditions()
            self.update_vel(order = self.order)
            # add source term at the source location
            for k, loc in enumerate(self.src_loc):
                i, j = loc
                self.u[i, j] += self.wavelet_u[k, it] * self.dt / self.rho_u[i, j] * self.dx * self.dz
                self.v[i, j] += self.wavelet_v[k, it] * self.dt / self.rho[i, j] * self.dx * self.dz
                self.w[i, j] += self.wavelet_w[k, it] * self.dt / self.rho_w[i, j] * self.dx * self.dz
            self.update_str(order = self.order)
            for l, loc in enumerate(self.receiver_loc):
                i, j = loc
                self.seismogram_u[l, it] = self.u[i, j]
                self.seismogram_v[l, it] = self.v[i, j]
                self.seismogram_w[l, it] = self.w[i, j]
    
            if it % self.isnap == 0:
                if show:
                    self.display_wavefield()
                # self.show(self.v, f'v :{it} step')
            
            if not np.all(np.isfinite(self.u)):
                return 1
            if not np.all(np.isfinite(self.v)):
                return 2                
            if not np.all(np.isfinite(self.w)):
                return 3
            
            # save wavefield if save
            if save and it % self.isnap == 0 and it != 0:
                self.u_save[:,:,it//self.isnap - 1] = self.u
                self.v_save[:,:,it//self.isnap - 1] = self.v
                self.w_save[:,:,it//self.isnap - 1] = self.w
                self.isnaps[it//self.isnap - 1] = it
        
        if show:
            self.show_seismogram(self.seismogram_u, 'u')
            self.show_seismogram(self.seismogram_v, 'v')
            self.show_seismogram(self.seismogram_w, 'w')

        print('end forward modeling')
        return 0

    def show_seismogram(self, seismogram_cpu, title = 'seismogram'):
        # Use callback if provided
        if self.plot_callback:
            self.plot_callback('seismogram', seismogram_cpu, title=title)
            return

        if not self.enable_matplotlib:
            return

        plt = _ensure_matplotlib()
        fig, axes = plt.subplots(len(seismogram_cpu),1, figsize=(5,4), sharey=False)
        for i in range(len(seismogram_cpu)):
            axes[i].plot(seismogram_cpu[i])
            axes[i].set_xticklabels([])
            axes[i].set_yticklabels([])

            if i == 0:
                axes[i].set_xlabel('time [s]')
            else:
                # axes[i].set_yticklabels([])
                pass
                #axes[i].set_title(label = 'graph',fontdict = {"fontsize":'x-small' })

        plt.suptitle(title)
        plt.subplots_adjust(left=0.06, right=0.98, bottom=0.02, top = 0.92, hspace= 0.023, wspace=0)
        #plt.ioff()
        plt.show()
        plt.pause(1.)
