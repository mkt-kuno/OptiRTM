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

# 1. backward modeling
import numpy as np
from numba import jit, prange
import matplotlib.pyplot as plt
import copy
from icecream import ic
# Import common RTM utilities
from rtm_utils import (
    compute_shear_avg_SH, compute_shear_avg_PSV, compute_rho_staggered,
    precompute_dt_over_rho, compute_absorbing_coeff, apply_absorbing_coeff,
    apply_absorbing_coeff_stress, check_array_finite, compute_cross_correlation
)
plt.style.use('fast')

# Configure icecream for better logging
ic.configureOutput(prefix='[Backward] ', includeContext=True)

# JIT-optimized functions for performance with aggressive optimizations
@jit(nopython=True, parallel=True, fastmath=True, cache=True, inline='always')
def update_vel_order2_jit(u, w, v, sxx, szz, sxz, syx, syz, inv_dx, inv_dz, dt, dt_rho_u, dt_rho_w, dt_rho):
    """
    Highly optimized velocity update for order 2 with:
    - Pre-computed reciprocals (inv_dx, inv_dz, dt/rho)
    - Fused operations to minimize memory access
    - Cache-friendly memory layout
    - Inlining hints for compiler
    """
    nx, nz = u.shape
    # P-SV and SH wave update fused for better cache utilization
    for i in prange(1, nx - 1):
        for j in range(1, nz - 1):
            # Pre-compute common terms for better instruction-level parallelism
            sxx_i1 = sxx[i+1, j]
            sxx_i = sxx[i, j]
            szz_j1 = szz[i, j+1]
            szz_j = szz[i, j]
            sxz_i1 = sxz[i+1, j]
            sxz_i = sxz[i, j]
            sxz_j1 = sxz[i, j+1]
            
            # P-SV wave: compute derivatives with pre-computed reciprocals
            sxx_x = (sxx_i1 - sxx_i) * inv_dx
            szz_z = (szz_j1 - szz_j) * inv_dz
            sxz_x = (sxz_i1 - sxz_i) * inv_dx
            sxz_z = (sxz_j1 - sxz_i) * inv_dz
            
            # Update velocity with fused multiply-add
            u[i, j] -= (sxx_x + sxz_z) * dt_rho_u[i, j]
            w[i, j] -= (sxz_x + szz_z) * dt_rho_w[i, j]
            
            # SH wave update: inline to avoid separate loop
            syx_x = (syx[i+1, j] - syx[i, j]) * inv_dx
            syz_z = (syz[i, j+1] - syz[i, j]) * inv_dz
            v[i, j] -= (syx_x + syz_z) * dt_rho[i, j]

@jit(nopython=True, parallel=True, fastmath=True, cache=True, inline='always')
def update_str_order2_jit(u, w, v, sxx, szz, sxz, syx, syz, inv_dx, inv_dz, neg_dt, lam, mu, mxz, myx, myz):
    """
    Highly optimized stress update for order 2 with:
    - Pre-computed reciprocals and negative dt
    - Fused computations reducing redundant calculations
    - Optimized memory access pattern
    """
    nx, nz = u.shape
    # Fused P-SV and SH wave update for better cache performance
    for i in prange(1, nx - 1):
        for j in range(1, nz - 1):
            # Load velocity values once for reuse
            u_ij = u[i, j]
            w_ij = w[i, j]
            v_ij = v[i, j]
            
            # Compute derivatives with pre-computed reciprocals
            u_x = (u_ij - u[i-1, j]) * inv_dx
            u_z = (u_ij - u[i, j-1]) * inv_dz
            w_x = (w_ij - w[i-1, j]) * inv_dx
            w_z = (w_ij - w[i, j-1]) * inv_dz
            
            # Compute common term once
            div_vel = u_x + w_z
            lam_div = lam[i, j] * div_vel
            
            # P-SV wave: fused computation with pre-negated dt
            sxx[i, j] += neg_dt * (lam_div + 2.0 * mu[i, j] * u_x)
            szz[i, j] += neg_dt * (lam_div + 2.0 * mu[i, j] * w_z)
            sxz[i, j] += neg_dt * mxz[i, j] * (u_z + w_x)
            
            # SH wave: inline for single loop
            v_x = (v_ij - v[i-1, j]) * inv_dx
            v_z = (v_ij - v[i, j-1]) * inv_dz
            syx[i, j] += neg_dt * myx[i, j] * v_x
            syz[i, j] += neg_dt * myz[i, j] * v_z

# Removed duplicate functions - now using shared utilities from rtm_utils.py:
# - apply_absorbing_coeff, apply_absorbing_coeff_stress
# - compute_absorbing_coeff
# - compute_shear_avg_SH, compute_shear_avg_PSV
# - compute_rho_staggered
# - precompute_dt_over_rho

class backward_modeling:
    """
    kwargs:
    observed data:np.array 観測データ
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
    receiver_loc:list 受信機の位置 [[i1,j1],[i2,j2],...]

    isnap:int 途中経過の表示ステップ数 default:10
    order:int 空間微分のオーダー(2 or 3) dedault:2
    ###
    snap:np.ndarray 途中経過の波場スナップショット for correlationg
    """

    def show(self, v:np.ndarray, suptitle):
        """
        v:np.ndarray 途中経過の波場スナップショット
        """
        mvmax = np.max(v)
        plt.figure(figsize=(8, 7))
        plt.imshow(v.T, aspect='equal', cmap='seismic', interpolation='nearest',vmin=-mvmax, vmax=mvmax)
        plt.colorbar()
        plt.title(suptitle)
        plt.show()

    def __init__(self, **kwargs):
        self.nx = kwargs['nx']
        self.nz = kwargs['nz']
        self.dx = kwargs['dx']
        self.dz = kwargs['dz']
        self.nt = kwargs['nt']
        self.fs = kwargs['fs']
        self.vs = kwargs['vs'] if 'vs' in kwargs else np.ones((self.nx,self.nz), dtype =np.float64)*200
        self.vp = kwargs['vp'] if 'vp' in kwargs else self.vs*np.sqrt(6) # at least root2 times larger than vs, poisson ratio = 0.25, vp/vs = 1.7320508, 
        self.rho= kwargs['rho']if 'rho'in kwargs else np.ones((self.nx,self.nz), dtype =np.float64)*1800
        self.absorbing_frame = kwargs['absorbing_frame'] if 'absorbing_frame' in kwargs else 60
        self.src_loc = kwargs['src_loc']if 'src_loc'in kwargs else [self.nx // 2,0] #source location, (i,j)
        self.obsdata_u = kwargs['observed_data_u']if 'observed_data_u'in kwargs else None # for x axis wave
        self.obsdata_v = kwargs['observed_data_v']if 'observed_data_v'in kwargs else None # for y axis wave
        self.obsdata_w = kwargs['observed_data_w']if 'observed_data_w'in kwargs else None # for z axis wave
        self.receiver_loc = kwargs['receiver_loc'] #receiver location, (i,j)
        self.isnap = kwargs['isnap']if 'isnap'in kwargs else 10
        self.order = kwargs['order']if 'order'in kwargs else 2
        self.receivers_height = kwargs['receivers_height'] if 'receivers_height' in kwargs else None
        self.surface_matrix = kwargs['surface_matrix'] if 'surface_matrix' in kwargs else None

        if self.surface_matrix is not None:
            if self.surface_matrix.shape != (self.nx, self.nz):
                raise ValueError('surface_matrix shape must be equal to (nx, nz)')

        self.steepness_array = kwargs['steepness_array'] if 'steepness_array' in kwargs else None 

    def initialize(self):
        """
        Initialize backward modeling with optimized pre-computations.
        
        This method:
        1. Initializes wavefield arrays (velocity and stress)
        2. Computes elastic parameters (mu, lambda) from velocity model
        3. Applies harmonic averaging for staggered grid
        4. Pre-computes reciprocals (inv_dx, inv_dz, dt/rho) for optimal performance
        5. Generates absorbing boundary coefficients
        
        All array operations use JIT-compiled functions with parallel=True and fastmath=True.
        """
        ic("Initializing backward modeling")
        ic(self.nx, self.nz, self.nt)
        
        self.synsrc_u = np.zeros((len(self.src_loc), self.nt), dtype=np.float64)
        self.synsrc_v = np.zeros((len(self.src_loc), self.nt), dtype=np.float64)
        self.synsrc_w = np.zeros((len(self.src_loc), self.nt), dtype=np.float64)

        # Compute elastic parameters
        self.mu = self.rho*self.vs**2
        self.lam = ((self.vp/self.vs)**2 - 2)*self.mu
        self.dt = 1 / self.fs
        ic("Elastic parameters computed")
        ic(self.dt)

        # Initialize wavefield arrays (velocity and stress)
        self.sxx = np.zeros((self.nx, self.nz), dtype=np.float64)
        self.sxz = np.zeros((self.nx, self.nz), dtype=np.float64)
        self.szz = np.zeros((self.nx, self.nz), dtype=np.float64)
        self.syx = np.zeros((self.nx, self.nz), dtype=np.float64)
        self.syz = np.zeros((self.nx, self.nz), dtype=np.float64)
        self.u = np.zeros((self.nx, self.nz), dtype=np.float64)
        self.v = np.zeros((self.nx, self.nz), dtype=np.float64)
        self.w = np.zeros((self.nx, self.nz), dtype=np.float64)

        # Apply optimized JIT functions for material property averaging
        ic("Computing harmonic averages for staggered grid")
        self.myx, self.myz = compute_shear_avg_SH(self.mu)
        self.mxz = compute_shear_avg_PSV(self.mu)
        self.rho_u, self.rho_w = compute_rho_staggered(self.rho)

        # Pre-compute reciprocals for optimal performance (eliminates divisions in inner loops)
        ic("Pre-computing reciprocals and dt/rho arrays")
        self.inv_dx = 1.0 / self.dx
        self.inv_dz = 1.0 / self.dz
        self.neg_dt = -self.dt  # Pre-negated for backward propagation
        self.dt_rho_u, self.dt_rho_w, self.dt_rho = precompute_dt_over_rho(
            self.dt, self.rho_u, self.rho_w, self.rho
        )
        
        # Generate absorbing boundary coefficients
        ic("Computing absorbing boundary coefficients")
        ic(self.absorbing_frame)
        self.absorb_coeff = compute_absorbing_coeff(self.nx, self.nz, self.absorbing_frame)
        
        ic("Backward modeling initialization complete")

        ## obsdata scaling
        gain = 16
        ms_V = 28.8
        maxV = 2.5
        self.maxAD = 2**23
        # self.obsdata_u = self.obsdata_u *(maxV / gain / ms_V) #maxAD too small so calculate finally
        # self.obsdata_v = self.obsdata_v *(maxV / gain / ms_V) #maxAD too small so calculate finally
        # self.obsdata_w = self.obsdata_w *(maxV / gain / ms_V) #maxAD too small so calculate finally

    def plot_wavefield(self):
        # 波動場の初期プロットを設定
        u_cpu = np.asarray(self.u).T
        v_cpu = np.asarray(self.v).T
        w_cpu = np.asarray(self.w).T

        # 図と軸の設定
        self.fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 7))
        extent = [0.0, float(self.nx * self.dx), float(self.nz * self.dz), 0.0]

        # 初期イメージの作成
        self.im_u = ax1.imshow(u_cpu, cmap='seismic', extent=extent, animated=True)
        ax1.set_title('U Wavefield')
        ax1.set_xlabel('x [m]')
        ax1.set_ylabel('z [m]')

        self.im_v = ax2.imshow(v_cpu, cmap='seismic', extent=extent, animated=True)
        ax2.set_title('V Wavefield')
        ax2.set_xlabel('x [m]')
        ax2.set_ylabel('z [m]')

        self.im_w = ax3.imshow(w_cpu, cmap='seismic', extent=extent, animated=True)
        ax3.set_title('W Wavefield')
        ax3.set_xlabel('x [m]')
        ax3.set_ylabel('z [m]')

        plt.tight_layout()
        plt.subplots_adjust(left=0.06, right=0.98, bottom=0.02, top = 0.92, hspace= 0.023, wspace= 0.12)
        plt.ion()
        plt.show(block=False)

    def display_wavefield(self, u_cpu = None, v_cpu = None, w_cpu = None, suptitle = 'Wavefield'):
        """
        display wavefield
        parameters
        u:np.array, default= self.u
        v:np.array, default= self.v
        w:np.array, default= self.w

        you can choose the wavefield to display setting u_cpu, v_cpu, w_cpu
            
        """
        # 波動場データを更新
        plt.suptitle(suptitle)
        u_cpu = self.u if u_cpu is None else u_cpu
        v_cpu = self.v if v_cpu is None else v_cpu
        w_cpu = self.w if w_cpu is None else w_cpu

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
            # Use optimized version with pre-computed reciprocals and dt/rho
            update_vel_order2_jit(self.u, self.w, self.v, self.sxx, self.szz, self.sxz, 
                                  self.syx, self.syz, self.inv_dx, self.inv_dz, self.dt, 
                                  self.dt_rho_u, self.dt_rho_w, self.dt_rho)
        elif order == 3:
            self._update_vel_order3()
        else:
            raise ValueError('order must be 2 or 3')
        apply_absorbing_coeff(self.u, self.v, self.w, self.absorb_coeff)

    def update_str(self, order):
        if order == 2:
            # Use optimized version with pre-computed reciprocals and negative dt
            update_str_order2_jit(self.u, self.w, self.v, self.sxx, self.szz, self.sxz, 
                                  self.syx, self.syz, self.inv_dx, self.inv_dz, self.neg_dt, 
                                  self.lam, self.mu, self.mxz, self.myx, self.myz)
        elif order == 3:
            self._update_str_order3()
        else:
            raise ValueError('order must be 2 or 3')
        apply_absorbing_coeff_stress(self.sxx, self.szz, self.sxz, self.syx, self.syz, self.absorb_coeff)   

    def _update_vel_order2(self):
        # P-SV wave update:
        sxx_x = (self.sxx[2:, 1:-1] - self.sxx[1:-1, 1:-1]) / self.dx
        szz_z = (self.szz[1:-1, 2:] - self.szz[1:-1, 1:-1]) / self.dz
        sxz_x = (self.sxz[2:, 1:-1] - self.sxz[1:-1, 1:-1]) / self.dx
        sxz_z = (self.sxz[1:-1, 2:] - self.sxz[1:-1, 1:-1]) / self.dz
        du = -(sxx_x + sxz_z) * (self.dt / self.rho_u[1:-1, 1:-1])
        dw = -(sxz_x + szz_z) * (self.dt / self.rho_w[1:-1, 1:-1])
        self.u[1:-1, 1:-1] += du
        self.w[1:-1, 1:-1] += dw
        # SH wave update:   
        syx_x = (self.syx[2:, 1:-1] - self.syx[1:-1, 1:-1]) / self.dx
        syz_z = (self.syz[1:-1, 2:] - self.syz[1:-1, 1:-1]) / self.dz
        dv = -(syx_x + syz_z) * (self.dt / self.rho[1:-1, 1:-1])
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
        u_x = (self.u[1:-1, 1:-1] - self.u[0:-2, 1:-1]) / self.dx
        u_z = (self.u[1:-1, 1:-1] - self.u[1:-1, 0:-2]) / self.dz
        w_x = (self.w[1:-1, 1:-1] - self.w[0:-2, 1:-1]) / self.dx
        w_z = (self.w[1:-1, 1:-1] - self.w[1:-1, 0:-2]) / self.dz
        dsxx = -self.dt*(self.lam[1:-1, 1:-1] * (u_x + w_z) + 2.0*self.mu[1:-1, 1:-1] * u_x)
        dszz = -self.dt*(self.lam[1:-1, 1:-1] * (u_x + w_z) + 2.0*self.mu[1:-1, 1:-1] * w_z)
        dsxz = -self.dt*(self.mxz[1:-1, 1:-1] * (u_z + w_x))
        self.sxx[1:-1, 1:-1] += dsxx
        self.szz[1:-1, 1:-1] += dszz
        self.sxz[1:-1, 1:-1] += dsxz

        # SH wave update:
        v_x = (self.v[1:-1, 1:-1] - self.v[0:-2, 1:-1]) / self.dx
        v_z = (self.v[1:-1, 1:-1] - self.v[1:-1, 0:-2]) / self.dz
        dsyx = -self.dt * self.myx[1:-1, 1:-1] * v_x
        dsyz = -self.dt * self.myz[1:-1, 1:-1] * v_z
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

    # Old methods removed - now using optimized JIT functions:
    # - compute_shear_avg_SH() replaces shear_avg_SH()
    # - compute_shear_avg_PSV() replaces shear_avg_PSV()
    # - compute_rho_staggered() replaces rhou() and rhow()
    # - compute_absorbing_coeff() replaces absorb()

    def set_boundary_condition(self):
        if self.receivers_height is None:
            # surface: free surface boundary condition Z=0
            self.syz[:, 0] = 0
            self.sxz[:, 0] = 0
            self.szz[:, 0] = 0
        else: # receivers_height is not None
            self.syz =  self.syz * self.surface_matrix
            self.sxz =  self.sxz * self.surface_matrix
            self.szz =  self.szz * self.surface_matrix
    
    def run(self, show=True):
        """
        run backward modeling
        retrun:
            0: simulation was conducted safely,
            4: u faced infinite,
            5: v faced infinite,
            6: w faced infinite.

        Show: bool, default=True
            show wavefield animatino or not                    
        """
        print('start backward modeling')
        self.initialize()
        if show:
            self.plot_wavefield()
        for it in range(self.nt):
            # free surface boundary condition Z=0
            self.set_boundary_condition()

            t = self.nt - it - 1# real timestep
            self.update_vel(order = self.order)
            # add source term at the source location
            for k, loc in enumerate(self.receiver_loc):
                i, j = loc
                self.u[i, j] += self.obsdata_u[k, t] * self.dt / self.rho_u[i, j] * self.dx * self.dz / self.maxAD
                self.v[i, j] += self.obsdata_v[k, t] * self.dt / self.rho[i, j] * self.dx * self.dz / self.maxAD
                self.w[i, j] += self.obsdata_w[k, t] * self.dt / self.rho_w[i, j] * self.dx * self.dz / self.maxAD
            self.update_str(order = self.order)
            for l, loc in enumerate(self.src_loc):
                i, j = loc
                self.synsrc_u[l, t] = self.u[i, j]
                self.synsrc_v[l, t] = self.v[i, j]
                self.synsrc_w[l, t] = self.w[i, j]

            if it % self.isnap == 0 :
                print(f'i={it}/{self.nt}')
                #self.show(self.v, f'v, it={it}')
                if show:
                    self.display_wavefield()
            
            # check if the wavefield is infinite: flag
            if not np.all(np.isfinite(self.u)):
                return 4
            if not np.all(np.isfinite(self.v)):
                return 5
            if not np.all(np.isfinite(self.w)):
                return 6

        print('end backward modeling')
        return 0
    
    def run_calc(self, import_fwdata_u:np.ndarray, import_fwdata_v:np.ndarray, import_fwdata_w:np.ndarray, isnaps:np.ndarray, show=True, method = 'closs_correlation', save = False):
        """
        run backward modeling and calcurate correlation with synthetic forward data
        parameters:
            impout_fwdata_u:np.array,
                forward modeling data u
            impout_fwdata_v:np.array,
                forward modeling data v
            impout_fwdata_w:np.array,
                forward modeling data w
            isnaps:np.array,
                snapshot timesteps of forward modeling u,v,w
            method:str, default='closs_correlation',
                closs_correlation' or 'convolution' only (20241002)
            Show: bool, default=True
                show wavefield animatino or not
            save: bool, default=False
                save correlation data or not

        retrun:
            0: simulation was conducted safely,
            4: u faced infinite,
            5: v faced infinite,
            6: w faced infinite.

        if simulation was conducted safely,
        result_u, result_v, result_w:np.array
            correlation data of u,v,w with synthetic forward data  
        """
        # print('start backward modeling')
        self.initialize()
        if show:
            self.plot_wavefield()
        
        ## make reslt image:
        result_u = np.zeros((self.nx, self.nz), dtype=np.float64)
        result_v = np.zeros((self.nx, self.nz), dtype=np.float64)
        result_w = np.zeros((self.nx, self.nz), dtype=np.float64)

        for it in range(self.nt):
            # free surface boundary condition Z=0
            self.set_boundary_condition()

            self.update_vel(order = self.order)

            # add source term at the source location
            # Extract indices from src_loc
            receiver_loc_array = np.array(self.receiver_loc)
            i_indices = receiver_loc_array[:, 0]
            j_indices = receiver_loc_array[:, 1]

            t = (self.nt - 1) - it # real timestep
            delta_u = self.obsdata_u[:, t] #* self.dt / self.rho_u[i_indices, j_indices] * self.dx * self.dz #/ self.maxAD
            delta_v = self.obsdata_v[:, t] #* self.dt / self.rho[i_indices, j_indices] * self.dx * self.dz #/ self.maxAD
            delta_w = self.obsdata_w[:, t] #* self.dt / self.rho_w[i_indices, j_indices] * self.dx * self.dz #/ self.maxAD

            self.u[i_indices, j_indices] += delta_u
            self.v[i_indices, j_indices] += delta_v
            self.w[i_indices, j_indices] += delta_w

            self.update_str(order=self.order)

            src_loc_array = np.array(self.src_loc)
            i_indices = src_loc_array[:, 0]
            j_indices = src_loc_array[:, 1]

            self.synsrc_u[:, t] = self.u[i_indices, j_indices]
            self.synsrc_v[:, t] = self.v[i_indices, j_indices]
            self.synsrc_w[:, t] = self.w[i_indices, j_indices]
            
            snapt = t
            if snapt in isnaps:
                indices = np.where(isnaps == snapt)[0][0]
                if method == 'closs_correlation':
                    delta_result_u = (import_fwdata_u[:, :, indices] * self.u[:, :]).squeeze()
                    delta_result_v = (import_fwdata_v[:, :, indices] * self.v[:, :]).squeeze()
                    delta_result_w = (import_fwdata_w[:, :, indices] * self.w[:, :]).squeeze()
                elif method == 'convolution': 
                    delta_result_u = (import_fwdata_u[:, :, indices] * self.u[:, :]).squeeze() / (import_fwdata_u[:, :, indices]**2).squeeze()
                    delta_result_v = (import_fwdata_v[:, :, indices] * self.v[:, :]).squeeze() / (import_fwdata_v[:, :, indices]**2).squeeze()
                    delta_result_w = (import_fwdata_w[:, :, indices] * self.w[:, :]).squeeze() / (import_fwdata_w[:, :, indices]**2).squeeze()                       
                else:
                    raise ValueError('method must be closs_correlation or convolution, now method is', method)
                
                result_u += delta_result_u
                result_v += delta_result_v
                result_w += delta_result_w
        
                if show:
                    sm_u = import_fwdata_u[:, :, indices].squeeze() * self.u[:, :]
                    sm_v = import_fwdata_v[:, :, indices].squeeze() * self.v[:, :]
                    sm_w = import_fwdata_w[:, :, indices].squeeze() * self.w[:, :]

                    self.display_wavefield(u_cpu = sm_u, v_cpu = sm_v, w_cpu = sm_w, suptitle=f'fw x bw at {snapt}timestep')
                    #self.display_wavefield(u_cpu=self.u, v_cpu= self.v, w_cpu=self.w, suptitle=f'fw x bw at {snapt}timestep')
                    # self.display_wavefield(u_cpu=delta_result_u, v_cpu=delta_result_v, w_cpu=delta_result_w, suptitle=f'fw x bw at {snapt}timestep')

            # check if the wavefield is infinite: flag
            if not np.all(np.isfinite(self.u)):
                return 4
            if not np.all(np.isfinite(self.v)):
                return 5
            if not np.all(np.isfinite(self.w)):
                return 6

        print('end backward modeling')
        self.result_u = result_u
        self.result_v = result_v
        self.result_w = result_w
        return 0

    def show_src(self):

        plt.figure(figsize=(8, 7))
        plt.plot(self.synsrc_u[0, :], label='u')
        plt.plot(self.synsrc_v[0, :], label='v')
        plt.plot(self.synsrc_w[0, :], label='w')
        plt.legend()
        plt.show()
