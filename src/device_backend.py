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
Device Backend Selection Module

Automatically detects and selects the best available compute device:
Priority: dGPU(NVIDIA) > dGPU(AMD) > iGPU(NVIDIA) > iGPU(AMD) > CPU
"""

import numpy as np
import warnings
from enum import Enum
from typing import Optional, Tuple

class DeviceType(Enum):
    """Enumeration of supported device types"""
    NVIDIA_DGPU = "NVIDIA_dGPU"
    AMD_DGPU = "AMD_dGPU"
    NVIDIA_IGPU = "NVIDIA_iGPU"
    AMD_IGPU = "AMD_iGPU"
    CPU = "CPU"

class DeviceBackend:
    """
    Device backend manager for automatic GPU/CPU selection.
    
    Detects available compute devices and selects the best one based on priority.
    Provides unified interface for both GPU and CPU execution.
    """
    
    def __init__(self, force_cpu: bool = False):
        """
        Initialize device backend.
        
        Args:
            force_cpu: If True, force CPU execution even if GPU is available
        """
        self.device_type = DeviceType.CPU
        self.device_name = "CPU"
        self.has_cuda = False
        self.has_rocm = False
        self.cuda = None
        self.cuda_available = False
        
        if not force_cpu:
            self._detect_devices()
        
        print(f"[Device Backend] Selected: {self.device_type.value} - {self.device_name}")
    
    def _detect_devices(self):
        """Detect available devices and select the best one according to priority"""
        
        # Try to detect NVIDIA GPUs (CUDA)
        nvidia_devices = self._detect_nvidia_devices()
        if nvidia_devices:
            self.device_type, self.device_name = nvidia_devices
            self.has_cuda = True
            return
        
        # Try to detect AMD GPUs (ROCm)
        amd_devices = self._detect_amd_devices()
        if amd_devices:
            self.device_type, self.device_name = amd_devices
            self.has_rocm = True
            return
        
        # Fall back to CPU
        self.device_type = DeviceType.CPU
        self.device_name = "CPU (Numba parallel)"
    
    def _detect_nvidia_devices(self) -> Optional[Tuple[DeviceType, str]]:
        """
        Detect NVIDIA GPUs using CUDA.
        Returns (DeviceType, device_name) or None if not available.
        """
        try:
            from numba import cuda
            
            if not cuda.is_available():
                return None
            
            # Get list of CUDA devices
            cuda.select_device(0)
            device = cuda.get_current_device()
            device_name = device.name.decode() if isinstance(device.name, bytes) else device.name
            
            # Try to determine if discrete or integrated
            # Heuristic: Integrated GPUs usually have less memory
            total_memory = device.total_memory
            memory_gb = total_memory / (1024**3)
            
            # If memory > 2GB, likely discrete GPU
            if memory_gb > 2.0:
                self.cuda = cuda
                self.cuda_available = True
                return (DeviceType.NVIDIA_DGPU, f"NVIDIA dGPU: {device_name} ({memory_gb:.1f}GB)")
            else:
                self.cuda = cuda
                self.cuda_available = True
                return (DeviceType.NVIDIA_IGPU, f"NVIDIA iGPU: {device_name} ({memory_gb:.1f}GB)")
        
        except ImportError:
            return None
        except Exception as e:
            warnings.warn(f"CUDA detection failed: {e}")
            return None
    
    def _detect_amd_devices(self) -> Optional[Tuple[DeviceType, str]]:
        """
        Detect AMD GPUs using ROCm.
        Returns (DeviceType, device_name) or None if not available.
        
        Note: ROCm support in Numba is limited. This is a placeholder for future support.
        """
        try:
            # ROCm support would go here
            # Currently Numba's ROCm support is experimental
            # For now, we'll return None
            return None
        except Exception:
            return None
    
    def is_gpu(self) -> bool:
        """Check if current backend is GPU"""
        return self.device_type != DeviceType.CPU
    
    def is_cuda(self) -> bool:
        """Check if current backend is CUDA"""
        return self.has_cuda and self.cuda_available
    
    def get_device_info(self) -> dict:
        """Get information about the selected device"""
        info = {
            'type': self.device_type.value,
            'name': self.device_name,
            'is_gpu': self.is_gpu(),
            'is_cuda': self.is_cuda(),
        }
        
        if self.is_cuda() and self.cuda:
            device = self.cuda.get_current_device()
            info['cuda_compute_capability'] = device.compute_capability
            info['total_memory_gb'] = device.total_memory / (1024**3)
            info['max_threads_per_block'] = device.MAX_THREADS_PER_BLOCK
        
        return info
    
    def to_device(self, array: np.ndarray) -> np.ndarray:
        """
        Transfer array to device memory if GPU, otherwise return as-is.
        
        Args:
            array: NumPy array
            
        Returns:
            Device array (CUDA array if GPU, NumPy array if CPU)
        """
        if self.is_cuda() and self.cuda:
            return self.cuda.to_device(array)
        return array
    
    def from_device(self, array) -> np.ndarray:
        """
        Transfer array from device memory to host if GPU, otherwise return as-is.
        
        Args:
            array: Device array or NumPy array
            
        Returns:
            NumPy array
        """
        if self.is_cuda() and self.cuda:
            if hasattr(array, 'copy_to_host'):
                return array.copy_to_host()
        return array

# Global backend instance
_backend = None

def get_backend(force_cpu: bool = False) -> DeviceBackend:
    """
    Get the global device backend instance.
    
    Args:
        force_cpu: Force CPU execution
        
    Returns:
        DeviceBackend instance
    """
    global _backend
    if _backend is None:
        _backend = DeviceBackend(force_cpu=force_cpu)
    return _backend

def reset_backend():
    """Reset the global backend instance"""
    global _backend
    _backend = None
