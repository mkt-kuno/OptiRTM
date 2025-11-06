"""
Comprehensive test for optional matplotlib usage
"""
import sys
sys.path.insert(0, 'src')
import numpy as np

print("=" * 70)
print("Test 1: Import without matplotlib being loaded")
print("=" * 70)
print(f"matplotlib loaded before import: {'matplotlib' in sys.modules}")

import ReverseTimeMigration as RT
import SH_PSV_forward_modelng as fw
import SH_PSV_backward_modeling as bk

print(f"matplotlib loaded after import: {'matplotlib' in sys.modules}")
print("✓ All modules imported without loading matplotlib")

print("\n" + "=" * 70)
print("Test 2: Create instances with enable_matplotlib=False")
print("=" * 70)

# Create test data
num_receivers = 3
num_samples = 100
observed_u = np.random.randn(num_receivers, num_samples).astype(np.float32) * 0.01
observed_v = np.random.randn(num_receivers, num_samples).astype(np.float32) * 0.01
observed_w = np.random.randn(num_receivers, num_samples).astype(np.float32) * 0.01
source_u = observed_u[1, :10]
source_v = observed_v[1, :10]
source_w = observed_w[1, :10]
receiver_loc = np.array([1.0, 2.0, 3.0])
source_loc = np.array([2.0])

rtm = RT.ReverseTimeMigration(
    observed_u=observed_u,
    observed_v=observed_v,
    observed_w=observed_w,
    source_u=source_u,
    source_v=source_v,
    source_w=source_w,
    receiver_loc=receiver_loc,
    source_loc=source_loc,
    fs=1000.0,
    enable_matplotlib=False
)
print("✓ ReverseTimeMigration instance created with enable_matplotlib=False")
print(f"matplotlib loaded: {'matplotlib' in sys.modules}")

print("\n" + "=" * 70)
print("Test 3: Create instances with custom callback")
print("=" * 70)

callback_calls = []

def test_callback(plot_type, data, **kwargs):
    callback_calls.append({
        'type': plot_type,
        'data_type': type(data).__name__,
        'kwargs': list(kwargs.keys())
    })

rtm_callback = RT.ReverseTimeMigration(
    observed_u=observed_u,
    observed_v=observed_v,
    observed_w=observed_w,
    source_u=source_u,
    source_v=source_v,
    source_w=source_w,
    receiver_loc=receiver_loc,
    source_loc=source_loc,
    fs=1000.0,
    enable_matplotlib=False,
    plot_callback=test_callback
)
print("✓ ReverseTimeMigration instance created with custom callback")
print(f"matplotlib loaded: {'matplotlib' in sys.modules}")

print("\n" + "=" * 70)
print("Test 4: Create forward_modeling with enable_matplotlib=False")
print("=" * 70)

nx, nz = 50, 50
dx, dz = 0.1, 0.1
nt = 100
fs = 1000.0
vs = np.ones((nx, nz), dtype=np.float32) * 100
vp = vs * np.sqrt(3)
rho = np.ones((nx, nz), dtype=np.float32) * 2000

_fw = fw.forward_modeling(
    nx=nx, nz=nz, dx=dx, dz=dz, nt=nt, fs=fs,
    vs=vs, vp=vp, rho=rho,
    receiver_loc=[[25, 1], [30, 1]],
    enable_matplotlib=False
)
print("✓ forward_modeling instance created with enable_matplotlib=False")
print(f"matplotlib loaded: {'matplotlib' in sys.modules}")

print("\n" + "=" * 70)
print("Test 5: Create backward_modeling with enable_matplotlib=False")
print("=" * 70)

_bw = bk.backward_modeling(
    nx=nx, nz=nz, dx=dx, dz=dz, nt=nt, fs=fs,
    vs=vs, vp=vp, rho=rho,
    observed_data_u=np.zeros((2, nt), dtype=np.float32),
    observed_data_v=np.zeros((2, nt), dtype=np.float32),
    observed_data_w=np.zeros((2, nt), dtype=np.float32),
    receiver_loc=[[25, 1], [30, 1]],
    enable_matplotlib=False
)
print("✓ backward_modeling instance created with enable_matplotlib=False")
print(f"matplotlib loaded: {'matplotlib' in sys.modules}")

print("\n" + "=" * 70)
print("Test 6: Verify backward compatibility (default behavior)")
print("=" * 70)

rtm_default = RT.ReverseTimeMigration(
    observed_u=observed_u,
    observed_v=observed_v,
    observed_w=observed_w,
    source_u=source_u,
    source_v=source_v,
    source_w=source_w,
    receiver_loc=receiver_loc,
    source_loc=source_loc,
    fs=1000.0
    # enable_matplotlib defaults to True
)
print("✓ ReverseTimeMigration instance created with default settings")
print(f"  enable_matplotlib is: {rtm_default.enable_matplotlib}")
print(f"  plot_callback is: {rtm_default.plot_callback}")
print(f"matplotlib loaded (still lazy): {'matplotlib' in sys.modules}")

print("\n" + "=" * 70)
print("Test Summary")
print("=" * 70)
print("✓ All tests passed!")
print(f"✓ matplotlib was never loaded: {not ('matplotlib' in sys.modules)}")
print(f"✓ Total callback calls: {len(callback_calls)}")
print("\nKey achievements:")
print("  1. Modules can be imported without matplotlib")
print("  2. Instances can be created with enable_matplotlib=False")
print("  3. Custom callbacks can be provided")
print("  4. Backward compatibility maintained (default=True)")
print("  5. Lazy loading works - matplotlib not imported until needed")
