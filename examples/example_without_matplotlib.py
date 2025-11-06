"""
Example demonstrating optional matplotlib usage in OptiRTM

This example shows how to use the RTM library without matplotlib,
which is useful for:
- Headless servers without display
- Environments where matplotlib is not available
- Custom visualization pipelines
- Automated workflows where plotting is not needed
"""

import sys
sys.path.append('src/')
import ReverseTimeMigration as RT
import numpy as np
import os


def custom_plot_callback(plot_type, data, **kwargs):
    """
    Custom callback function for visualization
    
    This function receives plot data and can:
    - Save data to files for later visualization
    - Use alternative plotting libraries
    - Send data to remote visualization services
    - Simply log what would have been plotted
    
    Args:
        plot_type: Type of plot ('wavefield', 'result', 'height_array', etc.)
        data: The data to be visualized
        **kwargs: Additional parameters (title, save, dir, savename, etc.)
    """
    print(f"Plot callback received: {plot_type}")
    if 'title' in kwargs:
        print(f"  Title: {kwargs['title']}")
    
    # Example: Save data to numpy files instead of plotting
    if plot_type == 'result':
        print(f"  Result data contains: {list(data.keys())}")
        if kwargs.get('save', False):
            save_dir = kwargs.get('dir', 'custom_output/')
            savename = kwargs.get('savename', 'result')
            os.makedirs(save_dir, exist_ok=True)
            np.savez(f"{save_dir}{savename}_custom.npz", **data)
            print(f"  Saved to {save_dir}{savename}_custom.npz")
    elif plot_type == 'wavefield':
        print(f"  Wavefield data shapes: u={data['u'].shape}, v={data['v'].shape}, w={data['w'].shape}")
    elif plot_type == 'height_array':
        print(f"  Height array shape: {data.shape}")
    elif plot_type == 'surface_matrix':
        print(f"  Surface matrix shape: {data.shape}")


# Example 1: Disable matplotlib completely
print("=" * 60)
print("Example 1: Disable matplotlib completely")
print("=" * 60)

# Create synthetic test data
sampling_freq = 100000  # [Hz] sampling frequency
time_to = 0.001  # [s] time to observe (short for testing)
velocity = 120  # [m/s] velocity of the medium

# Create simple synthetic data
num_receivers = 5
num_samples = int(sampling_freq * time_to)
distance = np.linspace(1.0, 5.0, num_receivers)
source_x = 3.0

observed_u = np.random.randn(num_receivers, num_samples).astype(np.float32) * 0.01
observed_v = np.random.randn(num_receivers, num_samples).astype(np.float32) * 0.01
observed_w = np.random.randn(num_receivers, num_samples).astype(np.float32) * 0.01

# Source wavelets (just first 20 samples)
source_u = observed_u[2, :20]
source_v = observed_v[2, :20]
source_w = observed_w[2, :20]

receiver_loc = np.array(distance)
source_loc = np.array(source_x)
fs = np.float32(sampling_freq)

# Create RTM instance WITHOUT matplotlib
RTM_no_plot = RT.ReverseTimeMigration(
    observed_u=observed_u,
    observed_v=observed_v,
    observed_w=observed_w,
    source_u=source_u,
    source_v=source_v,
    source_w=source_w,
    receiver_loc=receiver_loc,
    source_loc=source_loc,
    fs=fs,
    vmin=80,
    vmax=300,
    vstep=100,
    v_fix=velocity,
    debug=False,
    enable_matplotlib=False  # Disable matplotlib
)

print("✓ RTM instance created without matplotlib")
print("✓ No matplotlib import occurred")


# Example 2: Use custom callback for visualization
print("\n" + "=" * 60)
print("Example 2: Use custom callback for visualization")
print("=" * 60)

RTM_callback = RT.ReverseTimeMigration(
    observed_u=observed_u,
    observed_v=observed_v,
    observed_w=observed_w,
    source_u=source_u,
    source_v=source_v,
    source_w=source_w,
    receiver_loc=receiver_loc,
    source_loc=source_loc,
    fs=fs,
    vmin=80,
    vmax=300,
    vstep=100,
    v_fix=velocity,
    debug=False,
    enable_matplotlib=False,
    plot_callback=custom_plot_callback  # Use custom callback
)

print("✓ RTM instance created with custom plot callback")
print("✓ All visualization will go through the callback function")


# Example 3: Default behavior (backward compatible - matplotlib enabled)
print("\n" + "=" * 60)
print("Example 3: Default behavior (backward compatible)")
print("=" * 60)

RTM_default = RT.ReverseTimeMigration(
    observed_u=observed_u,
    observed_v=observed_v,
    observed_w=observed_w,
    source_u=source_u,
    source_v=source_v,
    source_w=source_w,
    receiver_loc=receiver_loc,
    source_loc=source_loc,
    fs=fs,
    vmin=80,
    vmax=300,
    vstep=100,
    v_fix=velocity,
    debug=False
    # enable_matplotlib defaults to True for backward compatibility
)

print("✓ RTM instance created with default settings")
print("✓ Matplotlib will be loaded only when visualization methods are called")


print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("""
The OptiRTM library now supports three modes of operation:

1. enable_matplotlib=False: No matplotlib usage at all
   - Ideal for headless servers and automated pipelines
   - Minimal dependencies

2. plot_callback=<function>: Custom visualization
   - Route all visualization through a callback
   - Integrate with custom plotting libraries
   - Save data for later visualization

3. enable_matplotlib=True (default): Standard behavior
   - Backward compatible with existing code
   - Matplotlib loaded lazily only when needed
   - Full visualization capabilities

API Parameters:
- enable_matplotlib: bool (default=True)
- plot_callback: callable or None (default=None)

Example usage:
    rtm = ReverseTimeMigration(
        ...,
        enable_matplotlib=False,  # Disable matplotlib
        plot_callback=my_callback  # Optional custom callback
    )
""")
