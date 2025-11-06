# Matplotlib Usage Control

OptiRTM now provides flexible control over matplotlib usage, allowing you to choose whether to use matplotlib for visualization or provide your own custom visualization solution.

## Overview

The matplotlib library is now **optional** and only imported when needed. This provides several benefits:

- **Reduced dependencies** for headless environments
- **Custom visualization** through callback functions
- **Backward compatibility** - existing code continues to work
- **Lazy loading** - matplotlib only loaded when actually used

## Usage Modes

### 1. Disable Matplotlib Completely

Ideal for headless servers, automated pipelines, or when you don't need visualization:

```python
import ReverseTimeMigration as RT

rtm = RT.ReverseTimeMigration(
    observed_u=observed_u,
    observed_v=observed_v,
    observed_w=observed_w,
    source_u=source_u,
    source_v=source_v,
    source_w=source_w,
    receiver_loc=receiver_loc,
    source_loc=source_loc,
    fs=fs,
    enable_matplotlib=False  # Disable matplotlib
)
```

### 2. Custom Visualization Callback

Provide your own visualization function to handle plotting:

```python
def my_plot_callback(plot_type, data, **kwargs):
    """
    Custom visualization callback
    
    Args:
        plot_type: Type of plot ('wavefield', 'result', 'height_array', etc.)
        data: The data to be visualized
        **kwargs: Additional parameters (title, save, dir, savename, etc.)
    """
    if plot_type == 'result':
        # Use your preferred plotting library
        import plotly.graph_objects as go
        # ... custom visualization code ...
    elif plot_type == 'wavefield':
        # Save data for later analysis
        np.save(f'wavefield_{kwargs.get("title", "data")}.npy', data)

rtm = RT.ReverseTimeMigration(
    observed_u=observed_u,
    observed_v=observed_v,
    observed_w=observed_w,
    source_u=source_u,
    source_v=source_v,
    source_w=source_w,
    receiver_loc=receiver_loc,
    source_loc=source_loc,
    fs=fs,
    enable_matplotlib=False,
    plot_callback=my_plot_callback  # Use custom callback
)
```

### 3. Default Behavior (Backward Compatible)

Existing code continues to work without modifications:

```python
rtm = RT.ReverseTimeMigration(
    observed_u=observed_u,
    observed_v=observed_v,
    observed_w=observed_w,
    source_u=source_u,
    source_v=source_v,
    source_w=source_w,
    receiver_loc=receiver_loc,
    source_loc=source_loc,
    fs=fs
    # enable_matplotlib defaults to True
)

# Matplotlib is loaded lazily only when visualization methods are called
rtm.run()
rtm.show_result()  # matplotlib imported here when first needed
```

## API Parameters

### ReverseTimeMigration Class

- **`enable_matplotlib`** (bool, default=True): 
  - Controls whether matplotlib can be used for visualization
  - Set to `False` to disable matplotlib completely
  
- **`plot_callback`** (callable or None, default=None):
  - Optional callback function for custom visualization
  - Receives `plot_type`, `data`, and additional keyword arguments
  - If provided, takes precedence over matplotlib visualization

### forward_modeling Class

The same parameters are available:

```python
import SH_PSV_forward_modelng as fw

_fw = fw.forward_modeling(
    nx=nx, nz=nz, dx=dx, dz=dz,
    # ... other parameters ...
    enable_matplotlib=False,  # Disable matplotlib
    plot_callback=my_callback  # Optional custom callback
)
```

### backward_modeling Class

The same parameters are available:

```python
import SH_PSV_backward_modeling as bk

_bw = bk.backward_modeling(
    nx=nx, nz=nz, dx=dx, dz=dz,
    # ... other parameters ...
    enable_matplotlib=False,  # Disable matplotlib
    plot_callback=my_callback  # Optional custom callback
)
```

## Plot Callback Interface

The callback function receives the following arguments:

```python
def plot_callback(plot_type: str, data: Any, **kwargs) -> None:
    """
    Custom plot callback interface
    
    Args:
        plot_type: Type of visualization requested
            - 'wavefield': Wavefield data during simulation
            - 'result': Final RTM result
            - 'seismogram': Seismogram data
            - 'height_array': Surface height array
            - 'surface_matrix': Surface matrix
            - 'wavelet': Source wavelet
            - 'wavefield_snapshot': Backward modeling snapshot
            - 'source': Synthetic source
            
        data: Visualization data (format depends on plot_type)
            - dict for 'wavefield' and 'result': {'u': array, 'v': array, 'w': array}
            - numpy array for other types
            
        **kwargs: Additional parameters
            - title: Plot title
            - save: Whether to save (for 'result')
            - dir: Save directory (for 'result')
            - savename: Save filename (for 'result')
            - extent: Axis extent (for 'result')
            - cmap: Colormap (for 'result')
    """
    pass
```

## Examples

See the following example files:

- **`examples/example_without_matplotlib.py`**: Demonstrates all three usage modes
- **`examples/example.py`**: Original example (backward compatible)

## Migration Guide

Existing code works without changes. To opt into the new features:

### For Headless Environments

Simply add `enable_matplotlib=False`:

```python
# Before (still works)
rtm = RT.ReverseTimeMigration(...)

# After (no matplotlib dependency)
rtm = RT.ReverseTimeMigration(..., enable_matplotlib=False)
```

### For Custom Visualization

Add a callback function:

```python
def my_callback(plot_type, data, **kwargs):
    # Your custom visualization logic
    pass

rtm = RT.ReverseTimeMigration(
    ...,
    enable_matplotlib=False,
    plot_callback=my_callback
)
```

## Requirements

When `enable_matplotlib=False`, matplotlib is **not required** to be installed. This reduces the dependency footprint for environments that don't need visualization.

Core dependencies (always required):
- numpy >= 1.26.4
- numba >= 0.59.0
- icecream >= 2.1.3

Optional dependencies:
- matplotlib >= 3.9.2 (only if `enable_matplotlib=True`)

## Implementation Details

- Matplotlib is imported using lazy loading (`_ensure_matplotlib()` function)
- Import only occurs when visualization methods are actually called
- `enable_matplotlib=True` is the default for backward compatibility
- Callback functions take precedence over matplotlib when both are available
- All visualization code checks the flags before attempting to plot

## Benefits

1. **Flexibility**: Choose the visualization method that fits your workflow
2. **Performance**: Avoid unnecessary imports in production environments
3. **Portability**: Run on systems without display or matplotlib
4. **Integration**: Easy integration with custom visualization pipelines
5. **Compatibility**: Existing code continues to work unchanged
