# Implementation Summary: Optional Matplotlib Usage

## Issue
The original issue (in Japanese) requested:
- Title: "Restrict the use of matplotlib.pyplot"
- Description: "Make it possible to choose via API whether to use callback method or import, and eliminate the state where import is always necessary"

## Solution Implemented

### 1. Lazy Loading of Matplotlib
- Matplotlib is no longer imported at module load time
- New `_ensure_matplotlib()` function imports matplotlib only when needed
- This reduces dependency requirements for headless environments

### 2. New API Parameters
Added to all visualization classes:
- **`enable_matplotlib`** (bool, default=True): Control whether matplotlib can be used
- **`plot_callback`** (callable, default=None): Optional callback for custom visualization

### 3. Modified Classes
- **ReverseTimeMigration** (`src/ReverseTimeMigration.py`)
- **forward_modeling** (`src/SH_PSV_forward_modelng.py`)
- **backward_modeling** (`src/SH_PSV_backward_modeling.py`)

### 4. Three Usage Modes

#### Mode 1: Disable Matplotlib
```python
rtm = ReverseTimeMigration(..., enable_matplotlib=False)
```
- No matplotlib dependency required
- Ideal for headless servers

#### Mode 2: Custom Callback
```python
def my_callback(plot_type, data, **kwargs):
    # Custom visualization logic
    pass

rtm = ReverseTimeMigration(..., enable_matplotlib=False, plot_callback=my_callback)
```
- Use alternative plotting libraries
- Custom data processing

#### Mode 3: Default Behavior (Backward Compatible)
```python
rtm = ReverseTimeMigration(...)  # enable_matplotlib=True by default
```
- Existing code works unchanged
- Matplotlib loaded lazily when visualization is called

## Files Changed

### Core Implementation
1. `src/ReverseTimeMigration.py` - Added lazy loading and API parameters
2. `src/SH_PSV_forward_modelng.py` - Added lazy loading and API parameters
3. `src/SH_PSV_backward_modeling.py` - Added lazy loading and API parameters

### Documentation
1. `MATPLOTLIB_USAGE.md` - Comprehensive usage guide (NEW)
2. `README.md` - Updated to mention optional matplotlib
3. `examples/example_without_matplotlib.py` - Demonstrates all modes (NEW)

### Testing
1. `test_matplotlib_optional.py` - Comprehensive test suite (NEW)

## Test Results

All tests pass successfully:
- ✓ Modules import without matplotlib
- ✓ Instances created with `enable_matplotlib=False`
- ✓ Custom callbacks work correctly
- ✓ Backward compatibility maintained
- ✓ Lazy loading verified (matplotlib not loaded until needed)
- ✓ No security vulnerabilities (CodeQL scan: 0 alerts)

## Backward Compatibility

**100% Backward Compatible**
- Default behavior unchanged (`enable_matplotlib=True`)
- Existing code works without modifications
- No breaking changes

## Benefits

1. **Flexibility**: Choose visualization method that fits your workflow
2. **Reduced Dependencies**: Run without matplotlib when not needed
3. **Performance**: Avoid unnecessary imports
4. **Portability**: Works on headless systems
5. **Integration**: Easy integration with custom pipelines
6. **Compatibility**: Existing code continues to work

## Migration Path

### For Users Who Don't Need Visualization
```python
# Add one parameter
rtm = ReverseTimeMigration(..., enable_matplotlib=False)
```

### For Users With Custom Visualization
```python
# Add callback
rtm = ReverseTimeMigration(..., plot_callback=my_callback)
```

### For Existing Users
```python
# No changes needed - works as before
rtm = ReverseTimeMigration(...)
```

## Implementation Quality

- **Code Review**: Addressed all review comments
- **Security Scan**: 0 vulnerabilities found
- **Testing**: Comprehensive test coverage
- **Documentation**: Detailed usage guide provided
- **Examples**: Working examples for all modes

## Metrics

- Lines of code changed: ~150
- New files created: 3 (doc, example, test)
- Test coverage: All core functionality
- Security issues: 0
- Breaking changes: 0
