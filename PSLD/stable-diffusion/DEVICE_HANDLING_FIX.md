# Device Handling Fix for MacBook (MPS, CUDA, CPU)

## Problem Summary

The original PSLD code was hardcoded to use CUDA devices, which caused issues on MacBook systems that use MPS (Metal Performance Shaders) or CPU. The code would fail when trying to run on non-CUDA devices.

## Fixes Applied

### 1. Added Smart Device Detection (`ldm/models/diffusion/psld.py`)

**Added `get_device()` function:**
```python
def get_device():
    """Get the best available device for MacBook (MPS, CUDA, or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
```

**Updated DDIMSampler class:**
- Added `self.device = get_device()` in `__init__`
- Updated `register_buffer()` to use `self.device` instead of hardcoded CUDA
- Updated `make_schedule()` to use `self.device` instead of `self.model.device`
- Updated `ddim_sampling()` to use `self.device` instead of `self.model.betas.device`

### 2. Fixed Style Constraint Detection

**Added proper style operator detection in both inpainting and general_inverse sections:**
```python
# Check if this is a style operator
if hasattr(operator, '__class__') and ('style' in operator.__class__.__name__.lower() or 'StyleOperator' in operator.__class__.__name__):
    # Style constraint logic
else:
    # Regular operator logic
```

### 3. Device Compatibility

The style operator in `guided_diffusion/measurements.py` already had proper device handling:
```python
def __init__(self, device=None):
    if device is None:
        self.device = torch.device("cuda" if torch.cuda.is_available()
        else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        else "cpu")
    else:
        self.device = device
```

## Device Priority Order

The system now automatically selects the best available device in this order:

1. **CUDA** - If NVIDIA GPU is available
2. **MPS** - If Apple Silicon GPU is available (MacBook M1/M2/M3)
3. **CPU** - As fallback

## Testing Results

All device handling tests pass successfully:

```
Test Results: 3/3 tests passed
✓ All tests passed! Device handling should work correctly.
```

- ✓ Device detection works correctly
- ✓ Style operator works on MPS device
- ✓ PSLD device handling works properly

## Usage

The device handling is now automatic. No changes needed in your code:

```python
# The sampler will automatically use the best available device
sampler = DDIMSampler(model)

# All tensors will be created on the correct device
# Style operators will work on any device (CUDA, MPS, CPU)
```

## Benefits

1. **MacBook Compatibility**: Works seamlessly on Apple Silicon MacBooks
2. **Automatic Fallback**: Gracefully falls back to CPU if no GPU is available
3. **CUDA Support**: Still works optimally on NVIDIA GPUs
4. **No Code Changes**: Existing code works without modification

## Troubleshooting

If you encounter device-related issues:

1. **Check device detection**: The system will print which device it's using
2. **MPS limitations**: Some operations might be slower on MPS compared to CUDA
3. **Memory issues**: MPS has different memory management than CUDA
4. **PyTorch version**: Ensure you have PyTorch 1.12+ for MPS support

## Performance Notes

- **CUDA**: Best performance for most operations
- **MPS**: Good performance on Apple Silicon, some operations may be slower
- **CPU**: Slowest but most compatible

The style constraints should now work properly on all device types!
