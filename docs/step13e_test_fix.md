# Step 13E Test Fix

The CUDA-safe device resolver now returns:

- `0` when CUDA is available to PyTorch
- `cpu` when CUDA is not available to PyTorch

The previous test incorrectly expected `cuda:0` to always resolve to `0`. That is only true after CUDA-enabled PyTorch is installed.

Run:

```powershell
py -m pytest tests\test_detection.py tests\test_cuda_device.py
py -m pytest
```
