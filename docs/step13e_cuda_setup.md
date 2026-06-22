# Step 13E: CUDA Setup Diagnosis

## What happened

Your log shows:

```text
torch-2.12.1+cpu
torch.cuda.is_available(): False
torch.cuda.device_count(): 0
```

That means your current virtual environment has a CPU-only PyTorch build. The RTX 4080 Super is not being used by PyTorch/YOLO yet.

The code did the correct thing for `--device auto`: it selected CPU because PyTorch reported no CUDA devices.

When you forced `--device cuda:0`, Ultralytics failed because PyTorch still could not see CUDA.

## Check CUDA

Run:

```powershell
py scripts\check_cuda_setup.py
```

## Fix

Keep your project virtual environment activated, then install CUDA-enabled PyTorch using the official PyTorch selector:

```text
https://pytorch.org/get-started/locally/
```

Choose:

```text
OS: Windows
Package: Pip
Language: Python
Compute Platform: CUDA
```

Use the command shown by the selector.

## Typical workflow

```powershell
.\.venv\Scripts\activate

py -m pip uninstall -y torch torchvision torchaudio
```

Then run the command from the PyTorch selector.

After reinstalling, verify:

```powershell
py scripts\check_cuda_setup.py
```

You want:

```text
torch.cuda.is_available(): True
torch.cuda.device_count(): 1
```

Then run:

```powershell
py scripts\run_yolo_detection.py --input-dir data\sample_images --device auto
```

Expected:

```text
Device: auto -> 0
```

## Notes

- Installing the NVIDIA CUDA Toolkit separately is often not enough. The important part is that PyTorch itself must be a CUDA-enabled build.
- Install PyTorch first, then Ultralytics.
- If you use pip, do this inside the same `.venv` used by PyCharm/Streamlit.
