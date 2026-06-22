r"""Check whether PyTorch can see CUDA.

Run from the repository root:

    py scripts\check_cuda_setup.py
"""

from __future__ import annotations

import platform
import subprocess
import sys


def main() -> int:
    """Print a concise CUDA/PyTorch diagnostic report."""
    print("CUDA / PyTorch diagnostic")
    print("=" * 32)
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")

    try:
        import torch
    except ImportError:
        print("PyTorch: NOT INSTALLED")
        print()
        print("Install PyTorch from the official selector:")
        print("  https://pytorch.org/get-started/locally/")
        return 1

    print(f"PyTorch: {torch.__version__}")
    print(f"torch.version.cuda: {torch.version.cuda}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")

    if torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        print(f"Current CUDA device index: {device_index}")
        print(f"CUDA device name: {torch.cuda.get_device_name(device_index)}")
        return 0

    print()
    print("CUDA is not available to PyTorch in this environment.")
    print("This usually means the installed torch package is CPU-only,")
    print("even if your PC has an NVIDIA GPU.")
    print()
    print("Your next step:")
    print("  1. Keep this virtual environment activated.")
    print("  2. Uninstall CPU-only torch packages.")
    print("  3. Reinstall CUDA-enabled PyTorch using the official PyTorch selector.")
    print()
    print("Useful checks:")
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            print("nvidia-smi: available")
            first_line = result.stdout.splitlines()[0] if result.stdout.splitlines() else ""
            print(first_line)
        else:
            print("nvidia-smi: not available or failed")
    except Exception:
        print("nvidia-smi: not available")

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
