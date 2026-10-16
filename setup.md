# AgroVision AI — Environment Setup (Windows, PowerShell)

Target dev machine: RTX 4070 Laptop GPU (8 GB VRAM), Windows 10/11.
Run every command from the project root (`<project-root>`).

> **Order matters.** PyTorch must be installed from the official CUDA index
> *before* anything else. If `ultralytics` is installed first, pip resolves its
> `torch` dependency from PyPI, which on Windows is the **CPU-only** build.
> That is also why torch is never listed in a plain `requirements.txt` here.

## 1. Python

Use Python 3.11 or 3.12 (64-bit). Check:

```powershell
python --version
```

## 2. Virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If activation is blocked by the execution policy, allow local scripts for your user once, then retry:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

Your prompt should now start with `(.venv)`. Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

## 3. PyTorch + torchvision (CUDA build) — separate step

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

`cu128` = CUDA 12.8 wheels, which support the RTX 40-series (Ada). The CUDA
runtime ships inside the wheel — you do **not** need to install the CUDA Toolkit,
only a recent NVIDIA driver. If pytorch.org's "Get Started" selector lists a newer
CUDA index for your driver, you may use it instead; the only requirement is that
`--index-url` points at a `download.pytorch.org/whl/cuXXX` index, never plain PyPI.

## 4. Remaining packages

```powershell
python -m pip install ultralytics opencv-python pandas matplotlib pytest
```

Because a CUDA torch is already present and satisfies ultralytics' requirement,
pip will leave it alone. The verification below confirms that.

## 5. Verify

```powershell
@'
import torch
print("torch version       :", torch.__version__)
print("cuda.is_available() :", torch.cuda.is_available())
print("torch CUDA version  :", torch.version.cuda)
if torch.cuda.is_available():
    print("GPU name            :", torch.cuda.get_device_name(0))
    print("VRAM (GB)           :", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1))
'@ | python -
```

Expected: `cuda.is_available() : True`, `torch CUDA version : 12.8`,
`GPU name : NVIDIA GeForce RTX 4070 Laptop GPU`, and a torch version ending in
`+cu128` (not `+cpu`).

Optional sanity check of the ultralytics install:

```powershell
yolo checks
```

## Troubleshooting: `cuda.is_available()` returns `False`

1. **CPU build installed.** If `torch.__version__` ends in `+cpu` (or `torch.version.cuda` is `None`),
   pip pulled torch from PyPI. Fix:
   ```powershell
   python -m pip uninstall -y torch torchvision
   python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
   ```
2. **Wrong Python / venv not active.** Make sure `(.venv)` is in the prompt and that
   `(Get-Command python).Source` points into `.venv\Scripts\`. A global Python may have a different torch.
3. **Driver too old.** Run `nvidia-smi`. The "CUDA Version" in its header is the maximum the
   driver supports and must be **≥** the wheel's CUDA version (12.8). If lower, update the
   NVIDIA driver (GeForce Experience / NVIDIA App or nvidia.com), reboot, re-check.
4. **`nvidia-smi` not found or shows no GPU.** The driver is missing/broken, or the laptop is in
   an iGPU-only / power-saving mode. Install the driver; in the vendor control software
   (e.g. MUX / "GPU mode") select Hybrid/Discrete, not Integrated-only.
5. **GPU hidden by environment.** Check `$env:CUDA_VISIBLE_DEVICES`. If it is set to `""` or `-1`,
   clear it: `Remove-Item Env:CUDA_VISIBLE_DEVICES`.
6. **Still failing.** `python -c "import torch; torch.zeros(1).cuda()"` prints the underlying
   CUDA error, which usually names the exact cause.
