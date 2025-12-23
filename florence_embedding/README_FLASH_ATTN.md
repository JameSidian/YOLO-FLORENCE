# Florence-2-large and flash_attn Requirement

## Problem
Florence-2-large requires `flash_attn` (Flash Attention) which is difficult to install on Windows, especially with CPU-only PyTorch.

## Solutions

### Option 1: Use Florence-2-base-ft (Recommended for Windows)
Florence-2-base-ft does NOT require flash_attn and works perfectly on Windows:

```powershell
python download_florence2.py --model-id microsoft/Florence-2-base-ft
```

**Pros:**
- ✅ Works on Windows without issues
- ✅ Smaller download (~1 GB vs ~3 GB)
- ✅ Faster inference
- ✅ Fine-tuned version (better for many tasks)

**Cons:**
- ❌ Slightly lower performance than large model

### Option 2: Try Installing flash-attn (May Fail on Windows)

If you really need the large model, you can try:

```powershell
pip install flash-attn --no-build-isolation
```

**Note:** This will likely fail on Windows CPU setup because:
- flash-attn requires CUDA
- Requires compilation (C++ compiler needed)
- Windows compatibility is limited

### Option 3: Use WSL2 or Linux
If you have WSL2 installed, you can run the scripts there where flash-attn installation is more reliable.

### Option 4: Use CUDA-enabled PyTorch
If you have an NVIDIA GPU, install CUDA-enabled PyTorch first, then try flash-attn:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install flash-attn --no-build-isolation
```

## Recommendation

**For Windows CPU setup, use Florence-2-base-ft.** It provides excellent results and doesn't require flash_attn.

To update all scripts to use base-ft instead of large:

1. Change default in `embed_screenshots.py`:
   ```python
   "model_id": "microsoft/Florence-2-base-ft"
   ```

2. Change default in `test_florence2_detection.py`:
   ```python
   default="microsoft/Florence-2-base-ft"
   ```

3. Change default in `download_florence2.py`:
   ```python
   default="microsoft/Florence-2-base-ft"
   ```

Note: base-ft uses revision='refs/pr/6' which the scripts already handle.




