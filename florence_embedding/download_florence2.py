#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download Florence-2 Model

Downloads Florence-2 model to a specific directory.
This allows you to control where the model is stored.

Usage:
    python download_florence2.py [--model-id microsoft/Florence-2-base-ft] [--cache-dir C:/path/to/dir]
"""

import argparse
import os
import sys
from pathlib import Path

# Workaround: Patch transformers to skip flash_attn import check
# Florence-2 models check for flash_attn during import, but have CPU fallback
# Based on: https://huggingface.co/microsoft/Florence-2-base/discussions/4
try:
    from transformers import AutoModelForCausalLM, AutoProcessor
    from transformers import dynamic_module_utils
    import torch
    TRANSFORMERS_AVAILABLE = True
    
    # Patch get_imports to remove flash_attn from required imports
    # This is the function that actually reads the file and extracts imports
    original_get_imports = dynamic_module_utils.get_imports
    
    def patched_get_imports(filename):
        """Patch get_imports to remove flash_attn from imports"""
        imports = original_get_imports(filename)
        # Remove flash_attn from the imports list if present
        if isinstance(imports, list):
            imports = [imp for imp in imports if imp != "flash_attn" and imp != "flash-attn"]
        elif isinstance(imports, set):
            imports = imports - {"flash_attn", "flash-attn"}
        return imports
    
    dynamic_module_utils.get_imports = patched_get_imports
    
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("ERROR: transformers not installed. Install with: pip install transformers")
    sys.exit(1)


def download_model(model_id: str, cache_dir: str, device: str = "cpu"):
    """Download Florence-2 model to specified directory"""
    
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading Florence-2 model: {model_id}")
    print(f"Target directory: {cache_path}")
    print(f"Device: {device}")
    print("\nThis may take several minutes depending on your internet connection...")
    print("Model sizes:")
    print("  - Florence-2-base-ft: ~1 GB (default - works on Windows)")
    print("  - Florence-2-large: ~3 GB (requires flash_attn - may not work on Windows)")
    print()
    
    device_obj = torch.device(device)
    
    try:
        # Download model with custom cache directory
        print("Downloading model...")
        model_kwargs = {
            "trust_remote_code": True,
            "cache_dir": str(cache_path)
        }
        
        # base-ft model requires revision='refs/pr/6'
        if "base-ft" in model_id.lower():
            model_kwargs["revision"] = 'refs/pr/6'
        
        try:
            # Use SDPA (scaled dot product attention) instead of flash_attn
            # This forces standard attention which works on CPU
            model_kwargs["attn_implementation"] = "sdpa"
            
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                **model_kwargs
            )
        except ImportError as e:
            if "flash_attn" in str(e):
                print("\n" + "="*70)
                print(f"ERROR: flash_attn check failed for {model_id}")
                print("="*70)
                print("\nFlorence-2-base-ft checks for flash_attn during import, but should work")
                print("without it using CPU fallback. The patch didn't work as expected.")
                print("\nOPTIONS:")
                print("  1. Try installing flash-attn anyway (may fail on Windows):")
                print("     pip install flash-attn --no-build-isolation")
                print("  2. Use WSL2/Linux environment")
                print("  3. Modify the cached modeling file to remove the flash_attn check")
                print("  4. Use a different vision model (e.g., CLIP variants)")
                print("\n" + "="*70)
                raise
            else:
                raise
        print("[OK] Model downloaded successfully")
        
        # Download processor
        print("Downloading processor...")
        processor_kwargs = {
            "trust_remote_code": True,
            "cache_dir": str(cache_path)
        }
        
        # base-ft model requires revision='refs/pr/6'
        if "base-ft" in model_id.lower():
            processor_kwargs["revision"] = 'refs/pr/6'
        
        processor = AutoProcessor.from_pretrained(
            model_id,
            **processor_kwargs
        )
        print("[OK] Processor downloaded successfully")
        
        # Optional: Move model to device to verify it loads
        print(f"Verifying model loads on {device}...")
        model = model.to(device_obj)
        model.eval()
        print("[OK] Model verification complete")
        
        print("\n" + "="*60)
        print("DOWNLOAD COMPLETE!")
        print("="*60)
        print(f"Model saved to: {cache_path}")
        print(f"\nTo use this model, set the cache directory when loading:")
        print(f'  cache_dir="{cache_path}"')
        print("\nOr set the environment variable:")
        print(f'  export HF_HOME="{cache_path}"')
        print(f'  (Windows: set HF_HOME="{cache_path}")')
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\nERROR: Failed to download model: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download Florence-2 model to a specific directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download to default location (dataprocessing directory)
  python download_florence2.py
  
  # Download Florence-2-large instead (requires flash_attn - may not work on Windows)
  python download_florence2.py --model-id microsoft/Florence-2-large
  
  # Download to custom directory
  python download_florence2.py --cache-dir C:/models/florence2
        """
    )
    parser.add_argument("--model-id", type=str, default="microsoft/Florence-2-base-ft",
                       help="Florence-2 model ID (default: microsoft/Florence-2-base-ft)")
    parser.add_argument("--cache-dir", type=str, 
                       default=r"C:\Users\brian\OneDrive\Desktop\dataprocessing\florence2_models",
                       help="Directory to save the model (default: florence2_models in project root)")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to use for verification (default: cpu)")
    
    args = parser.parse_args()
    
    if not TRANSFORMERS_AVAILABLE:
        print("ERROR: Please install transformers: pip install transformers")
        return 1
    
    success = download_model(args.model_id, args.cache_dir, args.device)
    
    if success:
        print("\n[OK] Model download completed successfully!")
        return 0
    else:
        print("\n[ERROR] Model download failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())

