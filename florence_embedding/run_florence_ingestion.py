#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Florence-2 Offline Ingestion Pipeline for Engineering Drawings

Processes a folder of screenshot images using Florence-2-base-ft model and outputs
structured JSON files containing OCR text, dense semantic description, and global caption.

This script leverages the existing Florence-2 setup in this folder, including:
- Windows-compatible model (Florence-2-base-ft)
- Shared model cache directory
- Proven model loading patterns

Usage:
    python run_florence_ingestion.py [--input-dir ./screenshots] [--output-dir ./florence_json]
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
from tqdm import tqdm

# Workaround: Patch transformers to skip flash_attn import check
# Florence-2 models check for flash_attn during import, but have CPU fallback
# Based on: https://huggingface.co/microsoft/Florence-2-base/discussions/4
try:
    from transformers import AutoModelForCausalLM, AutoProcessor
    from transformers import dynamic_module_utils
    import torch
    TRANSFORMERS_AVAILABLE = True
    
    # Patch get_imports to remove flash_attn from required imports
    original_get_imports = dynamic_module_utils.get_imports
    
    def patched_get_imports(filename):
        """Patch get_imports to remove flash_attn from imports"""
        imports = original_get_imports(filename)
        if isinstance(imports, list):
            imports = [imp for imp in imports if imp != "flash_attn" and imp != "flash-attn"]
        elif isinstance(imports, set):
            imports = imports - {"flash_attn", "flash-attn"}
        return imports
    
    dynamic_module_utils.get_imports = patched_get_imports
    
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("ERROR: transformers not installed. Install with: pip install transformers torch")

from PIL import Image


# =============================
# Configuration
# =============================
# Initialize device detection after imports
_device = "cpu"
if TRANSFORMERS_AVAILABLE:
    _device = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_CONFIG = {
    "model_id": "microsoft/Florence-2-base-ft",  # Windows-compatible variant
    "device": _device,
    "supported_formats": {".png", ".jpg", ".jpeg"},
    "max_new_tokens": 1024,  # Maximum tokens for generation
    "cache_dir": r"C:\Users\brian\OneDrive\Desktop\dataprocessing\florence2_models",  # Shared cache directory
}


# =============================
# Model Loading
# =============================
_model = None
_processor = None
_model_id = None

def load_florence2_model(model_id: str, device: str = "cpu", cache_dir: str = None):
    """
    Load Florence-2 model and processor, caching globally.
    
    Leverages the proven model loading pattern from embed_screenshots.py
    with Windows compatibility (base-ft variant, revision, cache directory).
    
    Args:
        model_id: HuggingFace model ID (e.g., "microsoft/Florence-2-base-ft")
        device: Device to load model on ("cpu" or "cuda")
        cache_dir: Optional custom directory for model cache
    """
    global _model, _processor, _model_id
    
    if _model is None or _model_id != model_id:
        print(f"Loading Florence-2 model: {model_id}")
        print(f"   Device: {device}")
        if cache_dir:
            print(f"   Cache directory: {cache_dir}")
        
        try:
            # Load processor
            print("   Loading processor...")
            processor_kwargs = {
                "trust_remote_code": True
            }
            if cache_dir:
                processor_kwargs["cache_dir"] = cache_dir
            
            # base-ft model requires revision='refs/pr/6'
            if "base-ft" in model_id.lower():
                processor_kwargs["revision"] = 'refs/pr/6'
            
            _processor = AutoProcessor.from_pretrained(
                model_id,
                **processor_kwargs
            )
            
            # Load model
            print("   Loading model...")
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float32 if device == "cpu" else torch.float16,
                "attn_implementation": "sdpa"  # Use standard attention instead of flash_attn
            }
            if cache_dir:
                model_kwargs["cache_dir"] = cache_dir
            
            # base-ft model requires revision='refs/pr/6'
            if "base-ft" in model_id.lower():
                model_kwargs["revision"] = 'refs/pr/6'
            
            _model = AutoModelForCausalLM.from_pretrained(
                model_id,
                **model_kwargs
            )
            _model.to(device)
            _model.eval()
            _model_id = model_id
            
            print(f"[OK] Model loaded successfully")
        except Exception as e:
            print(f"ERROR: Error loading model: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    return _model, _processor


# =============================
# Task Execution
# =============================
def run_florence_task(image: Image.Image, task_token: str, model, processor, device: str, max_new_tokens: int = 1024) -> str:
    """
    Run a single Florence-2 task on an image.
    
    Args:
        image: PIL Image
        task_token: Task token (e.g., "<OCR>", "<CAPTION>", "<DENSE_CAPTION>")
        model: Loaded Florence-2 model
        processor: Florence-2 processor
        device: Device to run on
        max_new_tokens: Maximum tokens to generate
        
    Returns:
        Generated text output
    """
    try:
        # Prepare inputs with task token
        inputs = processor(text=task_token, images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get input length to extract only newly generated tokens
        input_ids = inputs.get("input_ids")
        input_length = input_ids.shape[1] if input_ids is not None else 0
        
        # Generate output
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Deterministic output
            )
        
        # Extract only the newly generated tokens (skip input tokens)
        if input_length > 0 and generated_ids.shape[1] > input_length:
            generated_token_ids = generated_ids[0, input_length:]
        else:
            generated_token_ids = generated_ids[0]
        
        # Decode only the generated tokens
        # Use tokenizer if available, otherwise use processor.decode
        if hasattr(processor, 'tokenizer') and processor.tokenizer is not None:
            generated_text = processor.tokenizer.decode(
                generated_token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
        elif hasattr(processor, 'decode'):
            generated_text = processor.decode(
                generated_token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
        else:
            # Fallback: decode the full sequence and remove input
            generated_text = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )[0]
            # Remove task token if present
            if generated_text.startswith(task_token):
                generated_text = generated_text[len(task_token):].strip()
        
        # Clean up the output
        generated_text = generated_text.strip()
        
        return generated_text
    
    except Exception as e:
        print(f"  WARNING: Error running task {task_token}: {e}")
        import traceback
        traceback.print_exc()
        return ""


def process_image(image_path: Path, model, processor, device: str, max_new_tokens: int = 1024) -> Dict[str, Any]:
    """
    Process a single image with all Florence-2 tasks.
    
    Args:
        image_path: Path to image file
        model: Loaded Florence-2 model
        processor: Florence-2 processor
        device: Device to run on
        max_new_tokens: Maximum tokens to generate
        
    Returns:
        Dictionary with extraction results
    """
    try:
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Run OCR task
        ocr_text = run_florence_task(image, "<OCR>", model, processor, device, max_new_tokens)
        
        # Run CAPTION task
        caption = run_florence_task(image, "<CAPTION>", model, processor, device, max_new_tokens)
        
        # Run DENSE_CAPTION task
        # Note: Florence-2 uses <DENSE_REGION_CAPTION> but user specified <DENSE_CAPTION>
        # Try both to be safe
        dense_caption = run_florence_task(image, "<DENSE_CAPTION>", model, processor, device, max_new_tokens)
        if not dense_caption or len(dense_caption.strip()) == 0:
            # Fallback to DENSE_REGION_CAPTION if DENSE_CAPTION doesn't work
            dense_caption = run_florence_task(image, "<DENSE_REGION_CAPTION>", model, processor, device, max_new_tokens)
        
        return {
            "ocr_text": ocr_text,
            "caption": caption,
            "dense_caption": dense_caption
        }
    
    except Exception as e:
        print(f"  ERROR: Failed to process {image_path}: {e}")
        import traceback
        traceback.print_exc()
        return {
            "ocr_text": "",
            "caption": "",
            "dense_caption": ""
        }


# =============================
# File Operations
# =============================
def find_images(input_dir: Path) -> list[Path]:
    """Find all supported image files in input directory"""
    images = []
    for ext in DEFAULT_CONFIG["supported_formats"]:
        images.extend(input_dir.rglob(f"*{ext}"))
        images.extend(input_dir.rglob(f"*{ext.upper()}"))
    return sorted(images)


def save_json_output(image_path: Path, output_dir: Path, extraction_results: Dict[str, str]):
    """
    Save JSON output file for a single image.
    
    Args:
        image_path: Path to source image
        output_dir: Output directory for JSON files
        extraction_results: Dictionary with ocr_text, caption, dense_caption
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create output filename based on image filename
    image_stem = image_path.stem
    output_path = output_dir / f"{image_stem}.json"
    
    # Build output JSON structure
    output_data = {
        "image_id": image_path.name,
        "model": "florence-2-base-ft",
        "tasks_run": ["ocr", "caption", "dense_caption"],
        "extraction": {
            "caption": extraction_results.get("caption", ""),
            "dense_caption": extraction_results.get("dense_caption", ""),
            "ocr_text": extraction_results.get("ocr_text", "")
        },
        "metadata": {
            "source": "screenshot",
            "ingestion_version": "v1"
        }
    }
    
    # Save JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    return output_path


# =============================
# Main Entry Point
# =============================
def main():
    parser = argparse.ArgumentParser(
        description="Florence-2 Offline Ingestion Pipeline for Engineering Drawings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process images from default directory
  python run_florence_ingestion.py
  
  # Custom input and output directories
  python run_florence_ingestion.py --input-dir ./screenshots --output-dir ./florence_json
  
  # Use CPU explicitly
  python run_florence_ingestion.py --device cpu
  
  # Use custom cache directory
  python run_florence_ingestion.py --cache-dir C:/models/florence2
        """
    )
    parser.add_argument("--input-dir", "-i", type=str, default="./screenshots",
                       help="Directory containing input images (default: ./screenshots)")
    parser.add_argument("--output-dir", "-o", type=str, default="./florence_json",
                       help="Directory to save JSON outputs (default: ./florence_json)")
    parser.add_argument("--model-id", type=str, default=DEFAULT_CONFIG["model_id"],
                       help=f"Florence-2 model ID (default: {DEFAULT_CONFIG['model_id']})")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cuda/cpu, default: auto-detect)")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_CONFIG["max_new_tokens"],
                       help=f"Maximum tokens to generate (default: {DEFAULT_CONFIG['max_new_tokens']})")
    parser.add_argument("--cache-dir", type=str, default=DEFAULT_CONFIG.get("cache_dir", None),
                       help=f"Directory to cache/download Florence-2 model (default: {DEFAULT_CONFIG.get('cache_dir', 'system default')})")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Print extracted captions and OCR text to console for verification")
    
    args = parser.parse_args()
    
    if not TRANSFORMERS_AVAILABLE:
        print("ERROR: Please install transformers: pip install transformers torch")
        return
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"ERROR: Input directory not found: {input_dir}")
        return
    
    output_dir = Path(args.output_dir)
    
    # Determine device
    device = args.device or DEFAULT_CONFIG["device"]
    if device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, using CPU")
        device = "cpu"
    
    # Get cache directory
    cache_dir = args.cache_dir if args.cache_dir else DEFAULT_CONFIG.get("cache_dir")
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_dir = str(cache_dir)
    
    print(f"\n{'='*60}")
    print(f"Florence-2 Offline Ingestion Pipeline")
    print(f"{'='*60}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Model: {args.model_id}")
    print(f"Device: {device}")
    print(f"Max new tokens: {args.max_new_tokens}")
    if cache_dir:
        print(f"Cache directory: {cache_dir}")
    print(f"{'='*60}\n")
    
    # Step 1: Find all images
    print("Step 1: Finding all images...")
    images = find_images(input_dir)
    print(f"[OK] Found {len(images)} images")
    
    if len(images) == 0:
        print("ERROR: No images found!")
        return
    
    # Step 2: Load model
    print(f"\nStep 2: Loading Florence-2 model...")
    try:
        model, processor = load_florence2_model(args.model_id, device, cache_dir=cache_dir)
    except Exception as e:
        print(f"ERROR: Error loading model: {e}")
        return
    
    # Step 3: Process all images
    print(f"\nStep 3: Processing images...")
    successful = 0
    failed = 0
    
    for image_path in tqdm(images, desc="Processing images"):
        # Process image with all tasks
        extraction_results = process_image(image_path, model, processor, device, args.max_new_tokens)
        
        # Print results if verbose mode
        if args.verbose:
            print(f"\n{'='*60}")
            print(f"Image: {image_path.name}")
            print(f"{'='*60}")
            print(f"\nCAPTION:")
            print(f"  {extraction_results.get('caption', '')}")
            print(f"\nDENSE CAPTION:")
            print(f"  {extraction_results.get('dense_caption', '')}")
            print(f"\nOCR TEXT:")
            print(f"  {extraction_results.get('ocr_text', '')}")
            print(f"{'='*60}\n")
        
        # Save JSON output
        try:
            output_path = save_json_output(image_path, output_dir, extraction_results)
            successful += 1
        except Exception as e:
            print(f"  ERROR: Failed to save JSON for {image_path}: {e}")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"[OK] Processing complete!")
    print(f"  Total images: {len(images)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  JSON files saved to: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
