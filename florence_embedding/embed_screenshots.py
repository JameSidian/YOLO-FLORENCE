#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Florence-2 Image Embedding Script

Embeds all screenshots using Microsoft Florence-2 vision-language model:
1. Scans output directory for all PNG images
2. Uses Florence-2 vision encoder to generate embeddings for each image
3. Stores embeddings in FAISS index
4. Saves metadata mapping

Florence-2 is particularly well-suited for technical engineering drawings due to:
- Built-in OCR capabilities for reading text/labels in drawings
- Document understanding architecture (DocVQA)
- Better spatial relationship understanding
- Training on structured document data

Usage:
    python embed_screenshots.py [--output-dir output] [--embeddings-dir embeddings]
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import numpy as np

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
    print("ERROR: transformers not installed. Install with: pip install transformers")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("ERROR: faiss-cpu not installed. Install with: pip install faiss-cpu")

from PIL import Image
import cv2
import hashlib

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False


# =============================
# Configuration
# =============================
# Initialize device detection after imports
_device = "cpu"
if TRANSFORMERS_AVAILABLE:
    import torch
    _device = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_CONFIG = {
    "model_id": "microsoft/Florence-2-base-ft",  # Florence-2 model: base-ft (works on Windows, no flash_attn needed)
    "batch_size": 8,  # Can use larger batch size with base model
    "device": _device,
    "embedding_dim": 1024,  # Florence-2-base-ft: 1024 dimensions
    "preprocess_mode": "normalize_lines",  # Options: "none", "normalize_lines", "edge_detection", "adaptive_threshold"
    "cache_dir": r"C:\Users\brian\OneDrive\Desktop\dataprocessing\florence2_models",  # Custom cache directory for models
}

# =============================
# Supabase Configuration
# =============================
# Edit these values with your Supabase credentials
SUPABASE_URL = "https://nxrhvosfwdfixojqyvro.supabase.co"  # e.g., "https://your-project-id.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im54cmh2b3Nmd2RmaXhvanF5dnJvIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjEzMTM5MTQsImV4cCI6MjA3Njg4OTkxNH0.T0BWO-aHo_307ogBoisEXrqrDW0MZ_tvuWbbiCiq26U"  # Your Supabase anon key (for table operations)
SUPABASE_SERVICE_KEY = ""  # Optional: Service role key for storage uploads (bypasses RLS). Get from Settings → API → service_role key
SUPABASE_BUCKET = "images"  # Your storage bucket name

SUPABASE_CONFIG = {
    "url": SUPABASE_URL,
    "key": SUPABASE_KEY,
    "service_key": SUPABASE_SERVICE_KEY if SUPABASE_SERVICE_KEY else SUPABASE_KEY,  # Use service key for uploads if available
    "bucket_name": SUPABASE_BUCKET,
}

_supabase_client = None
_supabase_storage_client = None

def get_supabase_client():
    """Initialize and return Supabase client for table operations"""
    global _supabase_client
    if not SUPABASE_AVAILABLE:
        return None
    if _supabase_client is not None:
        return _supabase_client
    
    url = SUPABASE_CONFIG["url"]
    key = SUPABASE_CONFIG["key"]
    if not url or not key:
        return None
    
    try:
        _supabase_client = create_client(url, key)
        return _supabase_client
    except Exception as e:
        print(f"WARNING: Error initializing Supabase client: {e}")
        return None

def get_supabase_storage_client():
    """Initialize and return Supabase client for storage operations (uses service key if available)"""
    global _supabase_storage_client
    if not SUPABASE_AVAILABLE:
        return None
    if _supabase_storage_client is not None:
        return _supabase_storage_client
    
    url = SUPABASE_CONFIG["url"]
    key = SUPABASE_CONFIG["service_key"]  # Use service key for storage
    if not url or not key:
        return None
    
    try:
        _supabase_storage_client = create_client(url, key)
        return _supabase_storage_client
    except Exception as e:
        print(f"WARNING: Error initializing Supabase storage client: {e}")
        return None

def generate_id(project_key: str, page_number: int, image_path: str) -> str:
    """Generate unique ID for embedding"""
    unique_string = f"{project_key}_{page_number}_{image_path}"
    return hashlib.md5(unique_string.encode()).hexdigest()

def upload_image_to_bucket(image_path: str, bucket_name: str) -> Optional[str]:
    """Upload image to Supabase storage and return public URL"""
    # Use storage client (with service key if available) for uploads
    storage_client = get_supabase_storage_client()
    if not storage_client:
        return None
    
    try:
        with open(image_path, 'rb') as f:
            file_data = f.read()
        
        # Use relative path from project root as storage path
        # Normalize path separators
        storage_path = image_path.replace('\\', '/')
        
        # Upload to bucket (upsert to overwrite if exists)
        storage_client.storage.from_(bucket_name).upload(
            storage_path,
            file_data,
            file_options={"content-type": "image/png", "upsert": "true"}
        )
        
        # Get public URL
        public_url = storage_client.storage.from_(bucket_name).get_public_url(storage_path)
        return public_url
    except Exception as e:
        print(f"  WARNING: Failed to upload {image_path}: {e}")
        return None


# =============================
# Model Loading
# =============================
_model = None
_processor = None
_model_id = None

def load_florence2_model(model_id: str, device: str = "cpu", cache_dir: str = None):
    """Load Florence-2 model and processor, caching globally
    
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
            # Load processor for image preprocessing
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
            
            # Load model (Florence-2 uses AutoModelForCausalLM)
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
            
            # Get embedding dimension from vision encoder
            # Florence-2 uses DaViT vision encoder
            # Try to extract from model architecture
            embedding_dim = None
            
            # Method 1: Check for _encode_image method and test it
            if hasattr(_model, '_encode_image'):
                try:
                    # Create a dummy image to test
                    dummy_img = Image.new('RGB', (224, 224), color='white')
                    dummy_inputs = _processor(images=dummy_img, return_tensors="pt")
                    dummy_pixel_values = dummy_inputs["pixel_values"].to(device)
                    with torch.no_grad():
                        dummy_output = _model._encode_image(dummy_pixel_values)
                        if isinstance(dummy_output, tuple):
                            dummy_output = dummy_output[0]
                        # Shape should be [batch, seq_len, hidden_dim]
                        if len(dummy_output.shape) == 3:
                            embedding_dim = dummy_output.shape[-1]
                except Exception:
                    pass
            
            # Handle base-ft model revision requirement
            if "base-ft" in model_id.lower() and embedding_dim is None:
                # base-ft model uses revision='refs/pr/6' and has 1024-dim vision encoder
                embedding_dim = 1024
            
            # Method 2: Check vision_tower attributes
            if embedding_dim is None:
                if hasattr(_model, 'vision_tower'):
                    vision_tower = _model.vision_tower
                    if hasattr(vision_tower, 'embed_dim'):
                        embedding_dim = vision_tower.embed_dim
                    elif hasattr(vision_tower, 'config') and hasattr(vision_tower.config, 'hidden_size'):
                        embedding_dim = vision_tower.config.hidden_size
                elif hasattr(_model, 'model') and hasattr(_model.model, 'vision_tower'):
                    vision_tower = _model.model.vision_tower
                    if hasattr(vision_tower, 'embed_dim'):
                        embedding_dim = vision_tower.embed_dim
                    elif hasattr(vision_tower, 'config') and hasattr(vision_tower.config, 'hidden_size'):
                        embedding_dim = vision_tower.config.hidden_size
            
            # Method 3: Set default based on model name
            if embedding_dim is None:
                if "base-ft" in model_id.lower() or "base" in model_id.lower():
                    embedding_dim = 1024  # Florence-2-base-ft vision encoder dimension
                elif "large" in model_id.lower():
                    embedding_dim = 1024  # Florence-2-large vision encoder dimension
                else:
                    embedding_dim = 1024  # Default
            
            DEFAULT_CONFIG["embedding_dim"] = embedding_dim
            
            print(f"[OK] Model loaded successfully")
            print(f"   Embedding dimension: {DEFAULT_CONFIG['embedding_dim']}")
        except Exception as e:
            print(f"ERROR: Error loading model: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    return _model, _processor


# =============================
# Image Preprocessing
# =============================
def preprocess_image(image: Image.Image, mode: str = "normalize_lines") -> Image.Image:
    """
    Preprocess image to normalize line weights and improve embedding quality.
    
    Args:
        image: PIL Image
        mode: Preprocessing mode
            - "none": No preprocessing
            - "normalize_lines": Normalize line weights using morphological operations
            - "edge_detection": Use Canny edge detection
            - "adaptive_threshold": Use adaptive thresholding
    """
    if mode == "none":
        return image
    
    # Convert PIL to numpy array
    img_array = np.array(image.convert("RGB"))
    
    # Convert to grayscale for processing
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    if mode == "normalize_lines":
        # Normalize line weights by:
        # 1. Inverting if needed (assume white background)
        # 2. Applying morphological operations to normalize line thickness
        # 3. Converting back to RGB
        
        # Invert if background is dark (more common for drawings)
        # Check if image is mostly dark or light
        mean_intensity = np.mean(gray)
        if mean_intensity < 128:
            gray = 255 - gray
        
        # Normalize line thickness using morphological operations
        # Use a small kernel to thin/thicken lines to a standard width
        kernel = np.ones((2, 2), np.uint8)
        
        # First, thin lines that are too thick (erosion)
        # Then, ensure minimum line width (dilation)
        # This normalizes all lines to similar thickness
        normalized = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=1)
        normalized = cv2.morphologyEx(normalized, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Convert back to RGB
        result = cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGB)
        
    elif mode == "edge_detection":
        # Use Canny edge detection to extract only edges
        # This removes line weight information entirely
        edges = cv2.Canny(gray, 50, 150)
        # Convert edges to RGB (white edges on black background)
        result = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
    elif mode == "adaptive_threshold":
        # Use adaptive thresholding to binarize and normalize
        # This makes all lines the same weight
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        # Invert if needed (white background)
        mean_intensity = np.mean(thresh)
        if mean_intensity < 128:
            thresh = 255 - thresh
        result = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
        
    else:
        # Unknown mode, return original
        return image
    
    # Convert back to PIL Image
    return Image.fromarray(result)


# =============================
# Image Embedding
# =============================
def embed_image(image_path: str, model, processor, device: str = "cpu", 
                preprocess_mode: str = "normalize_lines") -> np.ndarray:
    """Generate embedding for a single image using Florence-2 vision encoder"""
    try:
        image = Image.open(image_path).convert("RGB")
        
        # Preprocess image to normalize line weights (optional, but can help for technical drawings)
        image = preprocess_image(image, mode=preprocess_mode)
        
        # Process image with Florence-2 processor
        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        
        # Extract vision encoder features
        with torch.no_grad():
            # Method 1: Use _encode_image if available (preferred method)
            if hasattr(model, '_encode_image'):
                vision_outputs = model._encode_image(pixel_values)
                # _encode_image returns (batch, seq_len, hidden_dim) or tuple
                if isinstance(vision_outputs, tuple):
                    vision_features = vision_outputs[0]
                else:
                    vision_features = vision_outputs
            # Method 2: Access vision_tower directly
            elif hasattr(model, 'vision_tower'):
                vision_outputs = model.vision_tower(pixel_values)
                if isinstance(vision_outputs, tuple):
                    vision_features = vision_outputs[0]
                elif hasattr(vision_outputs, 'last_hidden_state'):
                    vision_features = vision_outputs.last_hidden_state
                else:
                    vision_features = vision_outputs
            elif hasattr(model, 'model') and hasattr(model.model, 'vision_tower'):
                vision_outputs = model.model.vision_tower(pixel_values)
                if isinstance(vision_outputs, tuple):
                    vision_features = vision_outputs[0]
                elif hasattr(vision_outputs, 'last_hidden_state'):
                    vision_features = vision_outputs.last_hidden_state
                else:
                    vision_features = vision_outputs
            else:
                # Method 3: Use forward pass with a minimal prompt
                # Use a simple task prompt to extract vision features
                inputs_text = processor(text="<CAPTION>", return_tensors="pt", padding=True)
                inputs_text = {k: v.to(device) for k, v in inputs_text.items()}
                
                # Forward pass to get vision features
                outputs = model(pixel_values=pixel_values, **inputs_text, output_hidden_states=True)
                
                # Extract vision encoder outputs from hidden states
                if hasattr(outputs, 'vision_hidden_states') and outputs.vision_hidden_states:
                    vision_features = outputs.vision_hidden_states[-1]  # Last layer
                elif hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                    # Use first hidden state (vision encoder output)
                    vision_features = outputs.hidden_states[0]
                else:
                    raise ValueError("Could not find vision encoder outputs in model. Available attributes: " + 
                                   str(dir(outputs)))
            
            # Aggregate vision tokens into a single embedding vector
            # Vision features shape: [batch, seq_len, hidden_dim] for DaViT
            if len(vision_features.shape) == 3:  # [batch, seq_len, hidden_dim]
                # Mean pool over sequence dimension to get a single vector per image
                embedding = vision_features.mean(dim=1).squeeze(0)  # [hidden_dim]
            elif len(vision_features.shape) == 2:  # [batch, hidden_dim] - already pooled
                embedding = vision_features.squeeze(0)  # [hidden_dim]
            else:
                # Flatten if needed
                embedding = vision_features.flatten()
                # If too large, take first N dimensions (shouldn't happen)
                if len(embedding) > 2048:
                    embedding = embedding[:2048]
            
            # Convert to numpy and normalize
            embedding = embedding.cpu().numpy()
            
            # Ensure it's 1D
            if len(embedding.shape) > 1:
                embedding = embedding.flatten()
            
            # Normalize embedding (L2 normalization for cosine similarity)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            else:
                # If norm is zero, return zeros (shouldn't happen)
                embedding = np.zeros_like(embedding)
        
        return embedding.astype(np.float32)
    
    except Exception as e:
        print(f"  WARNING: Error embedding {image_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def embed_images_batch(image_paths: List[str], model, processor, device: str = "cpu",
                       preprocess_mode: str = "normalize_lines") -> List[np.ndarray]:
    """Generate embeddings for a batch of images"""
    embeddings = []
    
    for image_path in image_paths:
        embedding = embed_image(image_path, model, processor, device, preprocess_mode)
        if embedding is not None:
            embeddings.append(embedding)
        else:
            embeddings.append(None)
    
    return embeddings


# =============================
# Find All Images
# =============================
def find_all_images(output_dir: Path, only_page_subfolders: bool = True, start_from_project: Optional[str] = None) -> List[Dict[str, Any]]:
    """Find all PNG images in output directory, preserving structure
    
    Args:
        output_dir: Directory to scan for images
        only_page_subfolders: If True, only include images in page_XXX subfolders (skip root images)
    """
    images = []
    
    # Walk through output directory
    for project_dir in sorted(output_dir.iterdir()):
        if not project_dir.is_dir():
            continue
        
        project_number = project_dir.name
        
        # Skip projects before start_from_project if specified
        if start_from_project and project_number < start_from_project:
            continue
        
        # Look for manifest.json
        manifest_path = project_dir / "manifest.json"
        if manifest_path.exists():
            try:
                with open(manifest_path, 'r', encoding='utf-8') as f:
                    manifest = json.load(f)
                
                # Process each image from manifest
                for img_info in manifest.get("images", []):
                    filename = img_info.get("filename")
                    relative_path = img_info.get("relative_path", filename)
                    
                    # Skip full page images if only_page_subfolders is True
                    if only_page_subfolders and img_info.get("type") == "full_page":
                        continue
                    
                    # Determine full path
                    if img_info.get("type") == "full_page":
                        image_path = project_dir / filename
                    else:
                        # Region crop - use relative_path
                        image_path = project_dir / relative_path
                    
                    if image_path.exists() and image_path.suffix.lower() == ".png":
                        images.append({
                            "image_path": str(image_path),
                            "project_number": project_number,
                            "filename": filename,
                            "relative_path": relative_path if img_info.get("type") == "region_crop" else filename,
                            "type": img_info.get("type", "unknown"),
                            "page_number": img_info.get("page_number"),
                            "metadata": img_info
                        })
            except Exception as e:
                print(f"  WARNING: Error reading manifest {manifest_path}: {e}")
        
        # Also scan for PNG files directly (fallback)
        for png_file in project_dir.rglob("*.png"):
            # Skip if already in manifest
            if any(img["image_path"] == str(png_file) for img in images):
                continue
            
            # If only_page_subfolders, check if image is in a page_XXX subfolder
            if only_page_subfolders:
                rel_path = png_file.relative_to(project_dir)
                # Check if path contains page_XXX pattern
                path_parts = rel_path.parts
                is_in_page_folder = any(part.startswith("page_") for part in path_parts)
                if not is_in_page_folder:
                    continue  # Skip images not in page subfolders
            
            rel_path = png_file.relative_to(project_dir)
            images.append({
                "image_path": str(png_file),
                "project_number": project_number,
                "filename": png_file.name,
                "relative_path": str(rel_path),
                "type": "region_crop" if only_page_subfolders else "unknown",
                "page_number": None,
                "metadata": {}
            })
    
    return images


# =============================
# Save Embeddings
# =============================
def save_embeddings(embeddings: np.ndarray, metadata: List[Dict[str, Any]], 
                   embeddings_dir: Path, embedding_dim: int):
    """Save embeddings to FAISS HNSW index, metadata, and Supabase"""
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    # Create FAISS HNSW index (faster approximate search)
    # M=32 is a good default for HNSW (number of connections)
    # ef_construction=200 controls index construction quality
    index = faiss.IndexHNSWFlat(embedding_dim, 32)
    index.hnsw.ef_construction = 200
    index.add(embeddings)
    
    # Save FAISS index
    index_path = embeddings_dir / "embeddings.index"
    faiss.write_index(index, str(index_path))
    print(f"[OK] Saved FAISS HNSW index: {index_path}")
    
    # Save metadata
    metadata_path = embeddings_dir / "metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"[OK] Saved metadata: {metadata_path}")
    
    # Save config
    config = {
        "model_id": DEFAULT_CONFIG["model_id"],
        "embedding_dim": embedding_dim,
        "preprocess_mode": DEFAULT_CONFIG.get("preprocess_mode", "normalize_lines"),
        "total_images": len(metadata),
        "generation_timestamp": datetime.now().isoformat()
    }
    config_path = embeddings_dir / "config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"[OK] Saved config: {config_path}")
    
    # Save to Supabase
    supabase = get_supabase_client()
    if supabase:
        print(f"\nSaving {len(metadata)} embeddings to Supabase...")
        bucket_name = SUPABASE_CONFIG["bucket_name"]
        records = []
        
        for embedding, meta in tqdm(zip(embeddings, metadata), desc="Preparing records", total=len(metadata)):
            project_key = meta.get("project_number", "")
            page_number = meta.get("page_number")
            image_path = meta.get("image_path", "")
            
            if not project_key or page_number is None or not image_path:
                continue
            
            # Upload image to bucket and get URL
            image_url = upload_image_to_bucket(image_path, bucket_name)
            if not image_url:
                print(f"  WARNING: Skipping {image_path} - upload failed")
                continue
            
            records.append({
                "id": generate_id(project_key, page_number, image_path),
                "project_key": project_key,
                "page_number": page_number,
                "embedding": embedding.tolist(),
                "image_url": image_url,
            })
        
        # Batch insert to Supabase
        if records:
            chunk_size = 100
            total_inserted = 0
            for i in tqdm(range(0, len(records), chunk_size), desc="Uploading to Supabase"):
                try:
                    supabase.table("image_embeddings").insert(records[i:i+chunk_size]).execute()
                    total_inserted += len(records[i:i+chunk_size])
                except Exception as e:
                    print(f"  ERROR: Batch {i//chunk_size + 1}: {e}")
                    # Try one by one
                    for record in records[i:i+chunk_size]:
                        try:
                            supabase.table("image_embeddings").insert(record).execute()
                            total_inserted += 1
                        except Exception as e2:
                            print(f"    ERROR: Failed to insert: {record.get('id', 'unknown')}: {e2}")
            
            print(f"[OK] Saved {total_inserted}/{len(records)} embeddings to Supabase")
        else:
            print("WARNING: No valid records to upload to Supabase")
    else:
        print("WARNING: Supabase client not available, skipping Supabase save")
    
    return index_path, metadata_path


# =============================
# Main Entry Point
# =============================
def main():
    parser = argparse.ArgumentParser(
        description="Embed screenshots using Florence-2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Embed images from test_embeddings (only page subfolder images by default)
  python embed_screenshots.py
  
  # Custom output and embeddings directories
  python embed_screenshots.py --output-dir test_embeddings --embeddings-dir embeddings_florence
  
  # Include full page images from root
  python embed_screenshots.py --include-full-pages
  
  # Use Florence-2-large instead (requires flash_attn - may not work on Windows)
  python embed_screenshots.py --model-id microsoft/Florence-2-large
        """
    )
    parser.add_argument("--output-dir", "-o", type=str, default="test_embeddings",
                       help="Directory containing screenshots (default: test_embeddings)")
    parser.add_argument("--embeddings-dir", "-e", type=str, default="embeddings_florence",
                       help="Directory to save embeddings (default: embeddings_florence)")
    parser.add_argument("--model-id", type=str, default=DEFAULT_CONFIG["model_id"],
                       help=f"Florence-2 model ID (default: {DEFAULT_CONFIG['model_id']})")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"],
                       help=f"Batch size for processing (default: {DEFAULT_CONFIG['batch_size']})")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cuda/cpu, default: auto-detect)")
    parser.add_argument("--include-full-pages", action="store_true",
                       help="Include full page images from root (default: only page subfolder images)")
    parser.add_argument("--preprocess-mode", type=str, default="normalize_lines",
                       choices=["none", "normalize_lines", "edge_detection", "adaptive_threshold"],
                       help="Image preprocessing mode to normalize line weights (default: normalize_lines)")
    parser.add_argument("--start-from-project", type=str, default=None,
                       help="Start processing from this project number (e.g., '25-01-021') - skips projects before this")
    parser.add_argument("--cache-dir", type=str, default=DEFAULT_CONFIG.get("cache_dir", None),
                       help=f"Directory to cache/download Florence-2 model (default: {DEFAULT_CONFIG.get('cache_dir', 'system default')})")
    
    args = parser.parse_args()
    
    if not TRANSFORMERS_AVAILABLE:
        print("ERROR: Please install transformers: pip install transformers")
        return
    
    if not FAISS_AVAILABLE:
        print("ERROR: Please install faiss-cpu: pip install faiss-cpu")
        return
    
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"ERROR: Output directory not found: {output_dir}")
        return
    
    embeddings_dir = Path(args.embeddings_dir)
    
    # Determine device
    device = args.device or DEFAULT_CONFIG["device"]
    if device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, using CPU")
        device = "cpu"
    
    # Update config
    DEFAULT_CONFIG["model_id"] = args.model_id
    DEFAULT_CONFIG["batch_size"] = args.batch_size
    DEFAULT_CONFIG["device"] = device
    DEFAULT_CONFIG["preprocess_mode"] = args.preprocess_mode
    cache_dir = args.cache_dir if args.cache_dir else None
    if cache_dir:
        DEFAULT_CONFIG["cache_dir"] = cache_dir
        # Create directory if it doesn't exist
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    # Embedding dimension will be determined when model loads
    # Both Florence-2-base and Florence-2-large use 1024-dim vision encoder
    # But we'll verify this when the model actually loads
    
    print(f"\n{'='*60}")
    print(f"Florence-2 Image Embedding Script")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Embeddings directory: {embeddings_dir}")
    print(f"Model: {args.model_id}")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Preprocess mode: {args.preprocess_mode}")
    print(f"Embedding dimension: {DEFAULT_CONFIG['embedding_dim']}")
    print(f"{'='*60}\n")
    
    # Step 1: Find all images
    print("Step 1: Finding all images...")
    only_page_subfolders = not args.include_full_pages
    if only_page_subfolders:
        print("   Filter: Only including images in page_XXX subfolders (skipping root full-page images)")
    if args.start_from_project:
        print(f"   Starting from project: {args.start_from_project}")
    images = find_all_images(output_dir, only_page_subfolders=only_page_subfolders, start_from_project=args.start_from_project)
    print(f"[OK] Found {len(images)} images")
    
    if len(images) == 0:
        print("ERROR: No images found!")
        return
    
    # Step 2: Load model
    print(f"\nStep 2: Loading Florence-2 model...")
    try:
        cache_dir = DEFAULT_CONFIG.get("cache_dir")
        model, processor = load_florence2_model(args.model_id, device, cache_dir=cache_dir)
    except Exception as e:
        print(f"ERROR: Error loading model: {e}")
        return
    
    # Step 3: Generate embeddings
    print(f"\nStep 3: Generating embeddings...")
    embeddings_list = []
    metadata_list = []
    failed_count = 0
    
    # Process in batches
    for i in tqdm(range(0, len(images), args.batch_size), desc="Processing batches"):
        batch = images[i:i + args.batch_size]
        image_paths = [img["image_path"] for img in batch]
        
        batch_embeddings = embed_images_batch(image_paths, model, processor, device, args.preprocess_mode)
        
        for img_info, embedding in zip(batch, batch_embeddings):
            if embedding is not None:
                embeddings_list.append(embedding)
                metadata_list.append({
                    "index": len(embeddings_list) - 1,
                    "image_path": img_info["image_path"],
                    "project_number": img_info["project_number"],
                    "filename": img_info["filename"],
                    "relative_path": img_info["relative_path"],
                    "type": img_info["type"],
                    "page_number": img_info["page_number"],
                    "metadata": img_info["metadata"]
                })
            else:
                failed_count += 1
    
    if len(embeddings_list) == 0:
        print("ERROR: No embeddings generated!")
        return
    
    print(f"[OK] Generated {len(embeddings_list)} embeddings ({failed_count} failed)")
    
    # Step 4: Save embeddings
    print(f"\nStep 4: Saving embeddings...")
    embeddings_array = np.vstack(embeddings_list).astype(np.float32)
    
    try:
        index_path, metadata_path = save_embeddings(
            embeddings_array,
            metadata_list,
            embeddings_dir,
            DEFAULT_CONFIG["embedding_dim"]
        )
        
        print(f"\n{'='*60}")
        print(f"[OK] Embedding complete!")
        print(f"  Total images processed: {len(images)}")
        print(f"  Successful embeddings: {len(embeddings_list)}")
        print(f"  Failed: {failed_count}")
        print(f"  Embeddings saved to: {embeddings_dir}")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"ERROR: Error saving embeddings: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

