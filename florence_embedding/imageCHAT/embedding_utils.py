#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Embedding Utilities for CLIP and Text Embeddings
"""

import os
from pathlib import Path
from typing import Optional, List
import numpy as np
from PIL import Image
import cv2
import torch

# Load environment variables
try:
    from dotenv import load_dotenv
    BASE_DIR = Path(r"C:\Users\brian\OneDrive\Desktop\dataprocessing")
    env_path = BASE_DIR / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

# OpenAI for text embeddings
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# OpenCLIP for image embeddings
try:
    import open_clip
    OPENCLIP_AVAILABLE = True
except ImportError:
    OPENCLIP_AVAILABLE = False

# Configuration
# Check for OPENAI_API_KEY first, fallback to API_KEY for compatibility
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY", "")
CLIP_MODEL_NAME = "ViT-H-14"
CLIP_PRETRAINED = "laion2b_s32b_b79k"
CLIP_EMBEDDING_DIM = 1024

# Global model caches
_clip_model = None
_clip_preprocess = None
_clip_device = None
_openai_client = None


def get_openai_client():
    """Initialize OpenAI client"""
    global _openai_client
    if _openai_client is None:
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set in environment variables")
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


def load_clip_model(device: str = None):
    """Load CLIP model (cached globally)"""
    global _clip_model, _clip_preprocess, _clip_device
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if _clip_model is None or _clip_device != device:
        if not OPENCLIP_AVAILABLE:
            raise ImportError("open_clip not installed. Install with: pip install open-clip-torch")
        
        print(f"Loading CLIP model: {CLIP_MODEL_NAME} / {CLIP_PRETRAINED}")
        _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
            model_name=CLIP_MODEL_NAME,
            pretrained=CLIP_PRETRAINED,
            device=device
        )
        _clip_model.eval()
        _clip_device = device
        print(f"CLIP model loaded on {device}")
    
    return _clip_model, _clip_preprocess


def preprocess_image(image: Image.Image, mode: str = "normalize_lines") -> Image.Image:
    """
    Preprocess image to normalize line weights (same as embed_screenshots.py)
    """
    if mode == "none":
        return image
    
    img_array = np.array(image.convert("RGB"))
    
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    if mode == "normalize_lines":
        mean_intensity = np.mean(gray)
        if mean_intensity < 128:
            gray = 255 - gray
        
        kernel = np.ones((2, 2), np.uint8)
        normalized = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=1)
        normalized = cv2.morphologyEx(normalized, cv2.MORPH_OPEN, kernel, iterations=1)
        result = cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGB)
    else:
        return image
    
    return Image.fromarray(result)


def embed_image_clip(image: Image.Image, device: str = None) -> Optional[List[float]]:
    """
    Generate CLIP embedding for an image
    
    Args:
        image: PIL Image
        device: Device to use ("cuda" or "cpu")
    
    Returns:
        CLIP embedding vector (1024-dim) or None if error
    """
    try:
        if not OPENCLIP_AVAILABLE:
            raise ImportError("open_clip not installed. Install with: pip install open-clip-torch")
        
        if image is None:
            raise ValueError("Image is None")
        
        model, preprocess_fn = load_clip_model(device)
        
        # Preprocess image
        image = preprocess_image(image, mode="normalize_lines")
        
        # Process with CLIP
        image_tensor = preprocess_fn(image).unsqueeze(0).to(_clip_device)
        
        with torch.no_grad():
            image_features = model.encode_image(image_tensor)
            # Normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            embedding = image_features.cpu().numpy()[0]
        
        return embedding.tolist()
    
    except Exception as e:
        import traceback
        error_msg = f"Error generating CLIP embedding: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        # Return None but the error is logged
        return None


def embed_text_openai(text: str) -> Optional[List[float]]:
    """
    Generate text embedding using OpenAI text-embedding-3-small
    
    Args:
        text: Text to embed
    
    Returns:
        Text embedding vector (1536-dim) or None if error
    """
    try:
        if not OPENAI_AVAILABLE:
            raise ImportError("openai not available")
        
        client = get_openai_client()
        
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text.strip()
        )
        
        return response.data[0].embedding
    
    except Exception as e:
        print(f"Error generating text embedding: {e}")
        return None
