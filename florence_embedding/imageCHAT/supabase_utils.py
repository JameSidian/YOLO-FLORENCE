#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supabase Utilities for Image Descriptions and Image Embeddings Tables
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from supabase import create_client, Client
import numpy as np
import json
import ast

# Load environment variables
try:
    from dotenv import load_dotenv
    BASE_DIR = Path(r"C:\Users\brian\OneDrive\Desktop\dataprocessing")
    env_path = BASE_DIR / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

# Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://nxrhvostwdtixojqyvro.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

_supabase_client: Optional[Client] = None


def get_supabase_client() -> Client:
    """Initialize and return Supabase client"""
    global _supabase_client
    if _supabase_client is None:
        if not SUPABASE_KEY:
            raise ValueError("SUPABASE_KEY not set in environment variables")
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _supabase_client


def search_text_embeddings(
    query_embedding: List[float],
    top_k: int = 3,
    use_summary: bool = True
) -> List[Dict[str, Any]]:
    """
    Search image_descriptions table using text embedding similarity
    
    Args:
        query_embedding: Text embedding vector (1536-dim from text-embedding-3-small)
        top_k: Number of results to return
        use_summary: If True, search summary_embedding; if False, search text_verbatim_embedding
    
    Returns:
        List of matching image descriptions with similarity scores
    """
    client = get_supabase_client()
    
    embedding_column = "summary_embedding" if use_summary else "text_verbatim_embedding"
    
    # Try to use RPC function for vector search (if it exists in your database)
    # You can create this function in Supabase SQL Editor:
    # CREATE OR REPLACE FUNCTION search_text_embeddings(
    #   query_embedding vector(1536),
    #   embedding_col text,
    #   top_k int
    # ) RETURNS TABLE(...) AS $$
    # ...
    # $$ LANGUAGE plpgsql;
    
    try:
        # Attempt RPC call (will fail if function doesn't exist)
        response = client.rpc(
            'search_text_embeddings',
            {
                'query_embedding': str(query_embedding),
                'embedding_col': embedding_column,
                'top_k': top_k
            }
        ).execute()
        
        if response.data:
            return response.data
    except Exception as e:
        # RPC function doesn't exist, fall back to Python-side computation
        pass
    
    # Fallback: Fetch records and compute similarity in Python
    # Note: This is less efficient for large datasets. For production, create the RPC function above.
    response = client.table("image_descriptions").select(
        "id, project_key, page_num, region_number, image_id, relative_path, "
        "classification, location, level, orientation, element_type, "
        "grid_references, section_callouts, element_callouts, key_components, "
        "text_verbatim, summary, " + embedding_column
    ).not_.is_(embedding_column, "null").limit(5000).execute()  # Increased limit for better coverage
    
    if not response.data:
        return []
    
    # Compute cosine similarity efficiently using numpy
    query_vec = np.array(query_embedding, dtype=np.float32)
    query_norm = np.linalg.norm(query_vec)
    
    if query_norm == 0:
        return []
    
    results = []
    for row in response.data:
        stored_embedding = row.get(embedding_column)
        if stored_embedding is None:
            continue
        
        # Handle string representation of embeddings (from Supabase)
        if isinstance(stored_embedding, str):
            try:
                # Try to parse as JSON array first
                stored_embedding = json.loads(stored_embedding)
            except (json.JSONDecodeError, ValueError):
                # If that fails, try ast.literal_eval for Python list format
                try:
                    stored_embedding = ast.literal_eval(stored_embedding)
                except (ValueError, SyntaxError):
                    # If both fail, skip this embedding
                    continue
        
        stored_vec = np.array(stored_embedding, dtype=np.float32)
        stored_norm = np.linalg.norm(stored_vec)
        
        if stored_norm == 0:
            continue
        
        # Cosine similarity
        similarity = np.dot(query_vec, stored_vec) / (query_norm * stored_norm)
        distance = 1 - similarity
        
        result = {
            **{k: v for k, v in row.items() if k != embedding_column},
            "similarity": float(similarity),
            "distance": float(distance),
            "search_type": "text_embedding"
        }
        results.append(result)
    
    # Sort by similarity (descending) and return top_k
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]


def search_image_embeddings(
    query_embedding: List[float],
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """
    Search image_embeddings table using CLIP image embedding similarity
    
    Args:
        query_embedding: CLIP image embedding vector (1024-dim from ViT-H-14)
        top_k: Number of results to return
    
    Returns:
        List of matching image embeddings with similarity scores
    """
    client = get_supabase_client()
    
    # Try RPC function first (if it exists)
    try:
        response = client.rpc(
            'search_image_embeddings',
            {
                'query_embedding': str(query_embedding),
                'top_k': top_k
            }
        ).execute()
        
        if response.data:
            return response.data
    except Exception:
        pass
    
    # Fallback: Fetch and compute similarity
    # Note: image_embeddings table uses page_number (not page_num like image_descriptions)
    response = client.table("image_embeddings").select(
        "id, project_key, page_number, embedding, image_url"
    ).not_.is_("embedding", "null").limit(5000).execute()  # Increased limit
    
    if not response.data:
        return []
    
    # Compute cosine similarity efficiently
    query_vec = np.array(query_embedding, dtype=np.float32)
    query_norm = np.linalg.norm(query_vec)
    
    if query_norm == 0:
        return []
    
    results = []
    for row in response.data:
        stored_embedding = row.get("embedding")
        if stored_embedding is None:
            continue
        
        # Handle string representation of embeddings (from Supabase)
        if isinstance(stored_embedding, str):
            try:
                # Try to parse as JSON array first
                stored_embedding = json.loads(stored_embedding)
            except (json.JSONDecodeError, ValueError):
                # If that fails, try ast.literal_eval for Python list format
                try:
                    stored_embedding = ast.literal_eval(stored_embedding)
                except (ValueError, SyntaxError):
                    # If both fail, skip this embedding
                    continue
        
        stored_vec = np.array(stored_embedding, dtype=np.float32)
        stored_norm = np.linalg.norm(stored_vec)
        
        if stored_norm == 0:
            continue
        
        # Cosine similarity
        similarity = np.dot(query_vec, stored_vec) / (query_norm * stored_norm)
        distance = 1 - similarity
        
        result = {
            **{k: v for k, v in row.items() if k != "embedding"},
            "similarity": float(similarity),
            "distance": float(distance),
            "search_type": "image_embedding"
        }
        # Normalize page_number to page_num for consistency with image_descriptions
        if "page_number" in result:
            result["page_num"] = result.pop("page_number")
        results.append(result)
    
    # Sort by similarity (descending) and return top_k
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]


def get_image_descriptions_by_paths(
    project_key: str,
    relative_paths: List[str]
) -> List[Dict[str, Any]]:
    """
    Get image descriptions by project_key and relative_paths
    
    Args:
        project_key: Project identifier
        relative_paths: List of relative paths (e.g., ["page_001/region_01_red_box.png"])
    
    Returns:
        List of matching image descriptions
    """
    client = get_supabase_client()
    
    results = []
    for rel_path in relative_paths:
        response = client.table("image_descriptions").select("*").eq(
            "project_key", project_key
        ).eq("relative_path", rel_path).execute()
        
        if response.data:
            results.extend(response.data)
    
    return results


def construct_image_url(project_key: str, relative_path: str) -> str:
    """
    Construct Supabase storage URL from project_key and relative_path
    
    Args:
        project_key: Project identifier (e.g., "25-01-006")
        relative_path: Relative path (e.g., "page_001/region_01_red_box.png")
    
    Returns:
        Full Supabase storage URL
    """
    base_url = SUPABASE_URL.replace(".supabase.co", "")
    bucket_path = f"test_embeddings/{project_key}/{relative_path}"
    return f"{base_url}.supabase.co/storage/v1/object/public/images/{bucket_path}"
