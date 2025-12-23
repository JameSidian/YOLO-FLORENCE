#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangGraph Orchestrator for Routing Queries
"""

from typing import Dict, Any, List, Optional, Literal
from PIL import Image
from enum import Enum

from embedding_utils import embed_image_clip, embed_text_openai
from supabase_utils import (
    search_text_embeddings,
    search_image_embeddings,
    get_image_descriptions_by_paths,
    construct_image_url
)
from gpt4o_utils import describe_image_with_gpt4o, generate_text_response


class QueryType(Enum):
    """Types of queries the system can handle"""
    TEXT_TO_TEXT = "text_to_text"
    TEXT_TO_IMAGES = "text_to_images"
    IMAGE_TO_IMAGES = "image_to_images"
    IMAGE_TO_TEXT = "image_to_text"


def classify_query(
    has_text: bool,
    has_image: bool,
    user_intent: Optional[str] = None
) -> QueryType:
    """
    Classify the type of query based on input
    
    Args:
        has_text: Whether user provided text input
        has_image: Whether user provided image input
        user_intent: Optional explicit intent (e.g., "show me images", "answer my question")
    
    Returns:
        QueryType enum
    """
    if has_image and has_text:
        # If both provided, check intent or default to IMAGE_TO_TEXT
        if user_intent and ("image" in user_intent.lower() or "show" in user_intent.lower()):
            return QueryType.IMAGE_TO_IMAGES
        return QueryType.IMAGE_TO_TEXT
    
    if has_image:
        return QueryType.IMAGE_TO_IMAGES
    
    if has_text:
        # Check if user wants images or text response
        if user_intent:
            intent_lower = user_intent.lower()
            if any(word in intent_lower for word in ["show", "image", "picture", "drawing", "screenshot"]):
                return QueryType.TEXT_TO_IMAGES
        # Default to text response
        return QueryType.TEXT_TO_TEXT
    
    raise ValueError("Query must have at least text or image input")


def route_text_to_text(
    text_query: str,
    conversation_history: List[Dict[str, str]] = None,
    top_k: int = 3
) -> Dict[str, Any]:
    """
    Route: Text Query → Text Response
    
    Args:
        text_query: User's text question
        conversation_history: Previous conversation messages
        top_k: Number of results to retrieve
    
    Returns:
        Dict with 'response' (text) and 'sources' (list of descriptions)
    """
    # Embed text query
    query_embedding = embed_text_openai(text_query)
    if not query_embedding:
        return {
            "response": "I encountered an error processing your query. Please try again.",
            "sources": [],
            "images": []
        }
    
    # Search text embeddings
    descriptions = search_text_embeddings(query_embedding, top_k=top_k, use_summary=True)
    
    if not descriptions:
        return {
            "response": "I couldn't find any relevant information to answer your question. Please try rephrasing or asking about a different topic.",
            "sources": [],
            "images": []
        }
    
    # Generate response with GPT-4o
    response_text = generate_text_response(text_query, descriptions, conversation_history)
    
    return {
        "response": response_text,
        "sources": descriptions,
        "images": []  # No images for text-to-text
    }


def route_text_to_images(
    text_query: str,
    top_k: int = 3
) -> Dict[str, Any]:
    """
    Route: Text Query → Images
    
    Args:
        text_query: User's text query
        top_k: Number of images to return
    
    Returns:
        Dict with 'response' (text summary), 'sources' (descriptions), and 'images' (list of image URLs)
    """
    # Embed text query
    query_embedding = embed_text_openai(text_query)
    if not query_embedding:
        return {
            "response": "I encountered an error processing your query.",
            "sources": [],
            "images": []
        }
    
    # Search text embeddings
    descriptions = search_text_embeddings(query_embedding, top_k=top_k, use_summary=True)
    
    if not descriptions:
        return {
            "response": "I couldn't find any relevant images for your query.",
            "sources": [],
            "images": []
        }
    
    # Construct image URLs and format response
    images = []
    image_info = []
    
    for desc in descriptions:
        project_key = desc.get("project_key")
        relative_path = desc.get("relative_path")
        
        if project_key and relative_path:
            image_url = construct_image_url(project_key, relative_path)
            images.append(image_url)
            image_info.append({
                "url": image_url,
                "project_key": project_key,
                "page_num": desc.get("page_num"),
                "region_number": desc.get("region_number"),
                "description": desc.get("summary", "")[:200]
            })
    
    # Generate summary response
    response_text = f"I found {len(images)} relevant image(s) for your query:\n\n"
    for i, info in enumerate(image_info, 1):
        response_text += f"[{i}] Project {info['project_key']}, Page {info['page_num']}"
        if info.get('region_number'):
            response_text += f", Region {info['region_number']}"
        response_text += "\n"
    
    return {
        "response": response_text,
        "sources": descriptions,
        "images": images,
        "image_info": image_info
    }


def route_image_to_images(
    image: Image.Image,
    top_k: int = 3
) -> Dict[str, Any]:
    """
    Route: Image Upload → Similar Images
    
    Args:
        image: User's uploaded image
        top_k: Number of similar images to return
    
    Returns:
        Dict with 'response' (text), 'sources' (CLIP matches), and 'images' (list of image URLs)
    """
    # Generate CLIP embedding
    image_embedding = embed_image_clip(image)
    if not image_embedding:
        return {
            "response": "I encountered an error processing your image.",
            "sources": [],
            "images": []
        }
    
    # Search image embeddings (CLIP similarity)
    clip_matches = search_image_embeddings(image_embedding, top_k=top_k)
    
    if not clip_matches:
        return {
            "response": "I couldn't find any visually similar images.",
            "sources": [],
            "images": []
        }
    
    # Format response with image URLs
    images = []
    image_info = []
    
    for match in clip_matches:
        image_url = match.get("image_url")
        if image_url:
            images.append(image_url)
            image_info.append({
                "url": image_url,
                "project_key": match.get("project_key"),
                "page_num": match.get("page_num"),
                "similarity": match.get("similarity", 0),
                "search_type": "CLIP_visual"
            })
    
    # Generate summary response
    response_text = f"I found {len(images)} visually similar image(s):\n\n"
    for i, info in enumerate(image_info, 1):
        response_text += f"[{i}] Project {info['project_key']}, Page {info['page_num']}"
        if info.get('similarity'):
            response_text += f" (similarity: {info['similarity']:.3f})"
        response_text += "\n"
    
    return {
        "response": response_text,
        "sources": clip_matches,
        "images": images,
        "image_info": image_info,
        "search_type": "CLIP_visual"
    }


def route_image_to_text(
    image: Image.Image,
    text_query: Optional[str] = None,
    conversation_history: List[Dict[str, str]] = None,
    top_k: int = 3
) -> Dict[str, Any]:
    """
    Route: Image Upload → Text Response
    
    Uses dual path:
    1. CLIP embedding → search image_embeddings (visual similarity)
    2. GPT-4o Vision description → embed → search image_descriptions (semantic similarity)
    
    Args:
        image: User's uploaded image
        text_query: Optional additional text query
        conversation_history: Previous conversation messages
        top_k: Number of results per path
    
    Returns:
        Dict with 'response' (text), 'sources' (descriptions), and 'images' (list)
    """
    # Path 1: CLIP visual similarity
    image_embedding = embed_image_clip(image)
    clip_matches = []
    if image_embedding:
        clip_matches = search_image_embeddings(image_embedding, top_k=top_k)
    
    # Path 2: GPT-4o Vision description → text embedding
    vision_description = describe_image_with_gpt4o(image)
    text_matches = []
    if vision_description:
        # Embed the description
        desc_embedding = embed_text_openai(vision_description)
        if desc_embedding:
            text_matches = search_text_embeddings(desc_embedding, top_k=top_k, use_summary=True)
    
    # Combine results (keep separate for debugging as requested)
    all_descriptions = []
    
    # Add CLIP matches (convert to descriptions if possible)
    for clip_match in clip_matches:
        project_key = clip_match.get("project_key")
        page_num = clip_match.get("page_num")
        image_url = clip_match.get("image_url")
        
        # Try to get description for this image
        if project_key and image_url:
            # Extract relative_path from image_url
            # URL format: .../test_embeddings/{project_key}/{relative_path}
            if f"test_embeddings/{project_key}/" in image_url:
                rel_path = image_url.split(f"test_embeddings/{project_key}/")[-1]
                descs = get_image_descriptions_by_paths(project_key, [rel_path])
                if descs:
                    descs[0]["search_type"] = "CLIP_visual"
                    all_descriptions.extend(descs)
    
    # Add text matches
    for text_match in text_matches:
        text_match["search_type"] = "GPT4o_Vision_text"
        all_descriptions.append(text_match)
    
    # Remove duplicates (by project_key + page_num + region_number)
    seen = set()
    unique_descriptions = []
    for desc in all_descriptions:
        key = (desc.get("project_key"), desc.get("page_num"), desc.get("region_number"))
        if key not in seen:
            seen.add(key)
            unique_descriptions.append(desc)
    
    if not unique_descriptions:
        return {
            "response": "I couldn't find any relevant information for your image. Please try uploading a different image or adding a text query.",
            "sources": [],
            "images": []
        }
    
    # Generate response
    query = text_query if text_query else "What information is available about this image?"
    response_text = generate_text_response(query, unique_descriptions[:top_k], conversation_history)
    
    # Get image URLs
    images = []
    for desc in unique_descriptions[:top_k]:
        project_key = desc.get("project_key")
        relative_path = desc.get("relative_path")
        if project_key and relative_path:
            image_url = construct_image_url(project_key, relative_path)
            images.append(image_url)
    
    return {
        "response": response_text,
        "sources": unique_descriptions[:top_k],
        "images": images,
        "clip_matches": clip_matches,
        "text_matches": text_matches
    }


def orchestrate_query(
    text_query: Optional[str] = None,
    image: Optional[Image.Image] = None,
    conversation_history: List[Dict[str, str]] = None,
    top_k: int = 3
) -> Dict[str, Any]:
    """
    Main orchestrator function that routes queries to appropriate handlers
    
    Args:
        text_query: Optional text input
        image: Optional image input
        conversation_history: Previous conversation messages
        top_k: Number of results to return
    
    Returns:
        Dict with response, sources, images, and metadata
    """
    has_text = text_query and text_query.strip()
    has_image = image is not None
    
    if not has_text and not has_image:
        return {
            "response": "Please provide either a text query or upload an image.",
            "sources": [],
            "images": []
        }
    
    # Classify query type
    query_type = classify_query(has_text, has_image, text_query if has_text else None)
    
    # Route to appropriate handler
    if query_type == QueryType.TEXT_TO_TEXT:
        return route_text_to_text(text_query, conversation_history, top_k)
    
    elif query_type == QueryType.TEXT_TO_IMAGES:
        return route_text_to_images(text_query, top_k)
    
    elif query_type == QueryType.IMAGE_TO_IMAGES:
        return route_image_to_images(image, top_k)
    
    elif query_type == QueryType.IMAGE_TO_TEXT:
        return route_image_to_text(image, text_query, conversation_history, top_k)
    
    else:
        return {
            "response": "Unknown query type. Please try again.",
            "sources": [],
            "images": []
        }
