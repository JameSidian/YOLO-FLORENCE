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
            # Keywords that indicate user wants images
            image_keywords = [
                "show", "image", "picture", "drawing", "screenshot", 
                "retrieve", "get me", "find me", "display", "see",
                "detail", "details", "plan", "section",
                "elevation", "view", "visual", "illustration",
                "retrieve me", "show me", "get me a", "find me a"
            ]
            # Check for phrases first (more specific)
            image_phrases = ["retrieve me", "show me", "get me a", "find me a", "display the", "see the"]
            if any(phrase in intent_lower for phrase in image_phrases):
                return QueryType.TEXT_TO_IMAGES
            # Then check for individual keywords
            if any(word in intent_lower for word in image_keywords):
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
    top_k: int = 3,
    image_method: str = "clip"
) -> Dict[str, Any]:
    """
    Route: Image Upload → Similar Images
    
    Args:
        image: User's uploaded image
        top_k: Number of similar images to return
        image_method: "clip" for visual similarity or "gpt4o" for semantic similarity
    
    Returns:
        Dict with 'response' (text), 'sources' (matches), and 'images' (list of image URLs)
    """
    # Use user's preferred method
    if image_method == "gpt4o":
        # Use GPT-4o Vision + text search directly
        clip_matches = []
        text_matches = []
        try:
            # Use GPT-4o Vision to describe the image
            vision_description = describe_image_with_gpt4o(image)
            if vision_description:
                # Embed the description and search text embeddings
                desc_embedding = embed_text_openai(vision_description)
                if desc_embedding:
                    text_matches = search_text_embeddings(desc_embedding, top_k=top_k, use_summary=True)
        except Exception as e:
            import traceback
            print(f"GPT-4o Vision error: {str(e)}")
            traceback.print_exc()
        
        if text_matches:
            # Convert text matches to image format
            images = []
            image_info = []
            
            for match in text_matches:
                project_key = match.get("project_key")
                relative_path = match.get("relative_path")
                
                if project_key and relative_path:
                    image_url = construct_image_url(project_key, relative_path)
                    images.append(image_url)
                    image_info.append({
                        "url": image_url,
                        "project_key": project_key,
                        "page_num": match.get("page_num"),
                        "region_number": match.get("region_number"),
                        "similarity": match.get("similarity", 0),
                        "search_type": "GPT4o_semantic"
                    })
            
            if images:
                # Generate summary response
                response_text = f"I found {len(images)} semantically similar image(s) using image description:\n\n"
                for i, info in enumerate(image_info, 1):
                    response_text += f"[{i}] Project {info['project_key']}, Page {info['page_num']}"
                    if info.get('region_number'):
                        response_text += f", Region {info['region_number']}"
                    if info.get('similarity'):
                        response_text += f" (similarity: {info['similarity']:.3f})"
                    response_text += "\n"
                
                return {
                    "response": response_text,
                    "sources": text_matches,
                    "images": images,
                    "image_info": image_info,
                    "search_type": "GPT4o_semantic"
                }
        
        # No results found
        return {
            "response": "I couldn't find any semantically similar images. Please try a different image or use CLIP for visual similarity.",
            "sources": [],
            "images": []
        }
    
    # Default: Use CLIP (visual similarity)
    image_embedding = None
    clip_matches = []
    clip_error = None
    
    try:
        image_embedding = embed_image_clip(image)
        if image_embedding:
            clip_matches = search_image_embeddings(image_embedding, top_k=top_k)
    except Exception as e:
        clip_error = str(e)
        import traceback
        print(f"CLIP embedding error: {clip_error}")
        traceback.print_exc()
    
    # If we have CLIP matches, use them
    if clip_matches:
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
    
    # If CLIP failed (user chose CLIP but it didn't work)
    error_msg = "I couldn't find any visually similar images using CLIP."
    if clip_error:
        error_msg += f" Error: {clip_error}"
    error_msg += " Please try using GPT-4o Vision for semantic similarity or try a different image."
    
    return {
        "response": error_msg,
        "sources": [],
        "images": []
    }


def route_image_to_text(
    image: Image.Image,
    text_query: Optional[str] = None,
    conversation_history: List[Dict[str, str]] = None,
    top_k: int = 3,
    image_method: str = "clip"
) -> Dict[str, Any]:
    """
    Route: Image Upload → Text Response
    
    Uses selected method (CLIP or GPT-4o Vision) to find relevant descriptions
    
    Args:
        image: User's uploaded image
        text_query: Optional additional text query
        conversation_history: Previous conversation messages
        top_k: Number of results
        image_method: "clip" for visual similarity or "gpt4o" for semantic similarity
    
    Returns:
        Dict with 'response' (text), 'sources' (descriptions), and 'images' (list)
    """
    all_descriptions = []
    clip_matches = []
    text_matches = []
    
    if image_method == "gpt4o":
        # Use GPT-4o Vision description → text embedding
        vision_description = describe_image_with_gpt4o(image)
        if vision_description:
            # Embed the description
            desc_embedding = embed_text_openai(vision_description)
            if desc_embedding:
                text_matches = search_text_embeddings(desc_embedding, top_k=top_k, use_summary=True)
        
        # Add text matches
        for text_match in text_matches:
            text_match["search_type"] = "GPT4o_Vision_text"
            all_descriptions.append(text_match)
    else:
        # Use CLIP visual similarity
        image_embedding = None
        try:
            image_embedding = embed_image_clip(image)
            if image_embedding:
                clip_matches = search_image_embeddings(image_embedding, top_k=top_k)
        except Exception as e:
            # Log error but continue
            print(f"CLIP embedding failed: {e}")
            import traceback
            traceback.print_exc()
        
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
    top_k: int = 3,
    image_method: str = "clip"
) -> Dict[str, Any]:
    """
    Main orchestrator function that routes queries to appropriate handlers
    
    Args:
        text_query: Optional text input
        image: Optional image input
        conversation_history: Previous conversation messages
        top_k: Number of results to return
        image_method: "clip" for visual similarity or "gpt4o" for semantic similarity
    
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
        return route_image_to_images(image, top_k, image_method)
    
    elif query_type == QueryType.IMAGE_TO_TEXT:
        return route_image_to_text(image, text_query, conversation_history, top_k, image_method)
    
    else:
        return {
            "response": "Unknown query type. Please try again.",
            "sources": [],
            "images": []
        }
