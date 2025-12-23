#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT-4o Utilities for Vision and Text Generation
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from PIL import Image
import base64
from io import BytesIO

# Load environment variables
try:
    from dotenv import load_dotenv
    BASE_DIR = Path(r"C:\Users\brian\OneDrive\Desktop\dataprocessing")
    env_path = BASE_DIR / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Check for OPENAI_API_KEY first, fallback to API_KEY for compatibility
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY", "")
_openai_client = None


def get_openai_client():
    """Initialize OpenAI client"""
    global _openai_client
    if _openai_client is None:
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set in environment variables")
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


def describe_image_with_gpt4o(image: Image.Image) -> Optional[str]:
    """
    Use GPT-4o Vision to describe an uploaded image
    
    Args:
        image: PIL Image
    
    Returns:
        Text description of the image or None if error
    """
    try:
        if not OPENAI_AVAILABLE:
            raise ImportError("openai not available")
        
        client = get_openai_client()
        
        # Convert image to base64
        img_base64 = image_to_base64(image)
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this engineering drawing image in detail. Include all visible text, dimensions, structural elements, and annotations. Focus on technical details that would be useful for searching similar drawings."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        
        description = response.choices[0].message.content
        return description
    
    except Exception as e:
        print(f"Error describing image with GPT-4o: {e}")
        return None


def generate_text_response(
    user_query: str,
    context_descriptions: List[Dict[str, Any]],
    conversation_history: List[Dict[str, str]] = None
) -> str:
    """
    Generate natural language response using GPT-4o based on retrieved context
    
    Args:
        user_query: User's question
        context_descriptions: List of image descriptions with text_verbatim and summary
        conversation_history: Previous conversation messages
    
    Returns:
        Generated text response with citations
    """
    try:
        if not OPENAI_AVAILABLE:
            raise ImportError("openai not available")
        
        client = get_openai_client()
        
        # Build context from descriptions
        context_text = ""
        citations = []
        
        for i, desc in enumerate(context_descriptions, 1):
            project_key = desc.get("project_key", "Unknown")
            page_num = desc.get("page_num", "Unknown")
            region_num = desc.get("region_number")
            
            citation_ref = f"[{i}]"
            citations.append({
                "ref": citation_ref,
                "project_key": project_key,
                "page_num": page_num,
                "region_num": region_num
            })
            
            context_text += f"\n\n{citation_ref} Project {project_key}, Page {page_num}"
            if region_num:
                context_text += f", Region {region_num}"
            context_text += ":\n"
            
            if desc.get("summary"):
                context_text += f"Summary: {desc['summary']}\n"
            if desc.get("text_verbatim"):
                context_text += f"Text: {desc['text_verbatim'][:500]}...\n"
        
        # Build conversation messages
        messages = []
        
        # System message
        messages.append({
            "role": "system",
            "content": """You are a helpful assistant that answers questions about engineering drawings.
Always cite your sources using inline citations like [1], [2], etc.
Only use information from the provided context. If the context doesn't contain the answer, say so clearly.
Do not hallucinate or make up information."""
        })
        
        # Add conversation history
        if conversation_history:
            messages.extend(conversation_history)
        
        # Add current query with context
        user_message = f"""User Question: {user_query}

Context from Engineering Drawings:
{context_text}

Please answer the user's question based on the context above. Always include inline citations like [1], [2] when referencing specific drawings.
If the context doesn't contain enough information to answer the question, say so clearly."""
        
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        # Generate response
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=1000,
            temperature=0.7
        )
        
        answer = response.choices[0].message.content
        
        # Add citation details at the end
        if citations:
            answer += "\n\n**References:**\n"
            for cit in citations:
                answer += f"{cit['ref']} Project {cit['project_key']}, Page {cit['page_num']}"
                if cit.get('region_num'):
                    answer += f", Region {cit['region_num']}"
                answer += "\n"
        
        return answer
    
    except Exception as e:
        print(f"Error generating text response: {e}")
        return f"I encountered an error generating a response: {str(e)}"
