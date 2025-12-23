#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit Chat App for Engineering Drawing Search
Unified chat interface supporting text and image inputs/outputs
"""

import streamlit as st
from PIL import Image
from typing import List, Dict, Any
import time

from langgraph_orchestrator import orchestrate_query

# Page config
st.set_page_config(
    page_title="Engineering Drawing Assistant",
    page_icon="🏗️",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []


def format_conversation_history() -> List[Dict[str, str]]:
    """Format conversation history for GPT-4o"""
    history = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            history.append({"role": "user", "content": msg.get("content", "")})
        elif msg["role"] == "assistant":
            history.append({"role": "assistant", "content": msg.get("content", "")})
    return history


def display_images(image_urls: List[str], image_info: List[Dict] = None):
    """Display images inline in chat"""
    if not image_urls:
        return
    
    # Display images in a grid
    cols = st.columns(min(3, len(image_urls)))
    
    for i, img_url in enumerate(image_urls):
        with cols[i % len(cols)]:
            try:
                st.image(img_url, width='stretch')
                
                # Show metadata if available
                if image_info and i < len(image_info):
                    info = image_info[i]
                    metadata_text = f"Project {info.get('project_key', 'N/A')}, Page {info.get('page_num', 'N/A')}"
                    if info.get('region_number'):
                        metadata_text += f", Region {info['region_number']}"
                    st.caption(metadata_text)
            except Exception as e:
                st.error(f"Error loading image: {e}")
                st.text(f"URL: {img_url}")


def main():
    st.title("🏗️ Engineering Drawing Assistant")
    st.markdown("Ask questions or upload images to search through engineering drawings. Supports text and image inputs/outputs.")
    
    # Sidebar for settings
    with st.sidebar:
        st.header("Settings")
        top_k = st.slider("Number of results", 1, 10, 3)
        
        st.markdown("---")
        st.subheader("Image Processing Method")
        image_method = st.radio(
            "Choose method for image similarity search:",
            ["CLIP (Visual Similarity)", "GPT-4o Vision (Semantic Similarity)"],
            index=0,
            help="CLIP: Direct visual similarity using image embeddings. GPT-4o Vision: Describes image then searches text embeddings."
        )
        # Convert to simple format for backend
        image_method_short = "clip" if image_method == "CLIP (Visual Similarity)" else "gpt4o"
        
        st.markdown("---")
        st.markdown("**Features:**")
        st.markdown("- 📝 Text queries → Text responses")
        st.markdown("- 📝 Text queries → Image results")
        st.markdown("- 🖼️ Image upload → Similar images")
        st.markdown("- 🖼️ Image upload → Text responses")
        st.markdown("---")
        if st.button("Clear Conversation"):
            st.session_state.messages = []
            st.session_state.conversation_history = []
            st.rerun()
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # Display text content
            if message.get("content"):
                st.markdown(message["content"])
            
            # Display images if any
            if message.get("images"):
                display_images(
                    message["images"],
                    message.get("image_info")
                )
            
            # Display sources in expander for debugging
            message_idx = st.session_state.messages.index(message)
            if message.get("sources") and st.checkbox(f"Show sources (debug)", key=f"debug_{message_idx}"):
                with st.expander("Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.json(source)
    
    # Chat input
    user_input = st.chat_input("Ask a question or upload an image...")
    
    # File uploader for images (in sidebar to avoid conflicts)
    with st.sidebar:
        uploaded_file = st.file_uploader(
            "Upload an image",
            type=['png', 'jpg', 'jpeg'],
            key="image_uploader"
        )
    
    # Track if we need to process uploaded file
    process_upload = False
    if uploaded_file:
        # Check if this file was already processed
        file_id = id(uploaded_file)
        if "last_processed_file" not in st.session_state or st.session_state.last_processed_file != file_id:
            process_upload = True
            st.session_state.last_processed_file = file_id
    
    # Process user input
    if user_input or process_upload:
        # Add user message to history
        if user_input:
            user_content = user_input
        elif uploaded_file:
            user_content = "Uploaded image"
        else:
            user_content = ""
        
        if user_content:  # Only process if we have content
            st.session_state.messages.append({
                "role": "user",
                "content": user_content
            })
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(user_content)
                if uploaded_file:
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Uploaded Image", width=300)
            
            # Process query
            with st.chat_message("assistant"):
                with st.spinner("Processing your query..."):
                    # Get conversation history
                    conversation_history = format_conversation_history()[:-1]  # Exclude current message
                    
                    # Process image if uploaded
                    image_obj = None
                    if uploaded_file:
                        image_obj = Image.open(uploaded_file)
                    
                    # Orchestrate query
                    try:
                        result = orchestrate_query(
                            text_query=user_input if user_input else None,
                            image=image_obj,
                            conversation_history=conversation_history,
                            top_k=top_k,
                            image_method=image_method_short
                        )
                        
                        # Display response
                        response_text = result.get("response", "No response generated.")
                        st.markdown(response_text)
                        
                        # Display images if any
                        images = result.get("images", [])
                        image_info = result.get("image_info", [])
                        
                        if images:
                            st.markdown("**Relevant Images:**")
                            display_images(images, image_info)
                        
                        # Store assistant message
                        assistant_message = {
                            "role": "assistant",
                            "content": response_text,
                            "images": images,
                            "image_info": image_info,
                            "sources": result.get("sources", [])
                        }
                        
                        st.session_state.messages.append(assistant_message)
                        
                        # Update conversation history for next turn
                        st.session_state.conversation_history.append({
                            "role": "user",
                            "content": user_content
                        })
                        st.session_state.conversation_history.append({
                            "role": "assistant",
                            "content": response_text
                        })
                    except Exception as e:
                        error_msg = f"I encountered an error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg,
                            "images": [],
                            "sources": []
                        })


if __name__ == "__main__":
    main()
