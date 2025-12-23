#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert Structured JSON Files to Supabase Table

This script reads structured JSON files from extract_structured_info.py and:
1. Converts array fields to comma-separated strings
2. Generates text embeddings for text_verbatim and summary using text-embedding-3-small
3. Inserts data into Supabase image_descriptions table
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from openai import OpenAI
from supabase import create_client, Client
from tqdm import tqdm
import time

# Directories
BASE_DIR = r"C:\Users\brian\OneDrive\Desktop\dataprocessing"

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    # Load from parent directory where .env file is located
    env_path = Path(BASE_DIR) / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✅ Loaded environment variables from {env_path}")
    else:
        # Fallback: try to find .env in current or parent directories
        load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, will use environment variables directly

# ================= CONFIGURATION =================
# OpenAI API Key for embeddings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY_HERE")  # Set OPENAI_API_KEY environment variable or replace with your key

# Supabase Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://nxrhvostwdtixojqyvro.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")  # UPDATE THIS with your service role key
STRUCTURED_JSON_DIR = os.path.join(BASE_DIR, "florence_embedding", "structured_json")

# Batch settings
EMBEDDING_BATCH_SIZE = 100  # Process embeddings in batches
INSERT_BATCH_SIZE = 50  # Insert records in batches
# =================================================

# Initialize clients
openai_client: Optional[OpenAI] = None
supabase: Optional[Client] = None


def init_clients():
    """Initialize OpenAI and Supabase clients"""
    global openai_client, supabase
    
    if not OPENAI_API_KEY or len(OPENAI_API_KEY.strip()) < 20:
        raise ValueError("OpenAI API key not set or invalid!")
    
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    
    if not SUPABASE_KEY:
        raise ValueError("Supabase service role key not set! Set SUPABASE_KEY environment variable.")
    
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("✅ Clients initialized")


def array_to_comma_separated(arr: List[Any]) -> Optional[str]:
    """Convert array to comma-separated string, return None if empty"""
    if not arr or len(arr) == 0:
        return None
    return ", ".join(str(item) for item in arr if item is not None)


def generate_text_embedding(text: str) -> Optional[List[float]]:
    """Generate embedding for text using text-embedding-3-small"""
    if not text or not text.strip():
        return None
    
    try:
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text.strip()
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"      ⚠️ Error generating embedding: {e}")
        return None


def generate_embeddings_batch(texts: List[str]) -> List[Optional[List[float]]]:
    """Generate embeddings for a batch of texts"""
    if not texts:
        return []
    
    # Filter out empty texts and keep track of indices
    valid_texts = []
    valid_indices = []
    for i, text in enumerate(texts):
        if text and text.strip():
            valid_texts.append(text.strip())
            valid_indices.append(i)
    
    if not valid_texts:
        return [None] * len(texts)
    
    try:
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=valid_texts
        )
        
        # Map embeddings back to original positions
        embeddings = [None] * len(texts)
        for idx, embedding_data in enumerate(response.data):
            original_idx = valid_indices[idx]
            embeddings[original_idx] = embedding_data.embedding
        
        return embeddings
    except Exception as e:
        print(f"      ⚠️ Error generating batch embeddings: {e}")
        return [None] * len(texts)


def load_structured_json(project_id: str) -> Dict[str, Any]:
    """Load structured JSON for a project"""
    json_file = Path(STRUCTURED_JSON_DIR) / project_id / f"structured_{project_id}.json"
    
    if not json_file.exists():
        raise FileNotFoundError(f"Structured JSON not found: {json_file}")
    
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def convert_image_to_row(img: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a single image JSON object to Supabase row format"""
    row = {
        "project_key": img.get("project_id"),
        "page_num": img.get("page_number"),
        "region_number": img.get("region_number"),
        "image_id": img.get("image_id"),
        "relative_path": img.get("relative_path"),
        "classification": img.get("classification"),
        "location": img.get("location"),
        "level": img.get("level"),
        "orientation": img.get("orientation"),
        "element_type": img.get("element_type"),
        "text_verbatim": img.get("text_verbatim"),
        "summary": img.get("summary"),
        # Convert arrays to comma-separated strings
        "grid_references": array_to_comma_separated(img.get("grid_references", [])),
        "section_callouts": array_to_comma_separated(img.get("section_callouts", [])),
        "element_callouts": array_to_comma_separated(img.get("element_callouts", [])),
        "key_components": array_to_comma_separated(img.get("key_components", [])),
    }
    
    # Remove None values (Supabase will use NULL)
    row = {k: v for k, v in row.items() if v is not None}
    
    return row


def process_project(project_id: str, skip_existing: bool = True) -> int:
    """Process a single project and return number of records processed"""
    print(f"\n--- Processing Project: {project_id} ---")
    
    # Load structured JSON
    try:
        structured_data = load_structured_json(project_id)
        images = structured_data.get("images", [])
        print(f"   📂 Loaded {len(images)} image descriptions")
    except Exception as e:
        print(f"   ❌ Error loading JSON: {e}")
        return 0
    
    if not images:
        print(f"   ⚠️ No images found")
        return 0
    
    # Check existing records if skip_existing is True
    existing_keys = set()
    if skip_existing:
        try:
            response = supabase.table("image_descriptions").select("project_key, page_num, region_number").eq("project_key", project_id).execute()
            existing_keys = {(r["project_key"], r["page_num"], r.get("region_number")) for r in response.data}
            print(f"   📋 Found {len(existing_keys)} existing records")
        except Exception as e:
            print(f"   ⚠️ Could not check existing records: {e}")
    
    # Convert images to rows
    rows_to_process = []
    for img in images:
        project_key = img.get("project_id")
        page_num = img.get("page_number")
        region_num = img.get("region_number")
        
        # Skip if already exists
        if skip_existing and (project_key, page_num, region_num) in existing_keys:
            continue
        
        row = convert_image_to_row(img)
        rows_to_process.append(row)
    
    if not rows_to_process:
        print(f"   ✅ All images already processed")
        return 0
    
    print(f"   ⚙️ Processing {len(rows_to_process)} new images...")
    
    # Generate embeddings in batches
    print(f"   🔄 Generating embeddings...")
    text_verbatim_list = [row.get("text_verbatim", "") for row in rows_to_process]
    summary_list = [row.get("summary", "") for row in rows_to_process]
    
    # Generate embeddings in batches
    verbatim_embeddings = []
    summary_embeddings = []
    
    for i in range(0, len(rows_to_process), EMBEDDING_BATCH_SIZE):
        batch = rows_to_process[i:i + EMBEDDING_BATCH_SIZE]
        batch_verbatim = text_verbatim_list[i:i + EMBEDDING_BATCH_SIZE]
        batch_summary = summary_list[i:i + EMBEDDING_BATCH_SIZE]
        
        print(f"      Processing embedding batch {i // EMBEDDING_BATCH_SIZE + 1}/{(len(rows_to_process) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE}...")
        
        # Generate embeddings
        verbatim_batch = generate_embeddings_batch(batch_verbatim)
        summary_batch = generate_embeddings_batch(batch_summary)
        
        verbatim_embeddings.extend(verbatim_batch)
        summary_embeddings.extend(summary_batch)
        
        # Rate limiting
        time.sleep(0.1)
    
    # Add embeddings to rows
    for i, row in enumerate(rows_to_process):
        if verbatim_embeddings[i]:
            row["text_verbatim_embedding"] = verbatim_embeddings[i]
        if summary_embeddings[i]:
            row["summary_embedding"] = summary_embeddings[i]
    
    # Insert into Supabase in batches
    print(f"   💾 Inserting into Supabase...")
    total_inserted = 0
    
    for i in range(0, len(rows_to_process), INSERT_BATCH_SIZE):
        batch = rows_to_process[i:i + INSERT_BATCH_SIZE]
        batch_num = i // INSERT_BATCH_SIZE + 1
        total_batches = (len(rows_to_process) + INSERT_BATCH_SIZE - 1) // INSERT_BATCH_SIZE
        
        try:
            response = supabase.table("image_descriptions").insert(batch).execute()
            inserted_count = len(response.data) if response.data else len(batch)
            total_inserted += inserted_count
            print(f"      ✅ Batch {batch_num}/{total_batches}: Inserted {inserted_count} records")
        except Exception as e:
            print(f"      ❌ Batch {batch_num}/{total_batches} failed: {e}")
            # Try inserting one by one to identify problematic records
            for j, row in enumerate(batch):
                try:
                    supabase.table("image_descriptions").insert(row).execute()
                    total_inserted += 1
                except Exception as single_error:
                    print(f"         ⚠️ Failed to insert row {i+j}: {single_error}")
                    print(f"         Row data: {json.dumps(row, indent=2)[:500]}")
    
    print(f"   ✅ Complete! Inserted {total_inserted} records")
    return total_inserted


def main():
    """Main entry point"""
    import sys
    
    print("="*60)
    print("Structured JSON to Supabase Converter")
    print("="*60)
    
    # Initialize clients
    try:
        init_clients()
    except Exception as e:
        print(f"❌ Error initializing clients: {e}")
        return
    
    # Check if project ID provided as argument
    if len(sys.argv) > 1:
        project_id = sys.argv[1]
        skip_existing = "--force" not in sys.argv
        process_project(project_id, skip_existing=skip_existing)
    else:
        # Process all projects
        structured_json_path = Path(STRUCTURED_JSON_DIR)
        
        if not structured_json_path.exists():
            print(f"❌ Structured JSON directory not found: {structured_json_path}")
            return
        
        # Find all project directories
        projects = sorted([d.name for d in structured_json_path.iterdir() if d.is_dir()])
        
        if not projects:
            print(f"❌ No project directories found in {structured_json_path}")
            return
        
        print(f"Found {len(projects)} project(s): {', '.join(projects)}")
        print()
        
        total_inserted = 0
        for project_id in projects:
            try:
                count = process_project(project_id, skip_existing=True)
                total_inserted += count
            except Exception as e:
                print(f"❌ Error processing {project_id}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print("\n" + "="*60)
        print(f"✨ Processing Complete! Total records inserted: {total_inserted}")
        print("="*60)


if __name__ == "__main__":
    main()

