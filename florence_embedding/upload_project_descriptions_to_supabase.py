#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Upload Project Descriptions to Supabase

This script reads building synthesis JSON files and:
1. Generates text embeddings for overall_building_description using text-embedding-3-small
2. Inserts/updates data into Supabase project_description table
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from openai import OpenAI
from supabase import create_client, Client
from tqdm import tqdm
import time

# Directories
BASE_DIR = r"C:\Users\brian\OneDrive\Desktop\dataprocessing"

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    env_path = Path(BASE_DIR) / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✅ Loaded environment variables from {env_path}")
    else:
        load_dotenv()
except ImportError:
    pass

# ================= CONFIGURATION =================
# OpenAI API Key for embeddings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY", "")

# Supabase Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://nxrhvostwdtixojqyvro.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

BUILDING_SYNTHESIS_DIR = os.path.join(BASE_DIR, "florence_embedding", "building_synthesis")
# =================================================

# Initialize clients
openai_client: Optional[OpenAI] = None
supabase: Optional[Client] = None


def init_clients():
    """Initialize OpenAI and Supabase clients"""
    global openai_client, supabase
    
    if not OPENAI_API_KEY or len(OPENAI_API_KEY.strip()) < 20:
        raise ValueError("OpenAI API key not set or invalid! Set OPENAI_API_KEY or API_KEY environment variable.")
    
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    
    if not SUPABASE_KEY:
        raise ValueError("Supabase service role key not set! Set SUPABASE_KEY environment variable.")
    
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("✅ Clients initialized")


def generate_text_embedding(text: str) -> Optional[list]:
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


def load_building_synthesis_json(json_file: Path) -> Optional[Dict[str, Any]]:
    """Load a building synthesis JSON file"""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"      ❌ Error loading {json_file.name}: {e}")
        return None


def prepare_record(data: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare a record for Supabase insertion"""
    # Generate embedding for overall_building_description
    description = data.get("overall_building_description", "")
    embedding = None
    
    if description:
        print(f"      🔄 Generating embedding for project {data.get('project_id')}...")
        embedding = generate_text_embedding(description)
        if embedding:
            print(f"      ✅ Embedding generated ({len(embedding)} dimensions)")
        else:
            print(f"      ⚠️ Failed to generate embedding")
    
    # Prepare record with all fields
    record = {
        "project_id": data.get("project_id"),
        "project_name": data.get("project_name"),
        "client": data.get("client"),
        "location": data.get("location"),
        "building_type": data.get("building_type"),
        "number_of_levels": data.get("number_of_levels"),
        "levels": data.get("levels"),
        "dimensions_length": data.get("dimensions_length"),
        "dimensions_width": data.get("dimensions_width"),
        "dimensions_height": data.get("dimensions_height"),
        "dimensions_area": data.get("dimensions_area"),
        "gravity_system": data.get("gravity_system"),
        "lateral_system": data.get("lateral_system"),
        "concrete_strengths": data.get("concrete_strengths"),
        "steel_shapes": data.get("steel_shapes"),
        "rebar_sizes": data.get("rebar_sizes"),
        "other_materials": data.get("other_materials"),
        "structural_beams": data.get("structural_beams"),
        "structural_columns": data.get("structural_columns"),
        "structural_trusses": data.get("structural_trusses"),
        "key_elements": data.get("key_elements"),
        "overall_building_description": description,
        "overall_building_description_embedding": embedding
    }
    
    # Remove None values (Supabase doesn't like them)
    record = {k: v for k, v in record.items() if v is not None}
    
    return record


def upload_project_description(record: Dict[str, Any]) -> bool:
    """Upload or update a project description in Supabase"""
    try:
        project_id = record.get("project_id")
        
        # Check if record already exists
        existing = supabase.table("project_description").select("project_id").eq("project_id", project_id).execute()
        
        if existing.data:
            # Update existing record
            result = supabase.table("project_description").update(record).eq("project_id", project_id).execute()
            print(f"      ✅ Updated project {project_id}")
        else:
            # Insert new record
            result = supabase.table("project_description").insert(record).execute()
            print(f"      ✅ Inserted project {project_id}")
        
        return True
        
    except Exception as e:
        print(f"      ❌ Error uploading project {record.get('project_id')}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point"""
    print("="*60)
    print("Upload Project Descriptions to Supabase")
    print("="*60)
    
    # Initialize clients
    try:
        init_clients()
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return
    
    # Find all building synthesis JSON files
    synthesis_dir = Path(BUILDING_SYNTHESIS_DIR)
    
    if not synthesis_dir.exists():
        print(f"❌ Building synthesis directory not found: {synthesis_dir}")
        return
    
    json_files = sorted(synthesis_dir.glob("building_info_*.json"))
    
    if not json_files:
        print(f"❌ No building synthesis JSON files found in {synthesis_dir}")
        return
    
    print(f"\n📂 Found {len(json_files)} building synthesis file(s)")
    print()
    
    # Process each file
    success_count = 0
    error_count = 0
    
    for json_file in tqdm(json_files, desc="Processing files"):
        print(f"\n--- Processing: {json_file.name} ---")
        
        # Load JSON data
        data = load_building_synthesis_json(json_file)
        if not data:
            error_count += 1
            continue
        
        project_id = data.get("project_id", "unknown")
        print(f"   Project ID: {project_id}")
        
        # Prepare record
        record = prepare_record(data)
        
        # Upload to Supabase
        if upload_project_description(record):
            success_count += 1
        else:
            error_count += 1
        
        # Small delay to avoid rate limiting
        time.sleep(0.5)
    
    # Summary
    print("\n" + "="*60)
    print("✨ Upload Complete!")
    print(f"   ✅ Successfully processed: {success_count}")
    print(f"   ❌ Errors: {error_count}")
    print("="*60)


if __name__ == "__main__":
    main()
