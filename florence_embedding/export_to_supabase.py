#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export Structured Information to Supabase Format

This script reads the structured JSON files and exports them to formats suitable
for Supabase import (CSV or JSONL).

The output format includes all fields optimized for Supabase columns:
- image_id
- classification
- location
- section_callouts (as JSON array string)
- element_type
- text_verbatim
- summary
- page_number
- region_number
- relative_path
- project_id
"""

import os
import json
import csv
from pathlib import Path
from typing import List, Dict, Any

# ================= CONFIGURATION =================
BASE_DIR = r"C:\Users\brian\OneDrive\Desktop\dataprocessing"
STRUCTURED_JSON_DIR = os.path.join(BASE_DIR, "florence_embedding", "structured_json")
OUTPUT_DIR = os.path.join(BASE_DIR, "florence_embedding", "supabase_export")
# =================================================


def load_structured_json(project_id: str) -> List[Dict[str, Any]]:
    """Load structured JSON for a project"""
    json_file = Path(STRUCTURED_JSON_DIR) / project_id / f"structured_{project_id}.json"
    
    if not json_file.exists():
        return []
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            images = data.get("images", [])
            
            # Add project_id to each image
            for img in images:
                img["project_id"] = project_id
            
            return images
    except Exception as e:
        print(f"  ❌ Error loading {json_file}: {e}")
        return []


def export_to_csv(images: List[Dict[str, Any]], output_path: Path):
    """Export images to CSV format"""
    if not images:
        return
    
    # Define CSV columns
    fieldnames = [
        "project_id",
        "image_id",
        "relative_path",
        "page_number",
        "region_number",
        "classification",
        "location",
        "section_callouts",
        "element_type",
        "text_verbatim",
        "summary"
    ]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for img in images:
            row = {
                "project_id": img.get("project_id", ""),
                "image_id": img.get("image_id", ""),
                "relative_path": img.get("relative_path", ""),
                "page_number": img.get("page_number", ""),
                "region_number": img.get("region_number", ""),
                "classification": img.get("classification", ""),
                "location": img.get("location", ""),
                "section_callouts": json.dumps(img.get("section_callouts", [])) if isinstance(img.get("section_callouts"), list) else img.get("section_callouts", ""),
                "element_type": img.get("element_type", ""),
                "text_verbatim": img.get("text_verbatim", ""),
                "summary": img.get("summary", "")
            }
            writer.writerow(row)


def export_to_jsonl(images: List[Dict[str, Any]], output_path: Path):
    """Export images to JSONL format (one JSON object per line)"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for img in images:
            # Create clean record with only relevant fields
            record = {
                "project_id": img.get("project_id", ""),
                "image_id": img.get("image_id", ""),
                "relative_path": img.get("relative_path", ""),
                "page_number": img.get("page_number"),
                "region_number": img.get("region_number"),
                "classification": img.get("classification", ""),
                "location": img.get("location"),
                "section_callouts": img.get("section_callouts", []),
                "element_type": img.get("element_type", ""),
                "text_verbatim": img.get("text_verbatim", ""),
                "summary": img.get("summary", "")
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def export_project(project_id: str, formats: List[str] = ["csv", "jsonl"]):
    """Export a single project"""
    print(f"\n--- Exporting Project: {project_id} ---")
    
    images = load_structured_json(project_id)
    
    if not images:
        print(f"   ⚠️ No structured data found for {project_id}")
        return
    
    print(f"   📊 Found {len(images)} images")
    
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if "csv" in formats:
        csv_path = output_dir / f"{project_id}_export.csv"
        export_to_csv(images, csv_path)
        print(f"   ✅ CSV exported: {csv_path}")
    
    if "jsonl" in formats:
        jsonl_path = output_dir / f"{project_id}_export.jsonl"
        export_to_jsonl(images, jsonl_path)
        print(f"   ✅ JSONL exported: {jsonl_path}")


def export_all(formats: List[str] = ["csv", "jsonl"]):
    """Export all projects"""
    structured_dir = Path(STRUCTURED_JSON_DIR)
    
    if not structured_dir.exists():
        print(f"❌ Structured JSON directory not found: {structured_dir}")
        return
    
    projects = sorted([d.name for d in structured_dir.iterdir() if d.is_dir()])
    
    if not projects:
        print(f"❌ No projects found in {structured_dir}")
        return
    
    print(f"Found {len(projects)} project(s) to export")
    
    for project_id in projects:
        try:
            export_project(project_id, formats)
        except Exception as e:
            print(f"❌ Error exporting {project_id}: {e}")
            import traceback
            traceback.print_exc()
            continue


def main():
    """Main entry point"""
    import sys
    
    print("="*60)
    print("Export Structured Data to Supabase Format")
    print("="*60)
    
    if len(sys.argv) > 1:
        project_id = sys.argv[1]
        export_project(project_id)
    else:
        export_all()
    
    print("\n" + "="*60)
    print("✨ Export Complete!")
    print("="*60)


if __name__ == "__main__":
    main()

