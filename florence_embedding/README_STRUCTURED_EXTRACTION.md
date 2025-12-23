# Structured Information Extraction for Engineering Drawings

This pipeline extracts structured information from engineering drawing images for Supabase storage and semantic search.

## Overview

The pipeline processes region images (cropped sections) from engineering drawings and extracts:
- **Classification**: Plan, Elevation, Section, Detail, Notes, Schedule
- **Location**: North, East, Level 1, Level 2, etc.
- **Section Callouts**: Section cut markers like "A/S01" that link to details
- **Element Type**: Retaining wall, floating slab, main floor plan, etc.
- **Text Verbatim**: Exact text word-for-word from the image
- **Summary**: Rich summary explaining how text relates to linework (for embeddings)

## Folder Structure

```
dataprocessing/
├── test_embeddings/
│   └── 25-01-005/
│       ├── manifest.json
│       ├── page_001/
│       │   ├── region_01_red_box.png
│       │   ├── region_02_red_box.png
│       │   └── ...
│       └── page_002/
│           └── ...
├── florence_embedding/
│   ├── structured_json/
│   │   └── 25-01-005/
│   │       └── structured_25-01-005.json
│   └── supabase_export/
│       └── 25-01-005_export.csv
```

## Scripts

### 1. `extract_structured_info.py`

Main extraction script that processes images using GPT-4o vision.

**Usage:**
```bash
# Process a specific project
python extract_structured_info.py 25-01-005

# Process all projects in test_embeddings directory
python extract_structured_info.py
```

**Configuration:**
- Update `API_KEY` in the script with your OpenAI API key
- Adjust `BATCH_SIZE` (default: 5) to control how many images are processed per API call
- Modify `BASE_DIR` if your folder structure is different

**Features:**
- Automatically finds all region images in `page_XXX/` subdirectories
- Uses GPT-4o vision to extract all text and information directly from images
- Resumes from previous runs (skips already processed images)
- Saves incrementally after each batch

**Output:**
- Creates `structured_json/{project_id}/structured_{project_id}.json`
- Each image has: classification, location, section_callouts, element_type, text_verbatim, summary

### 2. `export_to_supabase.py`

Exports structured JSON to CSV/JSONL format for Supabase import.

**Usage:**
```bash
# Export a specific project
python export_to_supabase.py 25-01-005

# Export all projects
python export_to_supabase.py
```

**Output Formats:**
- CSV: `supabase_export/{project_id}_export.csv`
- JSONL: `supabase_export/{project_id}_export.jsonl`

## Output Schema

Each image record contains:

```json
{
  "image_id": "region_01_red_box.png",
  "classification": "Detail",
  "location": "North",
  "section_callouts": ["A/S01", "Section A-A"],
  "element_type": "Retaining Wall Detail",
  "text_verbatim": "7/16\" O.S.B. SHEATHING\n(1 OR 2 SIDES PER PLAN)\n6\"\n3\"\nANGLE FASTENED TO STUDS w/8 - 5/16\" x 3\" WOOD LAGS\n...",
  "summary": "This section view shows a retaining wall detail with foundation. The wall has a height of 5'-0\" as indicated by the vertical dimension line. Reinforcement consists of 15M rebar bars spaced at 10 inches on center (15M @10\" o/c) as shown in the rebar notation...",
  "page_number": 1,
  "region_number": 1,
  "relative_path": "page_001/region_01_red_box.png",
  "project_id": "25-01-005"
}
```

## Supabase Schema Recommendations

Create a table with these columns:

```sql
CREATE TABLE image_extractions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id TEXT NOT NULL,
  image_id TEXT NOT NULL,
  relative_path TEXT,
  page_number INTEGER,
  region_number INTEGER,
  classification TEXT,  -- Plan, Elevation, Section, Detail, Notes, Schedule
  location TEXT,        -- North, East, Level 1, etc.
  section_callouts JSONB,  -- Array of callout strings
  element_type TEXT,
  text_verbatim TEXT,
  summary TEXT,
  created_at TIMESTAMP DEFAULT NOW(),
  
  -- Indexes for fast filtering
  INDEX idx_project_id (project_id),
  INDEX idx_classification (classification),
  INDEX idx_location (location),
  INDEX idx_element_type (element_type),
  
  -- Full-text search on summary and text_verbatim
  INDEX idx_summary_search USING gin (to_tsvector('english', summary)),
  INDEX idx_text_search USING gin (to_tsvector('english', text_verbatim))
);
```

## Key Features

### Verbatim Text + Explanatory Summary

The summary incorporates verbatim text but explains relationships:
- **Verbatim**: "5'-0\"", "15M @10\" o/c"
- **Summary**: "The retaining wall has a height of 5'-0\" (five feet zero inches) as indicated by the dimension line. Reinforcement consists of 15M rebar bars spaced at 10 inches on center, as specified in the notation '15M @10\" o/c'."

### Fast Filtering + Semantic Search

- **Filtering**: Use `classification`, `location`, `element_type` for fast SQL queries
- **Semantic Search**: Use `summary` and `text_verbatim` fields for vector embeddings and full-text search

## Example Queries

```sql
-- Find all retaining wall details
SELECT * FROM image_extractions 
WHERE element_type ILIKE '%retaining wall%';

-- Find all details on Level 1
SELECT * FROM image_extractions 
WHERE location = 'Level 1' AND classification = 'Detail';

-- Semantic search for rebar information
SELECT * FROM image_extractions 
WHERE to_tsvector('english', summary || ' ' || text_verbatim) 
  @@ to_tsquery('rebar & spacing');

-- Find section callouts
SELECT * FROM image_extractions 
WHERE section_callouts @> '["A/S01"]'::jsonb;
```

## Notes

- The script processes images in batches to optimize API usage
- GPT-4o vision extracts all text and information directly from the images
- The summary is designed to be rich for semantic search while incorporating all verbatim values
- All text is extracted verbatim, then explained in the summary

