# JSON to Supabase Converter

This script converts structured JSON files from `extract_structured_info.py` into a Supabase table with text embeddings.

## Setup

### 1. Install Dependencies

```bash
pip install openai supabase tqdm
```

### 2. Set Environment Variables

Set your Supabase service role key (required for inserts):

```powershell
$env:SUPABASE_KEY = "your-service-role-key-here"
```

Or set in the script directly (line 20).

### 3. Create the Table in Supabase

Run the SQL script in your Supabase SQL editor:

```sql
-- Run: create_image_descriptions_table.sql
```

This creates the `image_descriptions` table with:
- All structured fields as columns
- Text embeddings for `text_verbatim` and `summary` (1536-dim vectors)
- Indexes for fast filtering and vector similarity search

## Usage

### Process a Single Project

```powershell
python json_to_supabase.py 25-01-006
```

### Process All Projects

```powershell
python json_to_supabase.py
```

This will:
1. Find all projects in `structured_json` directory
2. Skip existing records (by project_key + page_num + region_number)
3. Generate embeddings for `text_verbatim` and `summary` using text-embedding-3-small
4. Insert records in batches

### Force Re-process (Skip Existing Check)

```powershell
python json_to_supabase.py 25-01-006 --force
```

## Table Schema

The `image_descriptions` table includes:

**Metadata:**
- `project_key`, `page_num`, `region_number`, `image_id`, `relative_path`

**Classification:**
- `classification`, `location`, `level`, `orientation`, `element_type`

**Callouts (comma-separated):**
- `grid_references`, `section_callouts`, `element_callouts`, `key_components`

**Text Content:**
- `text_verbatim` (full text)
- `summary` (rich summary)
- `text_verbatim_embedding` (vector)
- `summary_embedding` (vector)

## Query Examples

### Semantic Search on Descriptions

```sql
-- Find images by text similarity
SELECT 
  image_id,
  location,
  summary,
  1 - (summary_embedding <=> query_embedding) as similarity
FROM image_descriptions
ORDER BY summary_embedding <=> query_embedding
LIMIT 10;
```

### Filter + Semantic Search

```sql
-- Find foundation details about rebar
SELECT 
  image_id,
  location,
  summary
FROM image_descriptions
WHERE level = 'Foundation'
  AND classification = 'Detail'
ORDER BY summary_embedding <=> query_embedding
LIMIT 10;
```

### Join with image_embeddings

```sql
-- Get both visual and text embeddings
SELECT 
  id.image_id,
  id.summary,
  id.location,
  ie.image_url,
  ie.embedding as visual_embedding,
  id.summary_embedding as text_embedding
FROM image_descriptions id
JOIN image_embeddings ie 
  ON id.project_key = ie.project_key 
  AND id.page_num = ie.page_num
WHERE id.project_key = '25-01-006';
```

