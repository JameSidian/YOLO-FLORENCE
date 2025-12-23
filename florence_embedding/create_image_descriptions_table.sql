-- Create image_descriptions table for Supabase
-- This table stores structured descriptions from engineering drawings
-- with text embeddings for semantic search

CREATE TABLE IF NOT EXISTS image_descriptions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  
  -- Link to project and image
  project_key TEXT NOT NULL,
  page_num INTEGER NOT NULL,
  region_number INTEGER, -- null for full-page images
  image_id TEXT NOT NULL, -- filename like "region_01_red_box.png"
  relative_path TEXT, -- "page_002/region_01_red_box.png"
  
  -- Classification and location
  classification TEXT, -- 'Plan', 'Section', 'Detail', 'Elevation', 'Schedule', 'Notes'
  location TEXT, -- e.g., "Foundation – main barn area"
  level TEXT, -- 'Foundation', 'Ground Floor', 'Roof', etc.
  orientation TEXT, -- 'North elevation', 'Transverse section looking East', etc.
  element_type TEXT, -- e.g., "Foundation Plan", "Shearwall Detail"
  
  -- Callouts and references (comma-separated strings)
  grid_references TEXT, -- e.g., "Grid 1-5, Grids A-C"
  section_callouts TEXT, -- e.g., "1/S1.0, 2/S2.0"
  element_callouts TEXT, -- e.g., "F1, W1, B1"
  key_components TEXT, -- e.g., "Slab-on-Grade, Foundation Walls, Rebar"
  
  -- Text content
  text_verbatim TEXT, -- full verbatim text from image
  summary TEXT, -- rich summary for semantic search
  
  -- Text embeddings (using text-embedding-3-small)
  text_verbatim_embedding vector(1536), -- text-embedding-3-small produces 1536-dim vectors
  summary_embedding vector(1536),
  
  -- Timestamps
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_image_descriptions_project ON image_descriptions(project_key);
CREATE INDEX IF NOT EXISTS idx_image_descriptions_page ON image_descriptions(project_key, page_num);
CREATE INDEX IF NOT EXISTS idx_image_descriptions_level ON image_descriptions(level);
CREATE INDEX IF NOT EXISTS idx_image_descriptions_classification ON image_descriptions(classification);

-- Vector similarity indexes (for embedding search)
CREATE INDEX IF NOT EXISTS idx_image_descriptions_verbatim_embedding ON image_descriptions 
  USING ivfflat (text_verbatim_embedding vector_cosine_ops) WITH (lists = 100);
  
CREATE INDEX IF NOT EXISTS idx_image_descriptions_summary_embedding ON image_descriptions 
  USING ivfflat (summary_embedding vector_cosine_ops) WITH (lists = 100);

-- Full-text search index on summary and text_verbatim
CREATE INDEX IF NOT EXISTS idx_image_descriptions_search_text ON image_descriptions 
  USING gin(to_tsvector('english', coalesce(summary, '') || ' ' || coalesce(text_verbatim, '')));

-- Composite unique constraint to prevent duplicates
CREATE UNIQUE INDEX IF NOT EXISTS idx_image_descriptions_unique 
  ON image_descriptions(project_key, page_num, region_number);

-- Comments for documentation
COMMENT ON TABLE image_descriptions IS 'Structured descriptions of engineering drawing regions with text embeddings';
COMMENT ON COLUMN image_descriptions.text_verbatim_embedding IS 'Embedding of text_verbatim using text-embedding-3-small (1536 dimensions)';
COMMENT ON COLUMN image_descriptions.summary_embedding IS 'Embedding of summary using text-embedding-3-small (1536 dimensions)';

