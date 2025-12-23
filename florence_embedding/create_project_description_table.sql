-- Create project_description table for Supabase
-- This table stores building-level synthesis information from engineering drawings
-- with text embeddings for semantic search

CREATE TABLE IF NOT EXISTS project_description (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  
  -- Project identification
  project_id TEXT NOT NULL UNIQUE,
  project_name TEXT,
  client TEXT,
  location TEXT,
  
  -- Building classification
  building_type TEXT, -- 'Agricultural Building', 'Residential Building', 'Commercial Building', 'Industrial Building', 'Mixed Use', or 'Unknown'
  number_of_levels INTEGER,
  levels TEXT, -- Comma-separated list of levels
  
  -- Dimensions
  dimensions_length TEXT, -- Format: 'XXX'-XX"'
  dimensions_width TEXT, -- Format: 'XXX'-XX"'
  dimensions_height TEXT,
  dimensions_area TEXT, -- Total area in sq ft
  
  -- Structural systems
  gravity_system TEXT, -- Comma-separated description of gravity load path
  lateral_system TEXT, -- Comma-separated list of lateral resisting elements
  
  -- Materials
  concrete_strengths TEXT, -- Comma-separated list
  steel_shapes TEXT, -- Comma-separated list
  rebar_sizes TEXT, -- Comma-separated list
  other_materials TEXT, -- Comma-separated list
  
  -- Structural members
  structural_beams TEXT, -- Comma-separated list
  structural_columns TEXT, -- Comma-separated list
  structural_trusses TEXT, -- Comma-separated list
  
  -- Key elements
  key_elements TEXT, -- Comma-separated list of key structural/architectural elements
  
  -- Comprehensive building description (500-800 words)
  overall_building_description TEXT,
  
  -- Text embedding for semantic search (using text-embedding-3-small)
  overall_building_description_embedding vector(1536),
  
  -- Timestamps
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_project_description_project_id ON project_description(project_id);
CREATE INDEX IF NOT EXISTS idx_project_description_building_type ON project_description(building_type);
CREATE INDEX IF NOT EXISTS idx_project_description_client ON project_description(client);

-- Vector similarity index (for embedding search)
CREATE INDEX IF NOT EXISTS idx_project_description_embedding ON project_description 
  USING ivfflat (overall_building_description_embedding vector_cosine_ops) WITH (lists = 100);

-- Full-text search index on overall_building_description
CREATE INDEX IF NOT EXISTS idx_project_description_search_text ON project_description 
  USING gin(to_tsvector('english', coalesce(overall_building_description, '')));

-- Comments for documentation
COMMENT ON TABLE project_description IS 'Building-level synthesis information from engineering drawings with text embeddings';
COMMENT ON COLUMN project_description.overall_building_description_embedding IS 'Embedding of overall_building_description using text-embedding-3-small (1536 dimensions)';
