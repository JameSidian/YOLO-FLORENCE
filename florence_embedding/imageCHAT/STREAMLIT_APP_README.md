# Engineering Drawing Chat Assistant

A unified Streamlit chat interface for searching and querying engineering drawings using natural language and image uploads.

## Features

- **Text Query → Text Response**: Ask questions and get natural language answers with citations
- **Text Query → Images**: Search for relevant drawings by description
- **Image Upload → Similar Images**: Upload a drawing and find visually similar ones
- **Image Upload → Text Response**: Upload a drawing and get a detailed explanation

## Architecture

- **LangGraph Orchestrator**: Routes queries to appropriate handlers based on input type
- **CLIP Embeddings**: Visual similarity search using OpenCLIP ViT-H-14 (1024-dim)
- **Text Embeddings**: Semantic search using OpenAI text-embedding-3-small (1536-dim)
- **GPT-4o**: Natural language generation and vision descriptions
- **Supabase**: Stores image descriptions and embeddings

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Ensure your `.env` file in the parent directory contains:

```env
OPENAI_API_KEY=your_openai_key
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_service_role_key
```

### 3. Run the App

```bash
cd imageCHAT
streamlit run streamlit_chat_app.py
```

Or from the parent directory:

```bash
streamlit run imageCHAT/streamlit_chat_app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage

### Text Queries

Simply type your question in the chat input:
- "What are the foundation details for project 25-01-006?"
- "Show me images of retaining walls"
- "What rebar is used in the slab?"

### Image Uploads

1. Click "Upload an image" in the sidebar
2. Select a PNG/JPG image
3. The system will:
   - Generate CLIP embedding for visual similarity
   - Use GPT-4o Vision to describe the image
   - Search both image and text embeddings
   - Return similar images or text response

### Conversation History

The app maintains conversation context, so you can ask follow-up questions like:
- "Tell me more about that"
- "What about the rebar spacing?"

## File Structure

```
florence_embedding/
└── imageCHAT/
    ├── streamlit_chat_app.py      # Main Streamlit UI
    ├── langgraph_orchestrator.py  # Query routing logic
    ├── supabase_utils.py          # Supabase database operations
    ├── embedding_utils.py         # CLIP and text embedding functions
    ├── gpt4o_utils.py             # GPT-4o vision and text generation
    ├── requirements.txt           # Python dependencies
    └── STREAMLIT_APP_README.md    # This file
```

## Performance Notes

- **Vector Search**: The current implementation uses Python-side similarity computation. For production with large datasets, create database RPC functions for efficient vector search.
- **CLIP Model**: First load may take time. Model is cached globally.
- **GPT-4o**: API calls may take a few seconds. Responses are streamed when possible.

## Troubleshooting

### "SUPABASE_KEY not set"
- Check your `.env` file exists and contains `SUPABASE_KEY`
- Ensure you're using the service role key (not anon key)

### "OPENAI_API_KEY not set"
- Add your OpenAI API key to `.env`

### "No results found"
- Try rephrasing your query
- Check that your Supabase tables have data
- Verify embeddings exist in both `image_descriptions` and `image_embeddings` tables

### CLIP model loading errors
- Ensure `open-clip-torch` is installed: `pip install open-clip-torch`
- Check you have sufficient memory/GPU

## Advanced: Database RPC Functions (Optional)

For better performance with large datasets, create these functions in Supabase SQL Editor:

```sql
-- Search text embeddings
CREATE OR REPLACE FUNCTION search_text_embeddings(
  query_embedding vector(1536),
  embedding_col text,
  top_k int
) RETURNS TABLE(...) AS $$
  SELECT * FROM image_descriptions
  WHERE embedding_col IS NOT NULL
  ORDER BY embedding_col <-> query_embedding
  LIMIT top_k;
$$ LANGUAGE plpgsql;

-- Search image embeddings
CREATE OR REPLACE FUNCTION search_image_embeddings(
  query_embedding vector(1024),
  top_k int
) RETURNS TABLE(...) AS $$
  SELECT * FROM image_embeddings
  WHERE embedding IS NOT NULL
  ORDER BY embedding <-> query_embedding
  LIMIT top_k;
$$ LANGUAGE plpgsql;
```

Then update `supabase_utils.py` to use these functions (already implemented as fallback).
