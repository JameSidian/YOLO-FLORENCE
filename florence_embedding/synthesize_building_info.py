#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Building-Level Information Synthesis

This script reads the structured JSON output from extract_structured_info.py and
synthesizes high-level building information from all image extractions.

It extracts:
- Lateral resisting system
- Gravity system
- Materials used
- Dimensions (overall building dimensions)
- Members used (W8X24, composite beams, trusses, etc.)
- Client
- Location
- Key elements
- Levels
- Overall design philosophy and decisions
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from openai import OpenAI
from PIL import Image
import base64
from io import BytesIO

# ================= CONFIGURATION =================
API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY_HERE")  # Set OPENAI_API_KEY environment variable or replace with your key
BASE_DIR = r"C:\Users\brian\OneDrive\Desktop\dataprocessing"
STRUCTURED_JSON_DIR = os.path.join(BASE_DIR, "florence_embedding", "structured_json")
OUTPUT_DIR = os.path.join(BASE_DIR, "florence_embedding", "building_synthesis")
TEST_EMBEDDINGS_DIR = os.path.join(BASE_DIR, "test_embeddings")
# =================================================

SYNTHESIS_PROMPT = r"""
You are a senior structural engineer analyzing a complete set of engineering drawing extractions for a building project.

Your task is to synthesize high-level building information from all the individual image extractions provided.

**STRUCTURAL DRAWING CONTEXT (HOW THE SET IS ORGANIZED):**
- Treat the provided `images` array as a complete structural drawing set that includes plans, sections, elevations, details, and schedules
- **Key / site / general arrangement plans** define the building footprint and its relationship to the site; use these for global geometry and orientation
- **Plans** (foundation, floor, roof/framing) are horizontal cuts showing X/Y geometry: grids, column/footing locations, wall lines, slab edges, openings, stairs, etc.
- **Sections and elevations** are vertical cuts/views (X/Z or Y/Z) showing levels, story heights, wall/footing depths, roof slopes, and how floor/roof systems bear on supports
- **Details** are enlarged views of typical conditions (e.g., typical column C1, beam B1, slab edge detail) that apply wherever their tags appear in plans/sections
- **Schedules / tables** (beam schedules, rebar schedules, lintel schedules, notes) decode the symbols and labels on the plans by listing member sizes, bar marks, spacings, and material specs
- Section/elevation callouts (e.g., bubbles like 3/S2, A-A) indicate explicit connectivity between views; use these to stitch plans, sections, and details into continuous load paths
- Always reason like a structural engineer: identify gravity and lateral load paths from roof → floors → beams/joists → columns/walls → foundations → soil, and infer system types (steel frame, concrete, wood, hybrid) and design intent
- Where drawings omit explicit data, infer reasonable engineering conclusions ONLY when strongly supported by multiple views / notes; otherwise leave fields null or note the uncertainty

**CRITICAL: USING LOCATION AND SECTION REFERENCES TO TIE INFORMATION TOGETHER:**
- **Location Fields**: Each image extraction includes `location`, `level`, `orientation`, and `grid_references` fields. USE THESE to understand spatial relationships:
  - `level` tells you which vertical level (Foundation, Ground Floor, Roof, etc.) - use this to group related elements
  - `orientation` (for sections/elevations) tells you the viewing direction (e.g., "North elevation", "Transverse section looking East") - use this to understand how sections relate to plans
  - `grid_references` (e.g., "Grid 1-5", "Grids A-C") tell you which part of the building - use these to locate elements spatially
  - `location` combines level + orientation + zone - use this to tie elements together (e.g., "Foundation – F1 footing along North wall")
- **Section Callouts**: The `section_callouts` field contains references like "1/S2.1", "A-A", "3/S3" that link plans to their corresponding sections/details:
  - When a plan shows "SEE SECTION 1/S2.1", find the image with that section callout to understand the vertical cut
  - Use section callouts to understand how horizontal plans connect to vertical sections
  - Cross-reference section callouts to build a complete 3D understanding of the building
- **Compass Directions**: Plans typically include a north arrow/compass indicator:
  - Identify which direction is North from the plan views
  - Use this to understand how elevations (North elevation, South elevation) and sections (looking North, looking East) relate to the plan geometry
  - When an elevation says "North elevation", it shows the North-facing exterior wall as seen from outside looking South
  - When a section says "Section A-A looking North", it's a cut through the building with the view looking North
- **Tying It Together**: Use location + section references to build a complete picture:
  - If a foundation plan shows "F1" footings at "Grid 3-7" on the "North wall", and a section "1/S2.1" shows those footings, tie them together
  - If a floor plan shows a moment connection detail callout "SEE DETAIL 3/D5" at "Grid 5", find that detail and understand it applies at that location
  - Use level information to understand vertical relationships: foundation elements support ground floor elements, which support upper floors, etc.

**CRITICAL REQUIREMENTS:**
1. Analyze ALL provided image extractions to understand the overall building design
2. Extract building-level information that spans across multiple drawings
3. Provide engineering-level understanding of design decisions and systems
4. Be comprehensive but precise

**OUTPUT FORMAT:**
Output a JSON object optimized for database storage with comma-separated values for filtering. Structure:

{
  "project_id": "<project identifier, e.g., '25-01-006'>",
  "project_name": "<Full project name if found in title blocks or drawing headers, e.g., 'Ray Wilts 75' x 256' Poultry Barn', or null>",
  "client": "<Client name if found in drawings/title blocks, or null>",
  "location": "<Building location/address if found in drawings/title blocks, or null>",
  "building_type": "<MUST be one of: 'Agricultural Building', 'Residential Building', 'Commercial Building', 'Industrial Building', 'Mixed Use', or 'Unknown'>",
  "number_of_levels": <integer count of distinct vertical levels>,
  "levels": "<Comma-separated list of all levels, e.g., 'Foundation, Ground Floor, Roof' or 'Basement, Ground Floor, Second Floor, Roof'>",
  "dimensions_length": "<OVERALL building length from outermost dimension lines, format 'XXX'-XX\"' or null>",
  "dimensions_width": "<OVERALL building width from outermost dimension lines, format 'XXX'-XX\"' or null>",
  "dimensions_height": "<Overall building height if determinable, or null>",
  "dimensions_area": "<Total building area if determinable (sq ft), or null>",
  "gravity_system": "<Comma-separated description of gravity load path: 'Roof Trusses, Loadbearing Walls, Foundation' or similar. Describe the complete path from roof to foundation. Examples: 'Steel Trusses, Wood Stud Walls, Continuous Footings', 'Steel Beams, Steel Columns, Spread Footings', 'Concrete Slab, Concrete Walls, Strip Footings'>",
  "lateral_system": "<Comma-separated list of lateral resisting elements, e.g., 'Shearwalls, Wind Columns' or 'Moment Frames, Braced Frames' or 'Shearwalls' if only one type>",
  "concrete_strengths": "<Comma-separated list of concrete strengths found, e.g., '25 MPa, 32 MPa' or '3000 PSI, 4000 PSI'>",
  "steel_shapes": "<Comma-separated list of steel shapes found, e.g., 'W8X24, W12X26, C6x8.2, HSS' or null if none>",
  "rebar_sizes": "<Comma-separated list of rebar sizes found, e.g., '15M, 20M, 25M' or null if none>",
  "other_materials": "<Comma-separated list of other key materials, e.g., 'Plywood, Prefinished Metal Cladding, Rigid Insulation' or null>",
  "structural_beams": "<Comma-separated list of beam designations, e.g., 'W8X24, W12X26, Composite Beam' or null if none>",
  "structural_columns": "<Comma-separated list of column designations, e.g., 'W6X25, HSS6x6' or null if none>",
  "structural_trusses": "<Comma-separated list of truss types, e.g., 'Steel Truss, Pre-engineered Truss' or null if none>",
  "key_elements": "<Comma-separated list of key structural/architectural elements, e.g., 'Wind Columns, Shearwalls, Retaining Walls, Moment Connections, Atrium, Mezzanine' or null if none. Include any notable features like retaining walls, atriums, special connections, etc.>",
  "overall_building_description": "<CRITICAL: This field contains ONLY NEW INFORMATION not already captured in other JSON fields above. DO NOT repeat: project_id, project_name, client, location, building_type, number_of_levels, levels, dimensions, gravity_system, lateral_system, concrete_strengths, steel_shapes, rebar_sizes, other_materials, structural_beams, structural_columns, structural_trusses, or key_elements. This description should be 500-800 words of DENSE TECHNICAL CONTENT focusing on: (1) DETAILED TECHNICAL SPECIFICATIONS - exact material grades, connection details, fabrication requirements, installation procedures, (2) LOAD PATH ANALYSIS - specific load transfer mechanisms, connection types (welded, bolted, moment, pinned), bearing details, (3) CONSTRUCTION DETAILS - joint configurations, reinforcement layouts, embedment details, anchor bolt patterns, (4) SPATIAL RELATIONSHIPS - grid line connections, section cut relationships, elevation transitions, (5) CODE COMPLIANCE NOTES - specific code references, design criteria, performance requirements, (6) FABRICATION NOTES - welding symbols, bolt specifications, surface preparation requirements, (7) ASSEMBLY SEQUENCES - erection procedures, connection order, temporary bracing requirements, (8) SPECIAL CONDITIONS - site-specific requirements, environmental considerations, access constraints, (9) MAXIMUM KEYWORD DENSITY - pack as many technical terms, material designations, connection types, and engineering jargon as possible into every sentence, (10) NO FLUFF - every sentence must contain technical information, use engineering terminology, avoid descriptive language, be direct and precise, (11) VERBATIM TERMINOLOGY - use exact technical terms, abbreviations, and specifications from drawings (e.g., 'W8X24', '25 MPa', '15M rebar', 'A325 bolts', 'E70XX weld', 'SEE DETAIL 3/S4.2'), (12) ENGINEERING JARGON - use terms like 'load path', 'moment connection', 'shear transfer', 'bearing capacity', 'deflection criteria', 'lateral stability', 'diaphragm action', 'composite action', etc. throughout.>"
}

**ANALYSIS GUIDELINES:**
- **USE LOCATION TO TIE INFORMATION TOGETHER**: Group elements by `level` (Foundation, Ground Floor, Roof, etc.) and use `orientation` and `grid_references` to understand spatial relationships. When multiple images reference the same location (e.g., "Foundation – Grid 3-7"), they describe the same physical element from different views.
- **USE SECTION REFERENCES TO UNDERSTAND CONNECTIVITY**: When a plan shows a section callout (e.g., "SEE SECTION 1/S2.1"), find the corresponding section image to understand the vertical cut. Use section callouts to trace load paths and understand how plans, sections, and details connect.
- **UNDERSTAND COMPASS ORIENTATION**: Identify North from plan views (look for north arrow/compass indicator). Use this to understand how elevations and sections relate to the plan geometry. A "North elevation" shows the North-facing wall; a "Section A-A looking North" is a cut viewed from the South looking North.
- **GRAVITY SYSTEM**: Describe the complete load path from roof to foundation as comma-separated components. Example: "Steel Trusses, Loadbearing Walls, Continuous Footings" means trusses bear on walls, walls bear on footings. Identify the actual components: roof system (trusses/beams), vertical support (walls/columns), foundation (footings/slab).
- **LATERAL SYSTEM**: List the lateral resisting elements as comma-separated values. Common elements: "Shearwalls", "Wind Columns", "Moment Frames", "Braced Frames", "Diaphragm Action". If multiple types exist, list all (e.g., "Shearwalls, Wind Columns").
- **MATERIALS**: Extract concrete strengths, steel shapes, and rebar sizes as comma-separated lists. Do NOT include "applications" - just the values themselves.
- **STRUCTURAL MEMBERS**: Extract beam, column, and truss designations/types as comma-separated lists. Do NOT include location or notes - just the designations/types.
- **KEY ELEMENTS**: Identify notable structural/architectural features like moment connections, retaining walls, atriums, mezzanines, wind columns, shearwalls, special openings, etc. List as comma-separated values.
- **LEVELS**: Count distinct levels and list them as comma-separated string (e.g., "Foundation, Ground Floor, Roof").
- Extract project name from title blocks or drawing headers
- Determine building type from context clues (barn/agricultural = Agricultural, house/apartment = Residential, office/retail = Commercial)
- If information is not available, use null (not empty strings or arrays)

**CRITICAL: OVERALL BUILDING DESCRIPTION REQUIREMENTS:**
- **NO REPETITION**: The `overall_building_description` field MUST NOT repeat any information already captured in other JSON fields (project_id, project_name, client, location, building_type, number_of_levels, levels, dimensions, gravity_system, lateral_system, concrete_strengths, steel_shapes, rebar_sizes, other_materials, structural_beams, structural_columns, structural_trusses, key_elements). This field is for NEW technical information only.
- **WORD COUNT**: MUST be 500-800 words of DENSE technical content. Every sentence must contain technical information - no filler, no fluff, no descriptive language.
- **MAXIMUM KEYWORD DENSITY**: Pack as many technical terms, material designations, connection types, code references, and engineering jargon as possible into every sentence. Include ALL extracted keywords from verbatim text.
- **VERBATIM TEXT USAGE**: Extensively incorporate exact phrases, technical terms, specifications, and terminology directly from the `text_verbatim` fields. Use exact material designations (e.g., 'W8X24', '25 MPa', '15M rebar'), connection details (e.g., 'A325 bolts', 'E70XX weld'), and drawing references (e.g., 'SEE DETAIL 3/S4.2').
- **ENGINEERING JARGON**: Use technical terminology throughout: load path, moment connection, shear transfer, bearing capacity, deflection criteria, lateral stability, diaphragm action, composite action, connection details, fabrication requirements, erection procedures, etc.
- **TECHNICAL FOCUS**: Focus on: detailed specifications (material grades, bolt sizes, weld types), load path analysis (connection types, bearing details), construction details (joint configurations, reinforcement layouts, anchor patterns), spatial relationships (grid connections, section relationships), code compliance (specific references, design criteria), fabrication notes (welding symbols, surface prep), assembly sequences (erection order, bracing), special conditions (site requirements, environmental factors).
- **NO FLUFF**: Every sentence must be technical and precise. Avoid descriptive language, general statements, or explanations of obvious concepts. Be direct and engineering-focused.

**CRITICAL: OVERALL DIMENSIONS EXTRACTION:**
- **MANDATORY**: Dimensions MUST be extracted. If not found in image extractions, they will be extracted from overall building images at the project root.
- **PRIMARY SOURCE**: Floor plans and foundation plans are the PRIMARY sources for overall building dimensions
- For "length" and "width": You MUST identify dimension lines that span the ENTIRE building from one exterior wall edge to the opposite exterior wall edge
- Look for the OUTERMOST dimension lines on plan views - these typically show the overall building dimensions
- Do NOT confuse partial dimensions (e.g., "137'-0\"" which might be an internal span) with overall dimensions
- Cross-reference multiple plan views (Ground Floor Plan, Foundation Plan, Roof Plan) to verify overall dimensions
- If multiple dimension lines exist, the OVERALL dimension should be the longest one that clearly spans the full building extent
- Look for dimension lines at the very edges of the building outline - these are typically the overall dimensions
- If you see dimensions like "256'-0\"", "295'-0\"", "295'-4\"" on the outermost edges, these are likely the overall dimensions
- Internal dimensions like "137'-0\"", "126'-0\"", "130'-0\"" are partial dimensions and should NOT be used for overall building dimensions
- When in doubt, look for the dimension line that encompasses the entire building footprint visible in the plan view
- **FLOOR PLANS AND FOUNDATION PLANS**: These are the most reliable sources - prioritize extracting dimensions from images classified as "Plan" with location indicating "Foundation" or "Ground Floor" or similar
"""

# Initialize client (will be validated in main)
client = None


def load_structured_json(project_id: str) -> Dict[str, Any]:
    """Load structured JSON for a project"""
    json_file = Path(STRUCTURED_JSON_DIR) / project_id / f"structured_{project_id}.json"
    
    if not json_file.exists():
        raise FileNotFoundError(f"Structured JSON not found: {json_file}")
    
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_page_metadata(project_id: str) -> Dict[str, Any] | None:
    """Load page-level metadata if available (contains title block info)"""
    page_meta_file = Path(STRUCTURED_JSON_DIR) / project_id / f"page_metadata_{project_id}.json"
    
    if not page_meta_file.exists():
        return None
    
    try:
        with open(page_meta_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def find_overall_building_images(project_id: str) -> List[Path]:
    """Find overall building images at the root of the test_embeddings folder for a project"""
    project_dir = Path(TEST_EMBEDDINGS_DIR) / project_id
    
    if not project_dir.exists():
        return []
    
    # Look for images at the root level (not in subdirectories like page_001/)
    # These are typically the full-page building images
    overall_images = []
    
    for img_file in project_dir.glob("*.png"):
        # Check if it's at root level (not in a subdirectory)
        if img_file.parent == project_dir:
            overall_images.append(img_file)
    
    # Also check for .jpg files
    for img_file in project_dir.glob("*.jpg"):
        if img_file.parent == project_dir:
            overall_images.append(img_file)
    
    return sorted(overall_images)


def extract_dimensions_from_image(image_path: Path, client: OpenAI) -> Optional[Dict[str, str]]:
    """Extract building dimensions from an overall building image using GPT-4o Vision"""
    try:
        # Load and encode image
        img = Image.open(image_path)
        
        # Convert to base64
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        # Create GPT-4o Vision prompt for dimension extraction
        prompt = """Analyze this engineering drawing image and extract the OVERALL building dimensions.

CRITICAL INSTRUCTIONS:
1. Look for dimension lines that span the ENTIRE building from one exterior wall edge to the opposite exterior wall edge
2. Focus on FLOOR PLANS and FOUNDATION PLANS - these typically show overall dimensions
3. Look for the OUTERMOST dimension lines at the edges of the building footprint
4. Extract length and width (or longest and shortest dimensions)
5. Look for overall building height if shown in elevations or sections
6. Look for total building area if shown

Return ONLY a JSON object with this structure:
{
  "dimensions_length": "<format: 'XXX'-XX\"' or null if not found>",
  "dimensions_width": "<format: 'XXX'-XX\"' or null if not found>",
  "dimensions_height": "<overall height or null if not found>",
  "dimensions_area": "<total area in sq ft or null if not found>"
}

If dimensions are not clearly visible or cannot be determined, use null for those fields.
Be precise - only extract dimensions that clearly span the entire building footprint."""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_base64}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500,
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        result_text = response.choices[0].message.content
        result_json = json.loads(result_text)
        
        # Validate that we got at least some dimension data
        if any(result_json.get(k) for k in ["dimensions_length", "dimensions_width", "dimensions_height", "dimensions_area"]):
            return result_json
        
        return None
        
    except Exception as e:
        print(f"      ⚠️ Error extracting dimensions from {image_path.name}: {e}")
        return None


def extract_and_validate_dimensions(project_id: str, existing_dimensions: Dict[str, Any], client: OpenAI) -> Dict[str, Any]:
    """Extract dimensions from overall building images and validate/update existing dimensions"""
    
    # Find overall building images
    overall_images = find_overall_building_images(project_id)
    
    if not overall_images:
        print(f"      ⚠️ No overall building images found at root level for {project_id}")
        return existing_dimensions
    
    print(f"      🔍 Found {len(overall_images)} overall building image(s), extracting dimensions...")
    
    # Try to extract dimensions from each image
    extracted_dims = {}
    for img_path in overall_images:
        dims = extract_dimensions_from_image(img_path, client)
        if dims:
            # Merge extracted dimensions (prefer non-null values)
            for key, value in dims.items():
                if value and value != "null" and not extracted_dims.get(key):
                    extracted_dims[key] = value
    
    # Merge with existing dimensions (prefer extracted if available, otherwise keep existing)
    final_dims = {}
    for key in ["dimensions_length", "dimensions_width", "dimensions_height", "dimensions_area"]:
        # Prefer extracted dimensions if available, otherwise use existing
        if extracted_dims.get(key) and extracted_dims[key] != "null":
            final_dims[key] = extracted_dims[key]
        elif existing_dimensions.get(key) and existing_dimensions[key] != "null":
            final_dims[key] = existing_dimensions[key]
        else:
            final_dims[key] = None
    
    # Log what we found
    if extracted_dims:
        print(f"      ✅ Extracted dimensions: {extracted_dims}")
    if any(final_dims.values()):
        print(f"      ✅ Final dimensions: {final_dims}")
    else:
        print(f"      ⚠️ No dimensions found in overall images or existing data")
    
    return final_dims


def synthesize_building_info(project_id: str, structured_data: Dict[str, Any], client: OpenAI) -> Dict[str, Any]:
    """Synthesize building-level information from structured image data"""
    
    # Load page metadata for title block information
    page_metadata = load_page_metadata(project_id)
    
    # Prepare context from all images
    images_data = structured_data.get("images", [])
    
    if not images_data:
        raise ValueError(f"No image data found for project {project_id}")
    
    # Create a summary of all image extractions for the LLM
    context_text = f"Project ID: {project_id}\n\n"
    
    # Add title block information from page metadata if available
    if page_metadata:
        pages = page_metadata.get("pages", [])
        if pages:
            context_text += "TITLE BLOCK INFORMATION (from full-page analysis):\n"
            context_text += "=" * 80 + "\n"
            for page in pages:
                title_block = page.get("text_verbatim_title_block", "")
                drawing_title = page.get("drawing_title", "")
                sheet_id = page.get("sheet_id", "")
                if title_block or drawing_title or sheet_id:
                    context_text += f"  Page {page.get('page_number', 'N/A')} (Sheet {sheet_id}): {drawing_title}\n"
                    if title_block:
                        # Truncate very long title blocks
                        tb_preview = title_block[:1000] + "..." if len(title_block) > 1000 else title_block
                        context_text += f"    Title Block: {tb_preview}\n"
            context_text += "\n"
    
    context_text += f"Total Images Analyzed: {len(images_data)}\n\n"
    
    # Build a section reference map to help understand connectivity
    section_ref_map = {}
    for img in images_data:
        section_callouts = img.get('section_callouts', [])
        for callout in section_callouts:
            if callout not in section_ref_map:
                section_ref_map[callout] = []
            section_ref_map[callout].append({
                'image_id': img.get('image_id'),
                'classification': img.get('classification'),
                'location': img.get('location'),
                'level': img.get('level'),
                'page': img.get('page_number')
            })
    
    # Add section reference connectivity information
    if section_ref_map:
        context_text += "SECTION REFERENCE CONNECTIVITY:\n"
        context_text += "=" * 80 + "\n"
        context_text += "The following section callouts link related views together. Use these to understand how plans, sections, and details connect:\n\n"
        for callout, refs in sorted(section_ref_map.items()):
            context_text += f"  Section Callout '{callout}' appears in:\n"
            for ref in refs:
                context_text += f"    - {ref['image_id']} ({ref['classification']}, {ref.get('location', 'N/A')}, Level: {ref.get('level', 'N/A')}, Page: {ref.get('page', 'N/A')})\n"
        context_text += "\n"
    
    # Add location-based grouping summary
    level_groups = {}
    for img in images_data:
        level = img.get('level')
        if level:
            if level not in level_groups:
                level_groups[level] = []
            level_groups[level].append({
                'image_id': img.get('image_id'),
                'classification': img.get('classification'),
                'location': img.get('location'),
                'element_type': img.get('element_type')
            })
    
    if level_groups:
        context_text += "LOCATION-BASED GROUPING (by Level):\n"
        context_text += "=" * 80 + "\n"
        context_text += "Elements grouped by vertical level. Use this to understand vertical relationships and load paths:\n\n"
        for level in sorted(level_groups.keys()):
            context_text += f"  Level: {level} ({len(level_groups[level])} images):\n"
            for item in level_groups[level][:10]:  # Limit to first 10 per level to avoid bloat
                context_text += f"    - {item['image_id']}: {item['classification']} - {item.get('element_type', 'N/A')} ({item.get('location', 'N/A')})\n"
            if len(level_groups[level]) > 10:
                context_text += f"    ... and {len(level_groups[level]) - 10} more\n"
        context_text += "\n"
    
    context_text += "Image Extractions:\n"
    context_text += "=" * 80 + "\n\n"
    
    for i, img in enumerate(images_data, 1):
        context_text += f"Image {i}: {img.get('image_id', 'unknown')}\n"
        context_text += f"  Classification: {img.get('classification', 'N/A')}\n"
        context_text += f"  Location: {img.get('location', 'N/A')}\n"
        if img.get('level') is not None:
            context_text += f"  Level: {img.get('level', 'N/A')}\n"
        if img.get('orientation') is not None:
            context_text += f"  Orientation: {img.get('orientation', 'N/A')}\n"
        grid_refs = img.get('grid_references', [])
        if grid_refs:
            context_text += f"  Grid References: {', '.join(grid_refs)}\n"
        context_text += f"  Page: {img.get('page_number', 'N/A')}\n"
        context_text += f"  Element Type: {img.get('element_type', 'N/A')}\n"
        
        key_components = img.get('key_components', [])
        if key_components:
            context_text += f"  Key Components: {', '.join(key_components)}\n"
        
        section_callouts = img.get('section_callouts', [])
        if section_callouts:
            context_text += f"  Section Callouts: {', '.join(section_callouts)}\n"

        element_callouts = img.get('element_callouts', [])
        if element_callouts:
            context_text += f"  Element Callouts: {', '.join(element_callouts)}\n"
        
        text_verbatim = img.get('text_verbatim', '')
        if text_verbatim:
            # Include more verbatim text to ensure all keywords and technical details are available for synthesis
            # For Plan views, include even more as they contain critical dimension and layout information
            if img.get('classification', '').lower() == 'plan':
                verbatim_preview = text_verbatim[:3000] + "..." if len(text_verbatim) > 3000 else text_verbatim
            elif img.get('classification', '').lower() in ['section', 'detail']:
                # Sections and details contain critical structural information
                verbatim_preview = text_verbatim[:2000] + "..." if len(text_verbatim) > 2000 else text_verbatim
            else:
                verbatim_preview = text_verbatim[:1000] + "..." if len(text_verbatim) > 1000 else text_verbatim
            context_text += f"  Text (verbatim - use extensively in description): {verbatim_preview}\n"
        
        summary = img.get('summary', '')
        if summary:
            context_text += f"  Summary: {summary}\n"
        
        context_text += "\n" + "-" * 80 + "\n\n"
    
    try:
        user_prompt = f"""Analyze the following engineering drawing extractions and synthesize building-level information.

CRITICAL INSTRUCTIONS FOR overall_building_description:
1. DO NOT REPEAT information already in other JSON fields (project_id, project_name, client, location, building_type, number_of_levels, levels, dimensions, gravity_system, lateral_system, concrete_strengths, steel_shapes, rebar_sizes, other_materials, structural_beams, structural_columns, structural_trusses, key_elements). This field is for NEW technical information only.
2. MUST be 500-800 words of DENSE technical content - every sentence must contain technical information, no fluff
3. MAXIMUM KEYWORD DENSITY - pack as many technical terms, material designations, connection types, and engineering jargon as possible into every sentence
4. USE VERBATIM TEXT extensively - incorporate exact phrases, specifications, and terminology from "Text (verbatim)" sections (e.g., 'W8X24', '25 MPa', 'A325 bolts', 'SEE DETAIL 3/S4.2')
5. ENGINEERING JARGON throughout - use terms like load path, moment connection, shear transfer, bearing capacity, deflection criteria, lateral stability, diaphragm action, composite action
6. FOCUS ON: detailed specifications (material grades, bolt sizes, weld types), load path analysis (connection types, bearing details), construction details (joint configurations, reinforcement layouts), spatial relationships (grid connections, section relationships), code compliance notes, fabrication requirements, assembly sequences, special conditions
7. NO FLUFF - be direct, precise, technical. Avoid descriptive language or general statements.

Engineering Drawing Extractions:
{context_text}"""
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYNTHESIS_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=16384,
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        
        result_text = response.choices[0].message.content
        result_json = json.loads(result_text)
        
        # Ensure project_id is set
        result_json["project_id"] = project_id
        
        # Extract and validate dimensions from overall building images
        existing_dims = {
            "dimensions_length": result_json.get("dimensions_length"),
            "dimensions_width": result_json.get("dimensions_width"),
            "dimensions_height": result_json.get("dimensions_height"),
            "dimensions_area": result_json.get("dimensions_area")
        }
        
        print(f"      🔍 Validating dimensions from overall building images...")
        validated_dims = extract_and_validate_dimensions(project_id, existing_dims, client)
        
        # Update result with validated dimensions (prefer extracted if available)
        for dim_key in ["dimensions_length", "dimensions_width", "dimensions_height", "dimensions_area"]:
            if validated_dims.get(dim_key):
                result_json[dim_key] = validated_dims[dim_key]
            elif not result_json.get(dim_key):
                result_json[dim_key] = None
        
        return result_json
    
    except Exception as e:
        print(f"      ❌ Synthesis failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_project(project_id: str, client: OpenAI):
    """Process a single project to synthesize building information"""
    
    print(f"\n--- Synthesizing Building Info for Project: {project_id} ---")
    
    # Load structured JSON
    try:
        structured_data = load_structured_json(project_id)
        print(f"   📂 Loaded {len(structured_data.get('images', []))} image extractions")
    except Exception as e:
        print(f"   ❌ Error loading structured JSON: {e}")
        return
    
    # Synthesize building information
    print(f"   ⚙️ Synthesizing building-level information...")
    building_info = synthesize_building_info(project_id, structured_data, client)
    
    if not building_info:
        print(f"   ❌ Synthesis failed")
        return
    
    # Save output
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"building_info_{project_id}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(building_info, f, indent=2, ensure_ascii=False)
    
    print(f"   ✅ Complete!")
    print(f"   💾 Saved to: {output_file}")
    
    # Print summary
    print(f"\n   📊 Building Summary:")
    print(f"      Building Type: {building_info.get('building_type', 'N/A')}")
    
    # Handle levels - could be a string or list
    levels = building_info.get('levels', [])
    if isinstance(levels, str):
        # If it's a comma-separated string, count the items
        level_count = len([l.strip() for l in levels.split(',') if l.strip()])
    elif isinstance(levels, list):
        level_count = len(levels)
    else:
        level_count = 0
    print(f"      Levels: {level_count}")
    
    # gravity_system and lateral_system are strings, not dictionaries
    gravity_system = building_info.get('gravity_system', 'N/A')
    if gravity_system and gravity_system != 'null':
        print(f"      Gravity System: {gravity_system}")
    else:
        print(f"      Gravity System: N/A")
    
    lateral_system = building_info.get('lateral_system', 'N/A')  # Note: key is 'lateral_system', not 'lateral_resisting_system'
    if lateral_system and lateral_system != 'null':
        print(f"      Lateral System: {lateral_system}")
    else:
        print(f"      Lateral System: N/A")


def main():
    """Main entry point"""
    global client
    
    print("="*60)
    print("Building-Level Information Synthesis")
    print("="*60)
    
    # Validate API key
    if not API_KEY or len(API_KEY.strip()) < 20:
        print("❌ ERROR: OpenAI API key not set or invalid!")
        print("   Set OPENAI_API_KEY environment variable or update API_KEY in the script")
        return
    
    # Initialize client
    client = OpenAI(api_key=API_KEY)
    
    # Check if project ID provided as argument
    import sys
    if len(sys.argv) > 1:
        project_id = sys.argv[1]
        process_project(project_id, client)
    else:
        # Process all projects in structured_json directory
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
        
        for project_id in projects:
            try:
                process_project(project_id, client)
            except Exception as e:
                print(f"❌ Error processing {project_id}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print("\n" + "="*60)
    print("✨ Synthesis Complete!")
    print("="*60)


if __name__ == "__main__":
    main()

