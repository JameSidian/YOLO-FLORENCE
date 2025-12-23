#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Structured Information Extraction for Engineering Drawings

This script processes region images from engineering drawings and extracts structured
information optimized for Supabase storage and semantic search.

For each image, it extracts:
- Classification: Plan, elevation, section, detail, notes, schedule
- Location: North, east, level 1, level 2, etc.
- Section callouts: Section cuts like "A/S01" that link to details
- Element type: Retaining wall, floating slab, main floor plan, etc.
- Text verbatim: Exact text word-for-word from the image
- Summary: Rich summary explaining how text relates to linework (for embeddings)

The summary incorporates verbatim text but explains relationships (e.g., "retaining wall 
height is 5' and rebar is 15M @10" o/c").
"""

import os
import json
import base64
import re
from pathlib import Path
from typing import Dict, Any, List
from openai import OpenAI
from tqdm import tqdm

# ================= CONFIGURATION =================
# API Key: Set OPENAI_API_KEY environment variable, or update the line below
API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY_HERE")  # Set OPENAI_API_KEY environment variable or replace with your key
BASE_DIR = r"C:\Users\brian\OneDrive\Desktop\dataprocessing"
TEST_EMBEDDINGS_DIR = os.path.join(BASE_DIR, "test_embeddings")
OUTPUT_DIR = os.path.join(BASE_DIR, "florence_embedding", "structured_json")
BATCH_SIZE = 5  # Process multiple images per API call
# =================================================

# System prompt for GPT-4o vision
EXTRACTION_PROMPT = r"""
You are a senior structural engineer analyzing technical drawing images.

Your task is to extract structured information from each image for database storage and semantic search.

**STRUCTURAL DRAWING CONTEXT (HOW TO READ THE DRAWINGS):**
- Treat every image as part of a coordinated structural drawing set (plans, sections, elevations, details, schedules)
- **Key / Site / General Arrangement plans**: show the building in context (site, roads, existing structures) and overall footprint
- **Plans**: top‑down cuts at a level (foundation plan, ground floor plan, roof/framing plan). X/Y axes show length and width; use these views to understand overall geometry, grids, and layout of foundations, columns, beams, slabs, doors, and openings
- **Sections**: vertical cuts (X/Z or Y/Z). Use these to understand heights, levels, vertical relationships, wall/footing depths, and how roof/floor systems bear on supports
- **Details**: enlarged portions of plans or sections that clarify reinforcement, connections, member build‑ups, etc. A single typical detail (e.g., column C1, beam B1) often applies wherever that tag appears on the plans
- **Schedules / tables**: tabular data listing members, reinforcement, spacings, bar sizes, etc. Use these to decode tags and symbols in the plans/sections (e.g., beam schedule, rebar schedule)
- Section/elevation callout symbols (e.g., bubbles with numbers and sheet references) define how drawings connect; use them to relate plan regions to their corresponding sections and details
- When interpreting ANY image, always reason like a structural engineer: identify load paths (roof → floors → beams/joists → columns/walls → foundations → soil), system types (steel, concrete, wood), and how typical details and schedules control repeated elements

**CRITICAL REQUIREMENTS:**
1. Extract ALL visible text VERBATIM (word-for-word, exactly as shown) in the "text_verbatim" field
2. **MANDATORY: Extract ALL section/elevation callouts** (e.g., "1/S1.0", "2/S2.0", "3/S1.0", "A-A", "B-B"). These are ESSENTIAL for understanding how drawings connect. Look for circular/diamond callout bubbles with numbers/letters above and sheet references below, or text like "SECTION 1 / S1.0". In plan views especially, scan the ENTIRE image for ALL section callout symbols - plans often have multiple callouts that MUST all be captured.
3. The "summary" field MUST incorporate ALL verbatim text values but explain their meaning and relationships
4. Example: If verbatim text contains "5'-0\"" and "15M @10\" o/c", the summary should say:
   "This detail shows a retaining wall with a height of 5'-0\" (five feet zero inches) as indicated by the dimension line. The reinforcement consists of 15M rebar bars spaced at 10 inches on center, as specified in the notation '15M @10\" o/c'. The wall thickness is 8 inches with a 4-inch concrete foundation below."
5. The summary should be rich and detailed for semantic search - explain HOW the text relates to the visual elements
6. Be precise with classifications, locations, and element types
7. **CRITICAL: All JSON strings must be properly escaped. Use \\" for quotes within strings, \\n for newlines, and ensure all strings are properly closed.**

**OUTPUT FORMAT:**
Output a JSON object with a single key "images" containing an array. Each image object must have these exact keys:

{
  "images": [
    {
      "image_id": "<filename>",
      "classification": "<One of: Plan, Elevation, Section, Detail, Notes, Schedule>",
      "location": "<Concise human-readable description of where this view/element sits in the building, combining level + orientation + zone where possible. Examples: 'Foundation – F1 footing along North wall', 'Ground Floor – main barn bay between grid 3-7', 'Roof – typical truss at mid-span', 'East wall elevation', or null if not determinable.>",
      "level": "<Normalized vertical level for this view/element: one of ['Site', 'Foundation', 'Basement', 'Ground Floor', 'Mezzanine', 'Second Floor', 'Third Floor', 'Roof', 'Parapet', 'Unknown']>",
      "orientation": "<For elevations/sections: primary orientation and cut direction, e.g., 'Transverse section looking North at mid-span', 'Longitudinal section looking East along grid 1', 'North elevation', 'South wall interior elevation', or null if not obvious>",
      "grid_references": [
        "<Any grid lines or bay references explicitly visible, e.g., 'Grid 1-5', 'Grids A-C', 'Between Grid 3 and Grid 7'. Return [] if none visible.>"
      ],
      "section_callouts": [
        "<e.g., 'A/S01', '3/D1', '1/S2.1', 'Section A-A', 'A-A'> - only SECTION or ELEVATION callout markers that reference other views/sheets"
      ],
      "element_type": "<e.g., 'Retaining Wall', 'Floating Slab', 'Main Floor Plan', 'Foundation Detail', 'Roof Framing', 'Wall Assembly', etc.>",
      "element_callouts": [
        "<Element / member / type tags such as 'F1', 'F2', 'W1', 'B1', 'T1', 'WE1', 'C1', 'BEAM 1', etc. that identify specific foundations, wall types, beam marks, truss types, slab types or other repeated elements. DO NOT put section or elevation references with a slash (e.g., '1/S2.1', 'A/A2.1') here.>"
      ],
      "key_components": [
        "<List of key structural components, materials, or systems visible in the image. For plans: e.g., 'Slab-on-Grade', 'Foundation Walls', 'Manure Storage Pad', 'Dropped Pen Floor Areas'. For details: e.g., 'Rebar', 'Concrete', 'Welded Wire Fabric', 'Footing'. Each distinct element type should be listed as a separate component.>"
      ],
      "text_verbatim": "<ALL visible text extracted exactly as shown, preserving line breaks and formatting. Include dimensions, notes, labels, schedules, callouts, etc. Use \n for line breaks.>",
      "summary": "<A comprehensive summary that: (1) Describes what the image shows visually (linework, structural elements), (2) Incorporates EVERY verbatim text value found in the image but explains their meaning and relationships, (3) Explains how text annotations relate to the linework and structural elements, (4) Is rich in detail for semantic search and embeddings. The summary must reference specific values from the verbatim text. MAXIMUM 500 WORDS - ensure the summary does not exceed 500 words. Example: 'This section view shows a retaining wall detail with foundation. The wall has a height of 5'-0\" as indicated by the vertical dimension line. Reinforcement consists of 15M rebar bars spaced at 10 inches on center (15M @10\" o/c) as shown in the rebar notation. The wall is 8 inches thick and includes a 4-inch concrete foundation below. The detail shows the connection between the wall and foundation, with angle fasteners specified as 8-5/16\" x 3\" wood lags fastened to studs.'>"
    }
  ]
}

**CLASSIFICATION GUIDELINES:**
- "Plan": Floor plans, site plans, roof plans (top-down views showing layout)
- "Elevation": Building elevations, exterior views (side views of building)
- "Section": Section cuts through the building (cut-away views)
- "Detail": Close-up details of specific connections or assemblies (zoomed-in views)
- "Notes": General notes, specifications, legends (text-heavy, no major linework)
- "Schedule": Tables, schedules (beam schedules, lintel schedules, rebar schedules, etc.)

**LOCATION & LEVEL GUIDELINES:**
- Always try to describe BOTH vertical level and horizontal position/orientation so the element can be tied into a global building model
- Use the "level" field for a **normalized vertical category** (e.g., "Foundation", "Basement", "Ground Floor", "Second Floor", "Roof"). This should come from plan titles, section notes, or callouts like "FOUNDATION PLAN", "GROUND FLOOR PLAN", "ROOF FRAMING PLAN", "SECTION @ GRID 3", etc.
- Use the "location" string to combine level + orientation + any zone information into a concise phrase a structural engineer would say on a coordination call (e.g., "Ground Floor – south bay at grid 7", "Roof – east end over manure storage area", "Foundation – F1 footings under east wall").
- Use the "orientation" field primarily for sections/elevations (e.g., "Transverse section looking North", "Longitudinal section looking East", "North elevation", "West wall elevation"). For plan views, orientation can often be null unless explicitly noted.
- Use "grid_references" to capture any grid lines or bay tags that appear (e.g., ["Grid 1-5", "Grids A-C"]). If only single grids appear (e.g., "3", "A"), include them as simple strings in the array.
- Extract this information from view titles, key notes, grid bubbles, section titles (e.g., "SECTION 1/S2.1 – BUILDING SECTION – BARN"), and nearby annotations.
- Return null for "location", "level", or "orientation" only when you truly cannot infer them from the image. For "grid_references", return [] if no grids are visible.

**SECTION CALLOUTS (CRITICAL - MUST CAPTURE ALL):**
- **MANDATORY**: You MUST extract EVERY section/elevation callout marker visible in the image. These are ESSENTIAL for understanding how drawings connect together.
- Look for section/elevation call markers that reference another view or sheet. Common formats include:
  - **Number/Sheet format**: "1/S1.0", "2/S2.0", "3/S1.0", "1/S2.1", "3/S3", "A/S01", "3/D1" (section number / sheet number)
  - **Letter-letter format**: "A-A", "B-B", "C-C", "Section A-A"
  - **Text references**: "SEE SECTION 1/S2.1", "SEE SECTION A-A", "SECTION 1", "SECTION 3"
  - **With spaces**: "1 / S1.0", "3 / S1.0" (extract as "1/S1.0", "3/S1.0" - normalize by removing spaces)
- **Where to find them:**
  - **In PLAN views**: Section callouts appear as circular or diamond-shaped bubbles with numbers/letters above and sheet references below (e.g., "1" above, "S1.0" below = "1/S1.0"). These indicate where section cuts are made through the plan. **CRITICAL: Plans often have MULTIPLE section callouts - you MUST capture ALL of them.**
  - **In SECTION/ELEVATION views**: The section title itself (e.g., "SECTION 1 / S1.0") is a section callout - extract it!
  - **In DETAIL views**: Section callouts may reference other details or sections
- **Visual patterns to recognize:**
  - Circular bubbles bisected by a cutting plane line, with section number on top and sheet number on bottom
  - Diamond-shaped symbols with section number and sheet reference
  - Triangular arrows pointing along a cutting plane with callout bubbles
  - Text labels like "SECTION 1 / S1.0" or "SECTION 3 WIND COLUMN"
- **Normalization**: If you see "1 / S1.0" or "1/S1.0" or "1-S1.0", normalize to "1/S1.0" format (number/sheet). If you see "SECTION 1 / S1.0", extract as "1/S1.0".
- **CRITICAL DISTINCTION**: 
  - **Section callouts** have a "/" or "-" separating section number from sheet number (e.g., "1/S1.0", "2/S2.0", "3/S1.0") OR are letter-letter format (e.g., "A-A", "B-B")
  - **Element callouts** are single tags without "/" or "-" (e.g., "F1", "W1", "B1", "T1") - these go in "element_callouts"
- **Do NOT include element or type designations such as 'F1', 'F2', 'W1', 'B1', 'T1', 'WE1', etc. in section_callouts** – those belong in "element_callouts"
- **If you see ANY callout symbol with a number/letter above and a sheet reference below, or any text containing "SECTION" followed by a number and sheet reference, it MUST be captured in section_callouts**
- Return empty array [] ONLY if you are absolutely certain no section/elevation callouts are present

**ELEMENT CALLOUTS (FOUNDATION / WALL / BEAM / TRUSS / SLAB TAGS):**
- Collect tags that label specific elements or typical details, e.g., "F1", "F2", "F3" (footing types), "W1", "W2" (wall types), "B1", "B2" (beam marks), "T1", "T2" (truss types), "WE1" (wall elevation), etc.
- These are often shown in diamonds, squares, or other symbols adjacent to foundations, walls, beams, or trusses and are cross‑referenced to schedules or typical details
- Do NOT include view references with a slash "/" (e.g., "1/S2.1", "3/A3.0") – those are section_callouts
- Return empty array [] if none are present

**ELEMENT TYPE:**
- Be specific and descriptive: "Retaining Wall Detail", "Main Floor Framing Plan", "Foundation Wall Section", "Roof Truss Connection", "Beam Schedule", etc.
- Combine the structural element with the view type when appropriate
- Use engineering terminology

**KEY COMPONENTS:**
- List all key structural components, materials, or systems visible in the image
- For floor plans: Each distinct element type should be listed separately (e.g., "Slab-on-Grade", "Foundation Walls", "Manure Storage Pad", "Dropped Pen Floor Areas", "Thickened Slab Edge")
- For details: List specific materials and components (e.g., "Rebar", "Concrete", "Welded Wire Fabric", "Footing", "Curb")
- Be specific and use engineering terminology
- Return empty array [] if no distinct components are identifiable

**TEXT VERBATIM:**
- Extract EVERY piece of text visible in the image
- Preserve formatting where possible (use \n for line breaks between lines)
- Include: dimensions, notes, labels, schedules, callouts, material specifications, etc.
- **CRITICAL FOR PLAN VIEWS: Extract ALL dimension lines and their values, especially overall building dimensions that span from one exterior edge to the opposite exterior edge. Include dimension values like "256'-0\"", "295'-0\"", "295'-4\"", "137'-0\"", "124'-0\"", etc. - even if they appear to be partial dimensions, extract them all as they may be needed for synthesis.**
- **CRITICAL FOR ALL VIEWS: Extract ALL section/elevation callout text exactly as shown. This includes text within callout bubbles (e.g., "1", "S1.0", "3", "S1.0"), section titles (e.g., "SECTION 1 / S1.0", "SECTION 3 WIND COLUMN"), and any text references to sections (e.g., "SEE SECTION 1/S2.1"). These callouts are ESSENTIAL for understanding drawing connectivity.**
- If text is partially obscured, indicate with [unclear] or [partially visible]
- Preserve special characters, units, and formatting exactly as shown

**SUMMARY (MOST IMPORTANT):**
- Must be rich and detailed for semantic search and embeddings
- MUST incorporate verbatim values from the text_verbatim field but explain their meaning
- MUST explain relationships between text and visual elements (e.g., "the dimension '12'-0\"' indicates the width of the room")
- MAXIMUM 500 WORDS - The summary cannot exceed 500 words. It can be less if all information is contained, but it cannot be more than 500 words.
- Should read naturally for semantic search queries
- Example structure: "This [classification] shows [what is visible]. [Specific verbatim values] indicate [what they mean]. [How text relates to linework]. [Additional context and relationships]."
- Before writing the final summary, internally identify all distinct structural systems present in the image (e.g., truss bracing system, slab system, foundation system). The summary MUST describe each system explicitly and explain how the extracted text applies to that system.

STRUCTURAL DRAWING INTERPRETATION RULES:

1. Identify the drawing type first (Plan, Section, Detail, Elevation).
2. Read ALL general notes before interpreting geometry.
3. Notes marked “TYP.” apply unless explicitly overridden.
4. Dimensions govern over scale; drawings may be N.T.S.
5. Structural members are identified by function first, size second.
6. Section and detail callouts reference other drawings and must be captured verbatim.
7. Fastener schedules and spacing notes are critical and must not be summarized away.
8. If multiple systems are present, identify and describe each system separately.
9. Assume industry-standard conventions unless explicitly contradicted by notes.
10. Always explain how text annotations relate to physical members shown in linework.
"""

# ================= PAGE-LEVEL (FULL PAGE) ANALYSIS =================
# Before looking at the individual close‑up region crops, it is very helpful
# to understand what each drawing sheet is showing overall: sheet number,
# drawing title (e.g. "FOUNDATION PLAN"), general notes, and which level /
# systems the sheet applies to. This section adds a first‑pass analysis of the
# full‑page images so that region‑level extraction can benefit from that
# context.

PAGE_EXTRACTION_PROMPT = r"""
You are a senior structural engineer analyzing FULL‑PAGE structural drawing sheets.

Your task in this mode is to understand, for each full sheet:
- What type of drawing it is (foundation plan, floor framing plan, roof framing, building section, details sheet, general notes, schedules, etc.)
- Which level(s) of the building it applies to (foundation, ground floor, roof, etc.)
- The key systems and components shown on the sheet
- The important general notes and title‑block information that apply to all regions on this page

OUTPUT FORMAT (PAGE‑LEVEL):
Return a JSON object with a single key "pages" containing an array. Each page
object must have these keys:

{
  "pages": [
    {
      "page_number": <integer page number if known, otherwise null>,
      "sheet_id": "<Sheet identifier from the title block if present, e.g. 'S1', 'S-2', 'GEN-1', or null>",
      "drawing_title": "<Main drawing title from the title block or view title, e.g. 'FOUNDATION PLAN', 'GROUND FLOOR FRAMING PLAN', 'BUILDING SECTION', 'TYPICAL DETAILS', etc.>",
      "overall_classification": "<One of: Plan, Elevation, Section, Detail, Notes, Schedule, Mixed>",
      "level": "<Normalized vertical level primarily represented on this sheet: one of ['Site', 'Foundation', 'Basement', 'Ground Floor', 'Mezzanine', 'Second Floor', 'Third Floor', 'Roof', 'Parapet', 'Multiple', 'Unknown']>",
      "primary_orientation": "<For elevation/section sheets: e.g., 'North elevation sheets', 'Longitudinal building sections looking East', or null>",
      "key_components": [
        "<High‑level list of key systems/components shown on this sheet, e.g., 'Slab‑on‑Grade Foundations', 'Strip Footings and Foundation Walls', 'Roof Truss Layout', 'Lateral Bracing Frames', 'General Structural Notes', 'Beam and Column Schedules'>"
      ],
      "text_verbatim_title_block": "<All text you can clearly read in the title block area (project name, project number, client, engineer, sheet title, sheet number, revision notes, dates, etc.), verbatim with \\n line breaks. If title block is not visible or very small in this full‑page image, return an empty string.>",
      "general_notes_verbatim": "<Any general notes, legends, or specification paragraphs that obviously apply to the entire sheet (not local callouts), captured verbatim with \\n for line breaks. If none, return an empty string.>",
      "page_overview_summary": "<Short (3‑6 sentence) engineering summary of what this sheet is doing overall, mentioning the main systems, levels, and any particularly important notes that govern the design on this page.>"
    }
  ]
}

Guidelines:
- Treat this as a coarse, sheet‑level understanding pass. Do NOT try to describe every small detail; focus on the overall intent and systems.
- When the sheet contains multiple different drawing types (e.g., plans plus details), use 'Mixed' for overall_classification and describe the mix in page_overview_summary.
- If you can read overall plan extents or major dimensions from the sheet, mention them in page_overview_summary (e.g., "overall barn length 256'-0\" and width 75'-0\" from foundation plan dimensions").
- When in doubt, be conservative and set fields to null or "Unknown" instead of guessing wildly.
"""

# Initialize client (will be validated in main)
client = None


def encode_image(image_path: str) -> str:
    """Encode image to base64 for API"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def truncate_summary(text: str, max_words: int = 500) -> str:
    """Truncate summary to maximum word count"""
    if not text:
        return text
    
    words = text.split()
    if len(words) <= max_words:
        return text
    
    # Truncate to max_words and add ellipsis
    truncated = ' '.join(words[:max_words])
    # Try to end at a sentence boundary
    last_period = truncated.rfind('.')
    if last_period > len(truncated) * 0.8:  # If period is in last 20%, use it
        return truncated[:last_period + 1]
    return truncated + '...'


def repair_json(json_str: str) -> str:
    """Attempt to repair common JSON issues, focusing on text_verbatim fields.

    NOTE: This is a best-effort sanitizer. It is intentionally conservative so
    that well‑formed JSON is left unchanged, but common mistakes inside the
    very long `text_verbatim` strings (unescaped quotes, etc.) are fixed.
    """
    import re

    repaired = json_str

    # 1) Specifically target text_verbatim blocks of the form:
    #    "text_verbatim": " ... ",
    # where the content may contain unescaped double quotes.
    #
    # We find the content lazily up to the following "summary" key so we don't
    # accidentally span across multiple fields.
    pattern = re.compile(
        r'"text_verbatim":\s*"(?P<content>.*?)"\s*,\s*\n\s*"summary"',
        re.DOTALL,
    )

    def _clean_text_verbatim(match: re.Match) -> str:
        content = match.group("content")

        # Protect already‑escaped quotes so we don't double‑escape them.
        placeholder = "\uFFFF"  # unlikely to appear in real text
        protected = content.replace(r"\\\"", placeholder)

        # Escape any remaining raw double quotes.
        protected = protected.replace('"', r"\"")

        # Restore the originally escaped quotes.
        cleaned = protected.replace(placeholder, r"\\\"")

        return '"text_verbatim": "' + cleaned + '",\n      "summary"'

    repaired, _ = pattern.subn(_clean_text_verbatim, repaired)

    # 2) As a very light safety net, ensure there is no stray control character
    #    that might break JSON parsing.
    repaired = repaired.replace("\r\n", "\n")

    return repaired


def parse_json_with_retry(json_str: str, max_retries: int = 2) -> Dict[str, Any]:
    """Parse JSON with retry and repair attempts"""
    global client
    
    for attempt in range(max_retries + 1):
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            if attempt < max_retries:
                # Try to repair common issues
                try:
                    # Try using manual repair
                    repaired = repair_json(json_str)
                    return json.loads(repaired)
                except:
                    # If repair fails, ask GPT to fix it
                    try:
                        fix_response = client.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {"role": "system", "content": "You are a JSON repair assistant. Fix the following malformed JSON and return ONLY valid JSON, no explanations."},
                                {"role": "user", "content": f"Fix this JSON:\n\n{json_str[:8000]}\n\nError: {str(e)}"}
                            ],
                            max_tokens=16384,
                            temperature=0.1,
                            response_format={"type": "json_object"}
                        )
                        fixed_json = fix_response.choices[0].message.content
                        return json.loads(fixed_json)
                    except Exception as fix_error:
                        print(f"      ⚠️ JSON repair attempt {attempt + 1} failed: {fix_error}")
                        pass
            else:
                # Last attempt failed, raise the error
                raise


def find_full_page_images(project_dir: Path) -> List[Dict[str, Any]]:
    """Find all full-page images for a project.

    Prefer using manifest.json (if present) so we stay consistent with how
    pages/regions were generated. Fall back to simple globbing for *page_*.png.
    """
    full_pages: List[Dict[str, Any]] = []
    manifest_path = project_dir / "manifest.json"

    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            for img in manifest.get("images", []):
                if img.get("type") == "full_page":
                    filename = img.get("filename")
                    if not filename:
                        continue
                    page_number = img.get("page_number")
                    img_path = project_dir / filename
                    if img_path.exists():
                        full_pages.append(
                            {
                                "path": img_path,
                                "filename": filename,
                                "page_number": page_number,
                            }
                        )
        except Exception as e:
            print(f"   ⚠️ Could not read manifest.json for full-page images: {e}")

    # Fallback: if manifest missing or empty, try simple glob for *page_*.png
    if not full_pages:
        for img_path in sorted(project_dir.glob("*page_*.png")):
            # Try to infer page number from the filename
            m = re.search(r"page_(\d+)", img_path.name)
            page_number = int(m.group(1)) if m else None
            full_pages.append(
                {
                    "path": img_path,
                    "filename": img_path.name,
                    "page_number": page_number,
                }
            )

    return full_pages


def extract_page_metadata(project_dir: Path, project_id: str) -> Dict[int, Dict[str, Any]]:
    """Run a first-pass analysis over full-page images to get sheet-level context.

    Returns a mapping: page_number -> page_metadata_dict
    """
    global client

    full_pages = find_full_page_images(project_dir)
    if not full_pages:
        print("   ⚠️ No full-page images found; skipping page-level analysis")
        return {}

    print(f"   📄 Found {len(full_pages)} full-page images for page-level analysis")

    # Build a single multi-image request for all full pages
    user_content: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": "Analyze these FULL-PAGE structural drawing sheets and return page-level metadata as described in the system prompt. Output a JSON object with a single key 'pages' containing an array of page metadata objects.",
        }
    ]

    for page in full_pages:
        base64_img = encode_image(str(page["path"]))
        user_content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_img}",
                    "detail": "high",
                },
            }
        )
        context_text = f"Full page image: {page['filename']}"
        if page.get("page_number") is not None:
            context_text += f"\nPage Number: {page['page_number']}"
        user_content.append({"type": "text", "text": context_text})

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": PAGE_EXTRACTION_PROMPT},
                {"role": "user", "content": user_content},
            ],
            max_tokens=8192,
            temperature=0.2,
            response_format={"type": "json_object"},
        )

        result_text = response.choices[0].message.content
        result_json = parse_json_with_retry(result_text)

        pages_array = result_json.get("pages", [])
        page_context: Dict[int, Dict[str, Any]] = {}

        for i, page_meta in enumerate(pages_array):
            # If model didn't set page_number, fall back to our known ordering
            if page_meta.get("page_number") is None and i < len(full_pages):
                page_meta["page_number"] = full_pages[i].get("page_number")

            page_number = page_meta.get("page_number")
            if page_number is None:
                continue

            # Keep compact versions of overview and notes for later context
            overview = page_meta.get("page_overview_summary") or ""
            general_notes = page_meta.get("general_notes_verbatim") or ""

            if len(overview) > 1200:
                overview = overview[:1200] + "..."
            if len(general_notes) > 2000:
                general_notes = general_notes[:2000] + "..."

            page_meta["page_overview_summary"] = overview
            page_meta["general_notes_verbatim"] = general_notes

            page_context[int(page_number)] = page_meta

        # Save page-level metadata alongside structured JSON for reuse/debugging
        output_dir = Path(OUTPUT_DIR) / project_id
        output_dir.mkdir(parents=True, exist_ok=True)
        page_meta_file = output_dir / f"page_metadata_{project_id}.json"
        with open(page_meta_file, "w", encoding="utf-8") as f:
            json.dump({"pages": list(page_context.values())}, f, indent=2, ensure_ascii=False)

        print(f"   💾 Saved page-level metadata to: {page_meta_file}")

        return page_context

    except Exception as e:
        print(f"   ⚠️ Page-level analysis failed: {e}")
        import traceback

        traceback.print_exc()
        return {}


def find_all_region_images(project_dir: Path) -> List[Dict[str, Any]]:
    """Find all region images in a project directory"""
    images = []
    
    # Look for page_XXX subdirectories
    for page_dir in sorted(project_dir.glob("page_*")):
        if not page_dir.is_dir():
            continue
        
        # Extract page number
        page_match = re.search(r'page_(\d+)', page_dir.name)
        page_number = int(page_match.group(1)) if page_match else None
        
        # Find all region images in this page directory
        for img_path in sorted(page_dir.glob("region_*_red_box.png")):
            # Extract region number
            region_match = re.search(r'region_(\d+)_red_box', img_path.name)
            region_number = int(region_match.group(1)) if region_match else None
            
            images.append({
                "path": img_path,
                "filename": img_path.name,
                "page_number": page_number,
                "region_number": region_number,
                "relative_path": f"{page_dir.name}/{img_path.name}"
            })
    
    return images


def process_image_batch(
    image_data_list: List[Dict[str, Any]],
    project_id: str,
    page_context: Dict[int, Dict[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    """Process a batch of region images with GPT-4o vision.

    If page_context is provided, it should be a mapping from page_number to the
    page-level metadata returned by extract_page_metadata, and that context
    will be included as additional text alongside each region image.
    """
    
    user_content = [
        {"type": "text", "text": "Analyze these images and extract structured information. Output a JSON object with a single key 'images' containing an array of image analysis objects."}
    ]
    
    # Add images and context
    for img_data in image_data_list:
        img_path = img_data["path"]
        
        # Encode image
        base64_img = encode_image(str(img_path))
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{base64_img}", "detail": "high"}
        })
        
        # Add context
        context_lines = [f"Image ID: {img_data['filename']}"]
        page_number = img_data.get("page_number")
        if page_number is not None:
            context_lines.append(f"Page Number: {page_number}")

        # If we have page-level metadata for this page, include a compact summary
        if page_context and page_number is not None:
            meta = page_context.get(int(page_number))
            if meta:
                sheet_id = meta.get("sheet_id") or ""
                drawing_title = meta.get("drawing_title") or ""
                overall_class = meta.get("overall_classification") or ""
                level = meta.get("level") or ""
                overview = meta.get("page_overview_summary") or ""

                header_bits = []
                if sheet_id:
                    header_bits.append(f"Sheet ID: {sheet_id}")
                if drawing_title:
                    header_bits.append(f"Drawing Title: {drawing_title}")
                if overall_class:
                    header_bits.append(f"Page Classification: {overall_class}")
                if level:
                    header_bits.append(f"Primary Level: {level}")

                if header_bits:
                    context_lines.append("Page-level context: " + " | ".join(header_bits))
                if overview:
                    context_lines.append(f"Page Overview: {overview}")

        context_text = "\n".join(context_lines)
        user_content.append({"type": "text", "text": context_text})
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": EXTRACTION_PROMPT},
                {"role": "user", "content": user_content}
            ],
            max_tokens=16384,
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        
        result_text = response.choices[0].message.content
        
        # Parse JSON with retry and repair
        try:
            result_json = parse_json_with_retry(result_text)
        except json.JSONDecodeError as e:
            print(f"      ⚠️ JSON parsing failed after retries. Saving raw response for debugging...")
            # Save raw response for debugging
            debug_dir = Path(OUTPUT_DIR) / project_id
            debug_dir.mkdir(parents=True, exist_ok=True)
            debug_file = debug_dir / f"debug_response_{image_data_list[0]['filename']}.txt"
            with open(debug_file, 'w', encoding='utf-8') as f:
                f.write(result_text)
            print(f"      💾 Raw response saved to: {debug_file}")
            print(f"      Error: {str(e)}")
            raise
        
        # Extract images array
        images_array = result_json.get("images", [])
        
        # Ensure each image has the correct image_id
        for i, img_obj in enumerate(images_array):
            if i < len(image_data_list):
                img_obj["image_id"] = image_data_list[i]["filename"]
                # Add metadata
                img_obj["project_id"] = project_id
                img_obj["page_number"] = image_data_list[i].get("page_number")
                img_obj["region_number"] = image_data_list[i].get("region_number")
                img_obj["relative_path"] = image_data_list[i].get("relative_path")
                
                # Truncate summary to 500 words if needed
                if "summary" in img_obj and img_obj["summary"]:
                    img_obj["summary"] = truncate_summary(img_obj["summary"], max_words=500)
        
        return images_array
    
    except Exception as e:
        print(f"      ❌ Batch failed: {e}")
        import traceback
        traceback.print_exc()
        return []


def process_project(project_id: str):
    """Process all region images for a single project"""
    
    project_dir = Path(TEST_EMBEDDINGS_DIR) / project_id
    output_dir = Path(OUTPUT_DIR) / project_id
    
    if not project_dir.exists():
        print(f"❌ Project directory not found: {project_dir}")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"structured_{project_id}.json"
    
    print(f"\n--- Processing Project: {project_id} ---")
    print(f"   Input: {project_dir}")
    print(f"   Output: {output_file}")
    
    # First pass: analyze full-page images to get sheet-level context
    page_context: Dict[int, Dict[str, Any]] = extract_page_metadata(project_dir, project_id)
    
    # Find all region images
    region_images = find_all_region_images(project_dir)
    
    if not region_images:
        print(f"   ⚠️ No region images found in {project_dir}")
        return
    
    print(f"   📸 Found {len(region_images)} region images")
    
    # Load existing data if resuming
    existing_data = {"images": []}
    existing_ids = set()
    
    if output_file.exists():
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                existing_ids = {img.get("image_id") for img in existing_data.get("images", [])}
            print(f"   📂 Resuming: {len(existing_ids)} images already processed")
        except Exception as e:
            print(f"   ⚠️ Could not load existing file: {e}")
    
    # Filter out already processed images
    images_to_process = [img for img in region_images if img["filename"] not in existing_ids]
    
    if not images_to_process:
        print(f"   ✅ All images already processed!")
        return
    
    print(f"   ⚙️ Processing {len(images_to_process)} new images...")
    
    # Process in batches
    session_input_tokens = 0
    session_output_tokens = 0
    
    for i in range(0, len(images_to_process), BATCH_SIZE):
        batch = images_to_process[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (len(images_to_process) + BATCH_SIZE - 1) // BATCH_SIZE
        
        print(f"   ⚙️ Processing Batch {batch_num}/{total_batches} ({len(batch)} images)...")
        
        results = process_image_batch(batch, project_id, page_context=page_context)
        
        if results:
            existing_data["images"].extend(results)
            
            # Save incrementally
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=2, ensure_ascii=False)
            
            print(f"      ✅ Saved {len(results)} images")
        else:
            print(f"      ⚠️ No results for this batch")
    
    print(f"   ✅ Complete! Total images: {len(existing_data['images'])}")
    print(f"   💾 Saved to: {output_file}")


def main():
    """Main entry point"""
    global client
    
    print("="*60)
    print("Structured Information Extraction for Engineering Drawings")
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
        process_project(project_id)
    else:
        # Process all projects in test_embeddings directory
        test_embeddings_path = Path(TEST_EMBEDDINGS_DIR)
        
        if not test_embeddings_path.exists():
            print(f"❌ Test embeddings directory not found: {test_embeddings_path}")
            return
        
        # Find all project directories
        projects = sorted([d.name for d in test_embeddings_path.iterdir() if d.is_dir()])
        
        if not projects:
            print(f"❌ No project directories found in {test_embeddings_path}")
            return
        
        print(f"Found {len(projects)} project(s): {', '.join(projects)}")
        print()
        
        for project_id in projects:
            try:
                process_project(project_id)
            except Exception as e:
                print(f"❌ Error processing {project_id}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print("\n" + "="*60)
    print("✨ Processing Complete!")
    print("="*60)


if __name__ == "__main__":
    main()

