"""
OCR Text Cleaner — Post-processing for raw OCR output.
Converts messy OCR text into clean, structured text that small LLMs can understand.
"""

import re
import logging

logger = logging.getLogger(__name__)


def clean_ocr_text(raw_text: str) -> str:
    """
    Full cleaning pipeline for OCR-extracted text.
    Makes the text structured, readable, and LLM-friendly.
    """
    if not raw_text:
        return raw_text

    text = raw_text

    # 1. Fix common OCR character mistakes
    text = fix_ocr_artifacts(text)

    # 2. Normalize whitespace
    text = normalize_whitespace(text)

    # 3. Split into logical sections (e.g., S1, S2, S3...)
    text = split_into_sections(text)

    # 4. Fix punctuation
    text = fix_punctuation(text)

    # 5. Final trim
    text = text.strip()

    logger.info(f"🧹 Text cleaned: {len(raw_text)} → {len(text)} chars")
    return text


def fix_ocr_artifacts(text: str) -> str:
    """Fix common OCR misreads."""
    replacements = {
        "websitelapp": "website/app",
        "changelmodify": "change/modify",
        "whenan": "when an",
        "a8": "as",
        "S1O": "S10",
        "SI:": "S1.",
        "S2 ": "S2. ",
        "51.": "S1.",
        "52.": "S2.",
        "53.": "S3.",
        "54.": "S4.",
        "55.": "S5.",
        "(e.g\"": "(e.g.",
        "Depending o ": "Depending on ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def normalize_whitespace(text: str) -> str:
    """Clean up extra spaces, tabs, and weird gaps."""
    # Collapse multiple spaces into one
    text = re.sub(r'[ \t]{2,}', ' ', text)
    # Remove spaces before punctuation
    text = re.sub(r'\s+([;:,.])', r'\1', text)
    # Normalize line endings
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text


def split_into_sections(text: str) -> str:
    """
    Detect section markers like S1, S2, S3... and put each on its own line.
    This makes the document scannable for both humans AND small LLMs.
    """
    # Pattern: S followed by number and a period or colon
    # Put a newline BEFORE each section marker
    text = re.sub(r'(?<!\n)\s*(S\d+[\.:])(?!\n)', r'\n\n\1', text)

    # Also handle "Section X." or numbered patterns like "43.1"
    text = re.sub(r'(?<!\n)\s*(\d+\.\d+)\s*', r' [\1]\n', text)

    return text


def fix_punctuation(text: str) -> str:
    """Fix OCR punctuation errors — semicolons that should be commas/periods."""
    # OCR often reads : as ; in certain fonts
    # Don't blindly replace, only fix patterns that look wrong
    # "information;" → "information,"  (mid-sentence semicolons → commas)
    # But keep actual semicolons in lists

    # Fix trailing colons that should be periods at end of items
    # e.g., "deface the website:" → "deface the website."
    # Only when followed by a newline or section break
    text = re.sub(r':\s*\n', '.\n', text)

    return text


def structure_for_llm(text: str, filename: str = "") -> str:
    """
    Final formatting specifically to help tiny LLMs.
    Adds a header and makes sections crystal clear.
    """
    header = f"DOCUMENT: {filename}\n{'='*50}\n\n" if filename else ""
    return header + text


def chunk_ocr_text(text: str) -> list[dict]:
    """
    Split the document text into structured chunks based on section markers
    or page markers. Returns a list of dicts: {"label": "S1" or "Page 1", "text": "..."}
    """
    if not text:
        return []

    # Ensure it's cleaned first
    cleaned = clean_ocr_text(text)
    
    # Split by: 
    # 1. Standard section markers: \n\nS1. 
    # 2. Page markers: --- Page 1 ---
    # Using a regex that captures the marker to identify the chunk type
    pattern = r'(?:\n\n|\r\n\r\n|^)(?P<marker>S\d+\.|--- Page \d+ ---)'
    
    # re.split but keeping the delimiter is tricky, we'll use finditer to segment manually
    chunks = []
    
    # Initial "Intro" if text doesn't start with a marker
    first_match = re.search(pattern, cleaned)
    if first_match and first_match.start() > 0:
        intro_text = cleaned[:first_match.start()].strip()
        if intro_text:
            chunks.append({"label": "Intro", "text": intro_text})
    elif not first_match:
        # No markers at all, treat whole text as Intro or split by length
        if len(cleaned) < 1500:
            return [{"label": "Intro", "text": cleaned.strip()}]
        else:
            # Fallback to fixed size chunks for massive unstructured text
            size = 2000
            for i in range(0, len(cleaned), size):
                chunks.append({
                    "label": f"Part {i//size + 1}", 
                    "text": cleaned[i:i+size].strip()
                })
            return chunks

    # Process segments
    items = list(re.finditer(pattern, cleaned))
    for i, match in enumerate(items):
        marker = match.group('marker').strip('- ')
        start = match.end()
        end = items[i+1].start() if i+1 < len(items) else len(cleaned)
        content = cleaned[start:end].strip()
        
        if content:
            # Clean up the marker label (e.g. "Page 1" or "S1.")
            label = marker.rstrip('.')
            chunks.append({"label": label, "text": content})
            
    return chunks
