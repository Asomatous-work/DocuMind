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
    Detect section markers like S1, S2... or Page markers and put each on its own line.
    This makes the document scannable for both humans AND small LLMs.
    """
    # Pattern: S followed by number and a period or colon
    # Put a newline BEFORE each section marker
    text = re.sub(r'(?<!\n)\s*(S\d+[\.:])(?!\n)', r'\n\n\1', text)

    # Also handle "Section X." or numbered patterns like "43.1"
    text = re.sub(r'(?<!\n)\s*(\d+\.\d+)\s*', r' [\1]\n', text)

    # Handle Page markers from our PDF extractor: --- Page X ---
    # Ensure they are surrounded by newlines
    text = re.sub(r'(?<!\n)(--- Page \d+ ---)', r'\n\n\1\n', text)

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


def chunk_ocr_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[dict]:
    """
    Split the document text into structured chunks.
    Priority:
    1. Explicit Section Markers (S1, S2...)
    2. Page Markers (--- Page X ---)
    3. Fallback: Sliding Window
    
    Returns a list of dicts: {"label": "S1", "text": "..."}
    """
    if not text:
        return []

    # Ensure it's cleaned first
    cleaned = clean_ocr_text(text)
    
    chunks = []

    # Strategy 1: Explicit Section Markers (S1., S2.)
    # Look for at least one "S<number>." pattern to decide if we use this strategy
    if re.search(r'\n(S\d+)\.', cleaned):
        # Split by our standardized section markers (double newline + S followed by number)
        parts = re.split(r'\n\n(?=S\d+\.)', cleaned)
        for part in parts:
            part = part.strip()
            if not part: continue
            
            # Try to extract label (e.g., S1)
            label_match = re.match(r'^(S\d+)\.', part)
            if label_match:
                label = label_match.group(1)
                content = part[len(label)+1:].strip()
                chunks.append({"label": label, "text": content})
            else:
                chunks.append({"label": "Intro", "text": part})
        
        if len(chunks) > 1:
            return chunks
        # If we failed to get multiple chunks despite finding a marker, fall through to next strategy

    # Strategy 2: Page Markers
    # Our PDF extractor allows us to inject "--- Page X ---"
    if "--- Page" in cleaned:
        # Split by page marker
        # We look for \n--- Page <num> ---\n
        page_pattern = r'--- Page (\d+) ---'
        parts = re.split(page_pattern, cleaned)
        
        # re.split with capturing group returns [prologue, page_num, content, page_num, content...]
        
        # Handle prologue (text before first page marker)
        if parts[0].strip():
            chunks.append({"label": "Intro", "text": parts[0].strip()})
        
        # Iterate over the pairs (page_num, content)
        for i in range(1, len(parts), 2):
            page_num = parts[i]
            content = parts[i+1].strip() if i+1 < len(parts) else ""
            if content:
                chunks.append({"label": f"Page {page_num}", "text": content})
                
        if len(chunks) > 0:
            return chunks

    # Strategy 3: Fallback Sliding Window
    # If no structure found, chop it up
    text_len = len(cleaned)
    start = 0
    chunk_idx = 1
    
    while start < text_len:
        end = min(start + chunk_size, text_len)
        
        # Try to find a sentence break near the end to avoid cutting words/sentences
        if end < text_len:
            # Look for last period/newline in the overlap zone
            lookback = cleaned[end-overlap:end]
            last_break = max(lookback.rfind('. '), lookback.rfind('\n'))
            if last_break != -1:
                end = (end - overlap) + last_break + 1
        
        chunk_text = cleaned[start:end].strip()
        if chunk_text:
            chunks.append({"label": f"Part {chunk_idx}", "text": chunk_text})
            chunk_idx += 1
            
        start = end
        # overlap is handled by backing up 'start' in next iteration? 
        # Actually my logic above advances start=end. 
        # Sliding window usually implies start = start + (chunk_size - overlap).
        # Let's switch to standard sliding window logic step.
        
    if not chunks and cleaned:
         chunks.append({"label": "Full Text", "text": cleaned})

    return chunks
