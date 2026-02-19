"""
Document Extractor — Text extraction for DOCX, DOC, and PDF files.
Bypasses OCR; reads text directly from the document structure.
"""

import io
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def _make_result(text: str, source_type: str, processing_time: float) -> dict:
    """
    Build a result dict that matches the OCREngine output format,
    so document results can be stored in the knowledge base with the
    same `add_document` call.
    """
    # Split into pseudo-blocks (one per non-empty line/paragraph)
    raw_lines = [line.strip() for line in text.splitlines() if line.strip()]
    blocks = [
        {
            "text": line,
            "confidence": 1.0,   # Native extraction is 100% confident
            "bbox": [],          # No bounding boxes for text documents
        }
        for line in raw_lines
    ]

    return {
        "text": text,
        "blocks": blocks,
        "block_count": len(blocks),
        "avg_confidence": 1.0,
        "processing_time_seconds": round(processing_time, 2),
        "source_type": source_type,
    }


def extract_text_from_docx(file_bytes: bytes) -> dict:
    """
    Extract text from a .docx file using python-docx.

    Args:
        file_bytes: Raw .docx file bytes.

    Returns:
        Result dict compatible with OCREngine output.
    """
    start = time.time()
    try:
        import docx  # python-docx
    except ImportError:
        raise RuntimeError(
            "python-docx is not installed. Run: pip install python-docx"
        )

    doc = docx.Document(io.BytesIO(file_bytes))

    paragraphs = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            paragraphs.append(text)

    # Also extract text from tables
    for table in doc.tables:
        for row in table.rows:
            row_text = " | ".join(
                cell.text.strip() for cell in row.cells if cell.text.strip()
            )
            if row_text:
                paragraphs.append(row_text)

    full_text = "\n".join(paragraphs)
    processing_time = time.time() - start

    logger.info(
        f"📄 DOCX extracted: {len(paragraphs)} paragraphs in {processing_time:.2f}s"
    )
    return _make_result(full_text, "docx", processing_time)


def extract_text_from_doc(file_bytes: bytes) -> dict:
    """
    Extract text from a legacy .doc file.

    .doc (Word 97-2003) is a binary format. python-docx cannot read it.
    We raise a clear error asking the user to convert to DOCX or PDF first.
    """
    raise ValueError(
        "Legacy .doc files (Word 97–2003) are not supported for direct text extraction. "
        "Please save the file as .docx or .pdf and re-upload."
    )


def extract_text_from_pdf(file_bytes: bytes) -> dict:
    """
    Extract text from a PDF file using PyMuPDF (fitz).

    Handles text-based PDFs efficiently. For scanned/image PDFs,
    this returns empty text (user should use image upload + OCR instead).

    Args:
        file_bytes: Raw PDF file bytes.

    Returns:
        Result dict compatible with OCREngine output.
    """
    start = time.time()
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise RuntimeError(
            "PyMuPDF is not installed. Run: pip install PyMuPDF"
        )

    pdf = fitz.open(stream=file_bytes, filetype="pdf")

    pages_text = []
    for page_num, page in enumerate(pdf, start=1):
        page_text = page.get_text("text").strip()
        if page_text:
            pages_text.append(f"--- Page {page_num} ---\n{page_text}")

    pdf.close()

    full_text = "\n\n".join(pages_text)
    processing_time = time.time() - start

    if not full_text.strip():
        logger.warning(
            "⚠️ PDF appears to be scanned/image-based — no text layer found. "
            "Consider uploading as an image for OCR processing."
        )

    logger.info(
        f"📄 PDF extracted: {len(pages_text)} pages in {processing_time:.2f}s"
    )
    return _make_result(full_text, "pdf", processing_time)


# MIME type and extension maps used by main.py
DOCUMENT_MIME_TYPES = {
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/msword": "doc",
    "application/pdf": "pdf",
}

DOCUMENT_EXTENSIONS = {
    ".docx": "docx",
    ".doc": "doc",
    ".pdf": "pdf",
}


def extract_document(file_bytes: bytes, filename: str, mime_type: str) -> dict:
    """
    Route to the correct extractor based on MIME type or file extension.

    Args:
        file_bytes: Raw file bytes.
        filename: Original filename (used as fallback for type detection).
        mime_type: MIME type reported by the browser.

    Returns:
        Result dict compatible with OCREngine output.
    """
    import os
    ext = os.path.splitext(filename.lower())[1]

    # Prefer MIME type; fall back to file extension
    doc_type = DOCUMENT_MIME_TYPES.get(mime_type) or DOCUMENT_EXTENSIONS.get(ext)

    if doc_type == "docx":
        return extract_text_from_docx(file_bytes)
    elif doc_type == "doc":
        return extract_text_from_doc(file_bytes)
    elif doc_type == "pdf":
        return extract_text_from_pdf(file_bytes)
    else:
        raise ValueError(
            f"Unsupported document type: mime='{mime_type}', ext='{ext}'"
        )
