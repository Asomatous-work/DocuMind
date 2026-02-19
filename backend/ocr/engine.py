"""
OCR Engine - High-accuracy text extraction using EasyOCR.
Supports multiple languages and document types.
"""

import easyocr
import numpy as np
from typing import Optional
from .preprocessor import ImagePreprocessor
from .text_cleaner import clean_ocr_text
import logging
import time

logger = logging.getLogger(__name__)


class OCREngine:
    """
    High-accuracy OCR engine built on EasyOCR with intelligent preprocessing.
    """

    _instance: Optional["OCREngine"] = None
    _reader: Optional[easyocr.Reader] = None

    def __new__(cls):
        """Singleton pattern - EasyOCR model loading is expensive."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._reader is None:
            logger.info("🔄 Loading EasyOCR models (first time takes ~30s)...")
            self._reader = easyocr.Reader(
                ["en"],  # Add more languages as needed: ['en', 'hi', 'ta']
                gpu=False,  # Set True if CUDA available
                verbose=False
            )
            logger.info("✅ EasyOCR models loaded successfully")

    def extract_text(
        self,
        image_bytes: bytes,
        source_type: str = "upload",
        detail: bool = True
    ) -> dict:
        """
        Extract text from image bytes with full preprocessing.

        Args:
            image_bytes: Raw image file bytes.
            source_type: 'upload' | 'camera' | 'digital' — selects preprocessing pipeline.
            detail: If True, return bounding boxes and confidence scores.

        Returns:
            dict with keys: text, blocks, confidence, processing_time
        """
        start_time = time.time()

        # Select preprocessing pipeline based on source
        try:
            if source_type == "camera":
                processed = ImagePreprocessor.full_pipeline(image_bytes)
            elif source_type == "digital":
                processed = ImagePreprocessor.light_pipeline(image_bytes)
            else:
                processed = ImagePreprocessor.full_pipeline(image_bytes)
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            # Fallback: try raw image
            nparr = np.frombuffer(image_bytes, np.uint8)
            import cv2
            processed = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        # Run EasyOCR
        results = self._reader.readtext(
            processed,
            detail=1,  # Always get full details
            paragraph=True,  # Merge into paragraphs
            width_ths=0.7,
            height_ths=0.7,
        )

        # Also run without paragraph merging for structured data
        results_raw = self._reader.readtext(
            processed,
            detail=1,
            paragraph=False,
        )

        # Build structured output
        blocks = []
        full_text_parts = []
        total_confidence = 0.0

        for bbox, text, confidence in results_raw:
            block = {
                "text": text.strip(),
                "confidence": round(float(confidence), 4),
                "bbox": [[int(p[0]), int(p[1])] for p in bbox],
            }
            blocks.append(block)
            total_confidence += confidence

        # Use paragraph-merged text for the full text
        for item in results:
            if len(item) >= 2:
                full_text_parts.append(item[1].strip())

        full_text = "\n".join(full_text_parts) if full_text_parts else ""
        # Clean and structure the text for LLM consumption
        full_text = clean_ocr_text(full_text)
        avg_confidence = (total_confidence / len(blocks)) if blocks else 0.0

        processing_time = round(time.time() - start_time, 2)

        result = {
            "text": full_text,
            "blocks": blocks if detail else [],
            "block_count": len(blocks),
            "avg_confidence": round(avg_confidence, 4),
            "processing_time_seconds": processing_time,
            "source_type": source_type,
        }

        logger.info(
            f"📝 OCR complete: {len(blocks)} blocks, "
            f"avg confidence: {avg_confidence:.2%}, "
            f"time: {processing_time}s"
        )

        return result

    def extract_from_base64(
        self,
        base64_image: str,
        source_type: str = "camera",
        detail: bool = True
    ) -> dict:
        """Extract text from a base64-encoded image (camera capture)."""
        try:
            processed = ImagePreprocessor.camera_pipeline(base64_image)
        except Exception as e:
            logger.error(f"Camera preprocessing failed: {e}")
            raise

        start_time = time.time()

        results = self._reader.readtext(
            processed,
            detail=1,
            paragraph=True,
            width_ths=0.7,
            height_ths=0.7,
        )

        results_raw = self._reader.readtext(
            processed,
            detail=1,
            paragraph=False,
        )

        blocks = []
        full_text_parts = []
        total_confidence = 0.0

        for bbox, text, confidence in results_raw:
            block = {
                "text": text.strip(),
                "confidence": round(float(confidence), 4),
                "bbox": [[int(p[0]), int(p[1])] for p in bbox],
            }
            blocks.append(block)
            total_confidence += confidence

        for item in results:
            if len(item) >= 2:
                full_text_parts.append(item[1].strip())

        full_text = "\n".join(full_text_parts) if full_text_parts else ""
        full_text = clean_ocr_text(full_text)
        avg_confidence = (total_confidence / len(blocks)) if blocks else 0.0
        processing_time = round(time.time() - start_time, 2)

        return {
            "text": full_text,
            "blocks": blocks if detail else [],
            "block_count": len(blocks),
            "avg_confidence": round(avg_confidence, 4),
            "processing_time_seconds": processing_time,
            "source_type": source_type,
        }
