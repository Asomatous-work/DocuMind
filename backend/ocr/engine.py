"""
OCR Engine - High-accuracy text extraction using EasyOCR.
Optimized for speed and literal transcription.
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
    High-accuracy OCR engine built on EasyOCR with optimized speed.
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
            logger.info("🔄 Loading EasyOCR models (optimized for CPU)...")
            self._reader = easyocr.Reader(
                ["en"],
                gpu=False,
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
        Extract text from image bytes with optimized preprocessing.
        """
        start_time = time.time()

        # Select preprocessing pipeline based on source
        try:
            # We'll use light_pipeline as it's much faster and works for most digital docs
            processed = ImagePreprocessor.light_pipeline(image_bytes)
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            nparr = np.frombuffer(image_bytes, np.uint8)
            import cv2
            processed = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        # Run EasyOCR
        results = self._reader.readtext(
            processed,
            detail=1,
            paragraph=False, # Single pass for maximum speed
            width_ths=0.5,
            height_ths=0.5,
        )

        blocks = []
        text_lines = []
        total_confidence = 0.0

        for bbox, text, confidence in results:
            text = text.strip()
            if not text: continue
            
            block = {
                "text": text,
                "confidence": round(float(confidence), 4),
                "bbox": [[int(p[0]), int(p[1])] for p in bbox],
            }
            blocks.append(block)
            text_lines.append(text)
            total_confidence += confidence

        full_text = "\n".join(text_lines)
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
            f"📝 EasyOCR complete: {len(blocks)} blocks, "
            f"time: {processing_time}s"
        )
        return result

    def extract_from_base64(
        self,
        base64_image: str,
        source_type: str = "camera",
        detail: bool = True
    ) -> dict:
        """Extract text from a base64-encoded image."""
        try:
            if "," in base64_image:
                data = base64_image.split(",", 1)[1]
            else:
                data = base64_image
            import base64
            image_bytes = base64.b64decode(data)
        except Exception as e:
            logger.error(f"Failed to decode base64: {e}")
            raise

        return self.extract_text(image_bytes, source_type=source_type, detail=detail)
