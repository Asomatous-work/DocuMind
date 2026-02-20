"""
Image Preprocessor for OCR Enhancement
Optimized for speed and accuracy.
"""

import cv2
import numpy as np
from PIL import Image
import io
import base64


class ImagePreprocessor:
    """
    Multi-stage image preprocessing pipeline designed to maximize OCR accuracy.
    """

    @staticmethod
    def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
        """Load image into OpenCV format."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image")
        return img

    @staticmethod
    def to_grayscale(img: np.ndarray) -> np.ndarray:
        """Convert to grayscale."""
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    @staticmethod
    def enhance_contrast(img: np.ndarray) -> np.ndarray:
        """Apply CLAHE for contrast."""
        gray = ImagePreprocessor.to_grayscale(img)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    @staticmethod
    def deskew(img: np.ndarray) -> np.ndarray:
        """Fast deskewing."""
        gray = ImagePreprocessor.to_grayscale(img)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=5)
        
        if lines is not None:
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angles.append(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            
            median_angle = np.median(angles)
            if abs(median_angle) > 0.5:
                (h, w) = img.shape[:2]
                M = cv2.getRotationMatrix2D((w // 2, h // 2), median_angle, 1.0)
                img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)
        return img

    @staticmethod
    def resize_for_ocr(img: np.ndarray) -> np.ndarray:
        """Optimize size for OCR."""
        h, w = img.shape[:2]
        if max(h, w) < 800:
            img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        elif max(h, w) > 2500:
            scale = 2000 / max(h, w)
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return img

    @classmethod
    def light_pipeline(cls, image_bytes: bytes) -> np.ndarray:
        """Fastest pipeline for digital docs/screenshots."""
        img = cls.load_image_from_bytes(image_bytes)
        img = cls.to_grayscale(img)
        # Digital images usually don't need much, just clear contrast
        return img

    @classmethod
    def camera_pipeline(cls, image_bytes: bytes, raw_bytes: bytes = None) -> np.ndarray:
        """Balanced pipeline for photos/camera."""
        if image_bytes is None and raw_bytes is not None:
            img = cls.load_image_from_bytes(raw_bytes)
        else:
            img = cls.load_image_from_bytes(image_bytes)
            
        img = cls.resize_for_ocr(img)
        img = cls.deskew(img)
        img = cls.enhance_contrast(img)
        # Removed denoise as it's too slow on CPU
        return img
