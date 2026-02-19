"""
Image Preprocessor for OCR Enhancement
Applies multiple CV techniques to maximize OCR accuracy on any document type.
"""

import cv2
import numpy as np
from PIL import Image
import io
import base64


class ImagePreprocessor:
    """
    Multi-stage image preprocessing pipeline designed to maximize OCR accuracy.
    Handles: receipts, invoices, handwritten notes, ID cards, prescriptions, etc.
    """

    @staticmethod
    def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
        """Load image from raw bytes into OpenCV format."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image from provided bytes")
        return img

    @staticmethod
    def load_image_from_base64(base64_string: str) -> np.ndarray:
        """Load image from base64 encoded string."""
        # Strip data URL prefix if present
        if "," in base64_string:
            base64_string = base64_string.split(",", 1)[1]
        image_bytes = base64.b64decode(base64_string)
        return ImagePreprocessor.load_image_from_bytes(image_bytes)

    @staticmethod
    def to_grayscale(img: np.ndarray) -> np.ndarray:
        """Convert to grayscale if not already."""
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    @staticmethod
    def denoise(img: np.ndarray) -> np.ndarray:
        """Apply non-local means denoising for cleaner text edges."""
        if len(img.shape) == 3:
            return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        return cv2.fastNlMeansDenoising(img, None, 10, 7, 21)

    @staticmethod
    def enhance_contrast(img: np.ndarray) -> np.ndarray:
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
        gray = ImagePreprocessor.to_grayscale(img)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    @staticmethod
    def adaptive_threshold(img: np.ndarray) -> np.ndarray:
        """Apply adaptive thresholding for binarization."""
        gray = ImagePreprocessor.to_grayscale(img)
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

    @staticmethod
    def deskew(img: np.ndarray) -> np.ndarray:
        """Detect and correct document skew angle."""
        gray = ImagePreprocessor.to_grayscale(img)
        # Use Hough lines to detect skew
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100,
                                minLineLength=100, maxLineGap=10)
        if lines is not None and len(lines) > 0:
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                if abs(angle) < 45:  # Only consider near-horizontal lines
                    angles.append(angle)
            if angles:
                median_angle = np.median(angles)
                if abs(median_angle) > 0.5:  # Only correct if skew > 0.5 degrees
                    (h, w) = img.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                    img = cv2.warpAffine(img, M, (w, h),
                                         flags=cv2.INTER_CUBIC,
                                         borderMode=cv2.BORDER_REPLICATE)
        return img

    @staticmethod
    def remove_shadows(img: np.ndarray) -> np.ndarray:
        """Remove shadows from document images."""
        rgb_planes = cv2.split(img) if len(img.shape) == 3 else [img]
        result_planes = []
        for plane in rgb_planes:
            dilated = cv2.dilate(plane, np.ones((7, 7), np.uint8))
            bg = cv2.medianBlur(dilated, 21)
            diff = 255 - cv2.absdiff(plane, bg)
            norm = cv2.normalize(diff, None, alpha=0, beta=255,
                                 norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            result_planes.append(norm)
        if len(img.shape) == 3:
            return cv2.merge(result_planes)
        return result_planes[0]

    @staticmethod
    def sharpen(img: np.ndarray) -> np.ndarray:
        """Sharpen text edges for better OCR recognition."""
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        return cv2.filter2D(img, -1, kernel)

    @staticmethod
    def resize_for_ocr(img: np.ndarray, target_dpi: int = 300) -> np.ndarray:
        """
        Upscale small images to improve OCR accuracy.
        Target: at least 300 DPI equivalent.
        """
        h, w = img.shape[:2]
        # If image is small, upscale it
        if max(h, w) < 1000:
            scale = 2.0
            img = cv2.resize(img, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_CUBIC)
        # If image is too large, downscale to prevent memory issues
        elif max(h, w) > 4000:
            scale = 3000 / max(h, w)
            img = cv2.resize(img, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_AREA)
        return img

    @classmethod
    def full_pipeline(cls, image_bytes: bytes) -> np.ndarray:
        """
        Run the complete preprocessing pipeline for maximum OCR accuracy.
        
        Pipeline order:
        1. Load → 2. Resize → 3. Deskew → 4. Shadow Removal →
        5. Denoise → 6. Contrast Enhancement → 7. Sharpen
        """
        img = cls.load_image_from_bytes(image_bytes)
        img = cls.resize_for_ocr(img)
        img = cls.deskew(img)
        img = cls.remove_shadows(img)
        img = cls.denoise(img)
        img = cls.enhance_contrast(img)
        img = cls.sharpen(img)
        return img

    @classmethod
    def light_pipeline(cls, image_bytes: bytes) -> np.ndarray:
        """
        Lighter pipeline for already-clean digital documents (PDFs, screenshots).
        """
        img = cls.load_image_from_bytes(image_bytes)
        img = cls.resize_for_ocr(img)
        img = cls.to_grayscale(img)
        img = cls.enhance_contrast(img)
        return img

    @classmethod
    def camera_pipeline(cls, base64_image: str) -> np.ndarray:
        """
        Pipeline optimized for camera-captured documents.
        Extra steps for shadow removal and deskew.
        """
        img = cls.load_image_from_base64(base64_image)
        img = cls.resize_for_ocr(img)
        img = cls.deskew(img)
        img = cls.remove_shadows(img)
        img = cls.denoise(img)
        img = cls.enhance_contrast(img)
        img = cls.sharpen(img)
        return img
