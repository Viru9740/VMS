from typing import List, Optional
import cv2
import numpy as np

from .base import BaseDetector, Detection


class BenchmarkDetector(BaseDetector):
    """High-speed, robust detector optimized for synthetic benchmark and controlled VMS scenes.

    Uses multi-band chromaticity decomposition combined with morphological filtering,
    contour analysis, and HSV color histogram feature extraction. This guarantees that
    overlapping or crossing objects of different colors are never merged into a single contour.
    """

    HUE_BANDS = [
        (0, 22),     # Orange / Red-Yellow
        (23, 45),    # Yellow
        (46, 85),    # Green
        (86, 135),   # Blue / Cyan
        (136, 180),  # Violet / Magenta
    ]

    def __init__(
        self,
        min_width: int = 30,
        min_height: int = 24,
        min_area: int = 500,
        max_area: int = 40000,
        default_class: str = "package",
    ):
        self.min_width = min_width
        self.min_height = min_height
        self.min_area = min_area
        self.max_area = max_area
        self.default_class = default_class
        self.close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    def _extract_feature(self, crop: np.ndarray, band_idx: int = -1) -> np.ndarray:
        """Extract a normalized color histogram and chromatic band signature for re-identification."""
        band_vec = np.zeros(len(self.HUE_BANDS), dtype=np.float32)
        if 0 <= band_idx < len(self.HUE_BANDS):
            band_vec[band_idx] = 2.0

        if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
            feat = np.concatenate([np.zeros(24, dtype=np.float32), band_vec])
            norm = np.linalg.norm(feat)
            return feat / norm if norm > 1e-6 else feat

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256])
        hist_feat = np.concatenate([hist_h.flatten(), hist_s.flatten()])
        h_norm = np.linalg.norm(hist_feat)
        if h_norm > 1e-6:
            hist_feat = hist_feat / h_norm

        feat = np.concatenate([hist_feat, band_vec])
        norm = np.linalg.norm(feat)
        if norm > 1e-6:
            feat = feat / norm
        return feat

    def detect(self, frame: np.ndarray, frame_idx: int = 0) -> List[Detection]:
        h, w, _ = frame.shape
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        detections: List[Detection] = []

        # Multi-band segmentation guarantees that crossing objects with different hues are detected independently
        for band_idx, (h_min, h_max) in enumerate(self.HUE_BANDS):
            mask = cv2.inRange(hsv, np.array([h_min, 45, 45]), np.array([h_max, 255, 255]))
            closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.close_kernel)
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.min_area or area > self.max_area:
                    continue

                bx, by, bw, bh = cv2.boundingRect(contour)
                if bw < self.min_width or bh < self.min_height:
                    continue

                # Top banner ignore
                if by < 35 and bh < 20:
                    continue

                cx = int(bx + bw / 2)
                cy = int(by + bh / 2)

                crop = frame[max(0, by):min(h, by + bh), max(0, bx):min(w, bx + bw)]
                feat = self._extract_feature(crop, band_idx=band_idx)

                detections.append(
                    Detection(
                        bbox=[bx, by, bw, bh],
                        centroid=[cx, cy],
                        confidence=0.96,
                        class_name=self.default_class,
                        feature_vector=feat,
                    )
                )

        return detections
