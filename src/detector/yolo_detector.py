from typing import List, Optional
import cv2
import numpy as np

from .base import BaseDetector, Detection


class RealtimeYoloDetector(BaseDetector):
    """Deep-learning detector using YOLOv8 for real-time live webcam, RTSP, and real-world video."""

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.40,
        target_classes: Optional[List[str]] = None,
    ):
        from ultralytics import YOLO

        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold
        # Default real-world targets: people, vehicles, bags, packages
        self.target_classes = target_classes or [
            "person", "car", "truck", "bus", "motorcycle", "bicycle",
            "backpack", "handbag", "suitcase", "bottle", "box", "cell phone"
        ]

    def _extract_feature(self, crop: np.ndarray) -> np.ndarray:
        if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
            return np.zeros(24, dtype=np.float32)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256])
        cv2.normalize(hist_h, hist_h, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist_s, hist_s, 0, 1, cv2.NORM_MINMAX)
        feat = np.concatenate([hist_h.flatten(), hist_s.flatten()]).astype(np.float32)
        norm = np.linalg.norm(feat)
        if norm > 1e-6:
            feat = feat / norm
        return feat

    def detect(self, frame: np.ndarray, frame_idx: int = 0) -> List[Detection]:
        h, w, _ = frame.shape
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)[0]

        detections: List[Detection] = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            class_name = results.names[cls_id]

            if self.target_classes and class_name not in self.target_classes:
                continue

            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bx, by = max(0, x1), max(0, y1)
            bw = max(10, min(w - bx, x2 - x1))
            bh = max(10, min(h - by, y2 - y1))

            cx = int(bx + bw / 2)
            cy = int(by + bh / 2)

            crop = frame[by:by + bh, bx:bx + bw]
            feat = self._extract_feature(crop)

            detections.append(
                Detection(
                    bbox=[bx, by, bw, bh],
                    centroid=[cx, cy],
                    confidence=conf,
                    class_name=class_name,
                    feature_vector=feat,
                )
            )

        return detections
