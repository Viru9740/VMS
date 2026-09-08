from typing import List, Optional
import os
import cv2
import numpy as np

from .base import BaseDetector, Detection


class OnnxDetector(BaseDetector):
    """Deep-learning detector using ONNX Runtime for real-world and unseen video generalization.

    Supports YOLOv8 / YOLO11 format ONNX models on CPU or DirectML.
    Includes adaptive MOG2/contour fallback when an ONNX model file is not supplied.
    """

    COCO_CLASSES = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ]

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.40,
        nms_threshold: float = 0.45,
        target_classes: Optional[List[str]] = None,
        input_size: int = 640,
    ):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.target_classes = target_classes or ["person", "backpack", "suitcase", "truck", "car"]
        self.input_size = input_size
        self.session = None

        if model_path and os.path.exists(model_path):
            try:
                import onnxruntime as ort
                self.session = ort.InferenceSession(
                    model_path,
                    providers=["CPUExecutionProvider"]
                )
                self.input_name = self.session.get_inputs()[0].name
            except Exception as e:
                print(f"[OnnxDetector] Notice: Failed to initialize ONNX session: {e}. Falling back to adaptive motion.")
                self.session = None

        # Adaptive background subtractor fallback
        self.subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=24, detectShadows=True)

    def _letterbox(self, img: np.ndarray):
        shape = img.shape[:2]
        r = min(self.input_size / shape[0], self.input_size / shape[1])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw = (self.input_size - new_unpad[0]) / 2
        dh = (self.input_size - new_unpad[1]) / 2

        resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return padded, r, dw, dh

    def detect(self, frame: np.ndarray, frame_idx: int = 0) -> List[Detection]:
        if self.session is not None:
            return self._detect_onnx(frame)
        return self._detect_adaptive_motion(frame)

    def _detect_onnx(self, frame: np.ndarray) -> List[Detection]:
        padded, r, dw, dh = self._letterbox(frame)
        blob = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        blob = blob.transpose((2, 0, 1)).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, axis=0)

        outputs = self.session.run(None, {self.input_name: blob})[0]
        # Format: (1, 84, 8400) for YOLOv8
        output = np.squeeze(outputs).T
        scores = np.max(output[:, 4:], axis=1)
        class_ids = np.argmax(output[:, 4:], axis=1)

        mask = scores >= self.confidence_threshold
        output = output[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        if len(scores) == 0:
            return []

        boxes = []
        for row in output[:, :4]:
            cx, cy, w, h = row
            x1 = int((cx - w / 2 - dw) / r)
            y1 = int((cy - h / 2 - dh) / r)
            boxes.append([x1, y1, int(w / r), int(h / r)])

        indices = cv2.dnn.NMSBoxes(boxes, scores.tolist(), self.confidence_threshold, self.nms_threshold)
        detections = []
        for i in indices:
            idx = i if isinstance(i, (int, np.integer)) else i[0]
            cid = class_ids[idx]
            cname = self.COCO_CLASSES[cid] if cid < len(self.COCO_CLASSES) else "object"
            if self.target_classes and cname not in self.target_classes:
                continue
            x, y, w, h = boxes[idx]
            detections.append(
                Detection(
                    bbox=[x, y, w, h],
                    centroid=[int(x + w / 2), int(y + h / 2)],
                    confidence=float(scores[idx]),
                    class_name=cname,
                )
            )
        return detections

    def _detect_adaptive_motion(self, frame: np.ndarray) -> List[Detection]:
        mask = self.subtractor.apply(frame)
        _, mask = cv2.threshold(mask, 220, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for c in contours:
            if cv2.contourArea(c) < 800:
                continue
            x, y, w, h = cv2.boundingRect(c)
            detections.append(
                Detection(
                    bbox=[x, y, w, h],
                    centroid=[int(x + w / 2), int(y + h / 2)],
                    confidence=0.75,
                    class_name="moving_object",
                )
            )
        return detections
