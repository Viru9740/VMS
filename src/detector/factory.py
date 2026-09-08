from typing import Dict, Any
from .base import BaseDetector
from .benchmark_detector import BenchmarkDetector
from .onnx_detector import OnnxDetector
from .yolo_detector import RealtimeYoloDetector


def create_detector(config: Dict[str, Any]) -> BaseDetector:
    """Instantiate the requested detector based on configuration."""
    detector_type = config.get("detector_type", "benchmark").lower()

    if detector_type in {"yolo", "realtime", "real_world"}:
        return RealtimeYoloDetector(
            model_name=config.get("yolo_model", "yolov8n.pt"),
            confidence_threshold=config.get("min_confidence", 0.35),
            target_classes=config.get("target_classes"),
        )
    elif detector_type == "onnx":
        return OnnxDetector(
            model_path=config.get("onnx_model_path"),
            confidence_threshold=config.get("min_confidence", 0.40),
            target_classes=config.get("target_classes"),
        )
    return BenchmarkDetector(
        min_width=config.get("min_width", 30),
        min_height=config.get("min_height", 24),
        min_area=config.get("min_area", 500),
        default_class=config.get("default_class", "package"),
    )
