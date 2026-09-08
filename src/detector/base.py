from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class Detection:
    """Standard detection representation for the VMS pipeline."""
    bbox: List[int]  # [x, y, w, h]
    centroid: List[int]  # [cx, cy]
    confidence: float  # [0.0, 1.0]
    class_name: str  # e.g. "package", "person", "vehicle"
    feature_vector: Optional[np.ndarray] = None  # Normalized appearance feature embedding
    raw_mask: Optional[np.ndarray] = None


class BaseDetector(ABC):
    """Abstract interface for all detector implementations."""

    @abstractmethod
    def detect(self, frame: np.ndarray, frame_idx: int = 0) -> List[Detection]:
        """Process a BGR video frame and return detected targets."""
        pass
