from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np


class ZoneManager:
    """Manages geometric surveillance zones (polygons or bounding boxes)."""

    def __init__(self, zones_def: Dict[str, Union[List[int], List[List[int]]]]):
        self.zones: Dict[str, np.ndarray] = {}
        self._parse_zones(zones_def)

    def _parse_zones(self, zones_def: Dict[str, Union[List[int], List[List[int]]]]):
        for name, geom in zones_def.items():
            if isinstance(geom, list):
                if len(geom) == 4 and all(isinstance(v, (int, float)) for v in geom):
                    # Bounding box format [x1, y1, x2, y2]
                    x1, y1, x2, y2 = map(int, geom)
                    poly = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
                else:
                    # Polygon format [[x, y], ...]
                    poly = np.array(geom, dtype=np.int32)
                self.zones[name] = poly

    def get_zone_for_point(self, point: Tuple[int, int]) -> Optional[str]:
        """Check which zone contains the point. Returns None if in intermediate unzoned space."""
        pt = (float(point[0]), float(point[1]))
        for name, poly in self.zones.items():
            if cv2.pointPolygonTest(poly, pt, False) >= 0:
                return name
        return None

    def draw(self, frame: np.ndarray, active_zone_counts: Optional[Dict[str, int]] = None):
        """Render zones on frame with semi-transparent tinted fill and high-contrast boundary."""
        overlay = frame.copy()
        palette = {
            "A": (60, 180, 75),       # Green
            "B": (230, 25, 75),       # Blue/Red
            "QUEUE": (245, 130, 49),  # Orange
        }

        for name, poly in self.zones.items():
            color = palette.get(name, (200, 200, 200))
            cv2.fillPoly(overlay, [poly], color)
            cv2.polylines(frame, [poly], True, (255, 255, 255), 2)

            # Label with count
            count_str = f": {active_zone_counts[name]}" if active_zone_counts and name in active_zone_counts else ""
            lx, ly = poly[0][0] + 10, poly[0][1] + 30
            cv2.putText(frame, f"ZONE {name}{count_str}", (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
