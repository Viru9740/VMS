import cv2
import numpy as np


def point_in_polygon(point, polygon):
    contour = np.array(polygon, dtype=np.int32)
    return cv2.pointPolygonTest(contour, tuple(map(float, point)), False) >= 0


class ZoneFlow:
    """Tracks last known zone per track and emits deduplicated A<->B transitions."""

    def __init__(self, zones):
        self.zones = zones
        self.last_zone = {}
        self.emitted = set()

    def locate(self, point):
        for name, polygon in self.zones.items():
            if point_in_polygon(point, polygon):
                return name
        return None

    def update(self, track_id, centroid):
        current = self.locate(centroid)
        previous = self.last_zone.get(track_id)
        event = None

        if current is not None:
            self.last_zone[track_id] = current

        if previous and current and previous != current:
            event_type = f"{previous}_TO_{current}"
            key = (track_id, event_type)
            if key not in self.emitted:
                self.emitted.add(key)
                event = {
                    "event_type": event_type,
                    "source_zone": previous,
                    "destination_zone": current,
                }
        return current, event


def draw_zones(frame, zones):
    for name, polygon in zones.items():
        contour = np.array(polygon, dtype=np.int32)
        cv2.polylines(frame, [contour], True, (255, 255, 255), 2)
        x, y = polygon[0]
        cv2.putText(frame, name, (x + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
