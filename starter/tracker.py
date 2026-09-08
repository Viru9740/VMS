from math import hypot


class CentroidTracker:
    """Simple baseline tracker using nearest-centroid association."""

    def __init__(self, max_match_distance=80, max_missed_frames=12):
        self.max_match_distance = max_match_distance
        self.max_missed_frames = max_missed_frames
        self.next_id = 1
        self.tracks = {}

    def update(self, detections):
        unmatched = set(range(len(detections)))

        for track_id, track in list(self.tracks.items()):
            best_idx = None
            best_dist = None
            tx, ty = track["centroid"]
            for idx in unmatched:
                dx, dy = detections[idx]["centroid"]
                dist = hypot(dx - tx, dy - ty)
                if dist <= self.max_match_distance and (best_dist is None or dist < best_dist):
                    best_idx = idx
                    best_dist = dist

            if best_idx is None:
                track["missed"] += 1
                if track["missed"] > self.max_missed_frames:
                    del self.tracks[track_id]
                continue

            detection = detections[best_idx]
            unmatched.remove(best_idx)
            track.update(detection)
            track["missed"] = 0

        for idx in unmatched:
            detection = detections[idx]
            self.tracks[self.next_id] = {**detection, "missed": 0}
            self.next_id += 1

        return {
            track_id: track
            for track_id, track in self.tracks.items()
            if track["missed"] == 0
        }
