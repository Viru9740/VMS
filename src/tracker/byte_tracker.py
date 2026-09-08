from collections import deque
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np

from ..detector.base import Detection
from .kalman_filter import KalmanFilter8D
from .matching import (
    bbox_ious,
    feature_cosine_distance,
    iou_cost_matrix,
    min_cost_matching,
)


class TrackState(Enum):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class STrack:
    """Individual object track maintaining continuous state, trajectory, and appearance."""

    _count = 0

    def __init__(self, detection: Detection, frame_idx: int, time_s: float):
        STrack._count += 1
        self.track_id = STrack._count
        self.state = TrackState.New
        self.is_activated = False

        self.kalman_filter = KalmanFilter8D()
        x, y, w, h = detection.bbox
        measurement = np.array([x + w / 2, y + h / 2, float(w) / max(1.0, float(h)), float(h)], dtype=np.float32)
        self.mean, self.covariance = self.kalman_filter.initiate(measurement)

        self.class_name = detection.class_name
        self.confidence = detection.confidence
        self.time_since_update = 0
        self.start_frame = frame_idx
        self.frame_idx = frame_idx

        self.features = deque(maxlen=25)
        if detection.feature_vector is not None:
            self.features.append(detection.feature_vector)

        self.trajectory: List[Tuple[int, float, int, int, List[int]]] = []
        cx, cy = int(measurement[0]), int(measurement[1])
        self.trajectory.append((frame_idx, time_s, cx, cy, [int(x), int(y), int(w), int(h)]))

    @classmethod
    def reset_counter(cls):
        cls._count = 0

    @property
    def current_feature(self) -> Optional[np.ndarray]:
        if not self.features:
            return None
        feat = np.mean(self.features, axis=0)
        norm = np.linalg.norm(feat)
        return feat / norm if norm > 1e-6 else feat

    def to_tlwh(self) -> List[int]:
        cx, cy, a, h = self.mean[:4]
        w = max(10.0, a * h)
        h = max(10.0, h)
        x = cx - w / 2
        y = cy - h / 2
        return [int(round(x)), int(round(y)), int(round(w)), int(round(h))]

    @property
    def centroid(self) -> Tuple[int, int]:
        cx, cy = self.mean[:2]
        return int(round(cx)), int(round(cy))

    def predict(self):
        if self.state != TrackState.Tracked:
            self.mean[4:8] *= 0.98
        self.mean, self.covariance = self.kalman_filter.predict(self.mean, self.covariance)

    def update(self, detection: Detection, frame_idx: int, time_s: float):
        self.frame_idx = frame_idx
        self.time_since_update = 0
        self.confidence = detection.confidence
        self.class_name = detection.class_name

        x, y, w, h = detection.bbox
        measurement = np.array([x + w / 2, y + h / 2, float(w) / max(1.0, float(h)), float(h)], dtype=np.float32)
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, measurement)

        if detection.feature_vector is not None:
            self.features.append(detection.feature_vector)

        self.state = TrackState.Tracked
        self.is_activated = True

        cx, cy = self.centroid
        self.trajectory.append((frame_idx, time_s, cx, cy, [int(x), int(y), int(w), int(h)]))

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed


class ByteTracker:
    """ByteTrack Multi-Object Tracker enhanced with appearance gating and Re-ID recovery."""

    def __init__(
        self,
        high_threshold: float = 0.40,
        low_threshold: float = 0.15,
        match_threshold: float = 0.70,
        reid_threshold: float = 0.55,
        max_lost_frames: int = 350,
        appearance_sim_threshold: float = 0.65,
    ):
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.match_threshold = match_threshold
        self.reid_threshold = reid_threshold
        self.max_lost_frames = max_lost_frames
        self.appearance_sim_threshold = appearance_sim_threshold

        self.tracked_stracks: List[STrack] = []
        self.lost_stracks: List[STrack] = []
        self.removed_stracks: List[STrack] = []
        self.frame_id = 0

    def reset(self):
        STrack.reset_counter()
        self.tracked_stracks.clear()
        self.lost_stracks.clear()
        self.removed_stracks.clear()
        self.frame_id = 0

    def update(self, detections: List[Detection], frame_idx: int, time_s: float) -> List[STrack]:
        self.frame_id = frame_idx

        # 1. Partition detections into high and low confidence
        det_high: List[Detection] = []
        det_low: List[Detection] = []
        for det in detections:
            if det.confidence >= self.high_threshold:
                det_high.append(det)
            elif det.confidence >= self.low_threshold:
                det_low.append(det)

        # 2. Kalman predict for all active and lost tracks
        for track in self.tracked_stracks:
            track.predict()
        for track in self.lost_stracks:
            track.predict()

        # 3. Stage 1: Match active tracks with high-confidence detections + Appearance Gating
        strack_pool = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        tracks_boxes = np.array([t.to_tlwh() for t in strack_pool], dtype=np.float32) if strack_pool else np.empty((0, 4))
        high_boxes = np.array([d.bbox for d in det_high], dtype=np.float32) if det_high else np.empty((0, 4))

        cost_matrix = iou_cost_matrix(tracks_boxes, high_boxes)

        # Apply appearance gating: block spatial association if color similarity is low
        t_feats = [t.current_feature for t in strack_pool]
        d_feats = [d.feature_vector for d in det_high]
        for ti, tf in enumerate(t_feats):
            if tf is None:
                continue
            for di, df in enumerate(d_feats):
                if df is None:
                    continue
                sim = float(np.dot(tf, df))
                if sim < self.appearance_sim_threshold:
                    cost_matrix[ti, di] = 1.0  # Gate out cross-identity switches

        matches_s1, u_tracks_s1, u_det_high = min_cost_matching(cost_matrix, self.match_threshold)

        for track_idx, det_idx in matches_s1:
            strack_pool[track_idx].update(det_high[det_idx], frame_idx, time_s)

        # 4. Stage 2: Match remaining active tracks with low-confidence detections
        unmatched_tracks = [strack_pool[i] for i in u_tracks_s1]
        u_tracks_boxes = np.array([t.to_tlwh() for t in unmatched_tracks], dtype=np.float32) if unmatched_tracks else np.empty((0, 4))
        low_boxes = np.array([d.bbox for d in det_low], dtype=np.float32) if det_low else np.empty((0, 4))

        cost_matrix_low = iou_cost_matrix(u_tracks_boxes, low_boxes)

        # Apply appearance gating in stage 2 as well
        t_feats_low = [t.current_feature for t in unmatched_tracks]
        d_feats_low = [d.feature_vector for d in det_low]
        for ti, tf in enumerate(t_feats_low):
            if tf is None:
                continue
            for di, df in enumerate(d_feats_low):
                if df is None:
                    continue
                sim = float(np.dot(tf, df))
                if sim < self.appearance_sim_threshold:
                    cost_matrix_low[ti, di] = 1.0

        matches_s2, u_tracks_s2, _ = min_cost_matching(cost_matrix_low, 0.50)

        for track_idx, det_idx in matches_s2:
            unmatched_tracks[track_idx].update(det_low[det_idx], frame_idx, time_s)

        for track_idx in u_tracks_s2:
            unmatched_tracks[track_idx].mark_lost()

        # 5. Stage 3: Match remaining high-confidence detections with lost tracks (Re-ID recovery)
        remaining_high_indices = list(u_det_high)
        if self.lost_stracks and remaining_high_indices:
            lost_with_feats = [t for t in self.lost_stracks if t.current_feature is not None]
            det_with_feats_idx = [i for i in remaining_high_indices if det_high[i].feature_vector is not None]

            if lost_with_feats and det_with_feats_idx:
                lost_feats = np.array([t.current_feature for t in lost_with_feats])
                det_feats = np.array([det_high[i].feature_vector for i in det_with_feats_idx])

                feat_cost = feature_cosine_distance(lost_feats, det_feats)
                matches_reid, _, u_det_reid_cols = min_cost_matching(feat_cost, self.reid_threshold)
                matched_lost_ids = set()
                matched_det_orig_indices = set()

                for lost_idx, det_col in matches_reid:
                    track = lost_with_feats[lost_idx]
                    orig_det_idx = det_with_feats_idx[det_col]
                    track.update(det_high[orig_det_idx], frame_idx, time_s)
                    self.tracked_stracks.append(track)  # Re-activate lost track!
                    matched_lost_ids.add(track.track_id)
                    matched_det_orig_indices.add(orig_det_idx)

                self.lost_stracks = [t for t in self.lost_stracks if t.track_id not in matched_lost_ids]
                remaining_high_indices = [i for i in remaining_high_indices if i not in matched_det_orig_indices]

        # 6. Initialize new tracks from truly unmatched high-confidence detections
        for det_idx in remaining_high_indices:
            det = det_high[det_idx]
            new_track = STrack(det, frame_idx, time_s)
            new_track.state = TrackState.Tracked
            new_track.is_activated = True
            self.tracked_stracks.append(new_track)

        # 7. Update active and lost lists
        new_tracked = []
        new_lost = []
        for track in self.tracked_stracks:
            if track.state == TrackState.Tracked:
                new_tracked.append(track)
            elif track.state == TrackState.Lost:
                new_lost.append(track)

        self.lost_stracks.extend(new_lost)
        self.tracked_stracks = new_tracked

        # Age lost tracks and prune those exceeding max_lost_frames
        active_lost = []
        for track in self.lost_stracks:
            track.time_since_update += 1
            if track.time_since_update <= self.max_lost_frames:
                active_lost.append(track)
            else:
                track.mark_removed()
                self.removed_stracks.append(track)
        self.lost_stracks = active_lost

        return [t for t in self.tracked_stracks if t.is_activated]
