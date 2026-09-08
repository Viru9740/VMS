from collections import deque
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from ..tracker.byte_tracker import STrack


class VisualAnnotator:
    """Renders professional VMS telemetry overlays, motion trails, HUD and event alerts."""

    COLOR_PALETTE = [
        (50, 168, 82),    # Emerald green
        (52, 152, 219),   # Sky blue
        (155, 89, 182),   # Amethyst purple
        (241, 196, 15),   # Sun yellow
        (230, 126, 34),   # Carrot orange
        (26, 188, 156),   # Turquoise
        (231, 76, 60),    # Crimson
        (243, 156, 18),   # Amber
    ]

    def __init__(self, scenario_id: str = "S01_BASIC_GOODS"):
        self.scenario_id = scenario_id
        self.track_trails: Dict[int, deque] = {}
        self.recent_events: deque = deque(maxlen=3)

    def register_event(self, event_type: str, track_id: int, time_s: float):
        self.recent_events.append({
            "text": f"{event_type} (Track #{track_id})",
            "time": time_s,
            "display_until": time_s + 2.0,
        })

    def draw_hud(
        self,
        frame: np.ndarray,
        frame_idx: int,
        fps_realtime: float,
        time_s: float,
        active_track_count: int,
        event_counts: Dict[str, int],
        queue_occupancy: Optional[int] = None,
    ):
        h, w, _ = frame.shape
        hud_h = 56

        # Semi-transparent top HUD bar
        hud_overlay = frame[:hud_h, :].copy()
        cv2.rectangle(hud_overlay, (0, 0), (w, hud_h), (18, 22, 28), -1)
        cv2.addWeighted(hud_overlay, 0.85, frame[:hud_h, :], 0.15, 0, frame[:hud_h, :])
        cv2.line(frame, (0, hud_h), (w, hud_h), (50, 60, 75), 1)

        # Telemetry columns
        col1 = f"NOP INTELLIGENCE | {self.scenario_id} | t={time_s:05.2f}s (F#{frame_idx:04d})"
        cv2.putText(frame, col1, (16, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (240, 240, 240), 2)

        queue_str = f" | Queue: {queue_occupancy}" if queue_occupancy is not None else ""
        col2 = f"Active Tracks: {active_track_count}{queue_str} | Pipeline: {fps_realtime:04.1f} FPS"
        cv2.putText(frame, col2, (16, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 200, 240), 1)

        # Right-hand cumulative event tallies
        counts_str = " | ".join(f"{k}: {v}" for k, v in sorted(event_counts.items())) if event_counts else "No events yet"
        (tw, _), _ = cv2.getTextSize(f"EVENTS: {counts_str}", cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        cv2.putText(frame, f"EVENTS: {counts_str}", (max(w // 2, w - tw - 20), 34), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (100, 255, 140), 1)

    def draw_tracks(
        self,
        frame: np.ndarray,
        tracks: List[STrack],
        track_zones: Dict[int, Optional[str]],
        track_dwells: Optional[Dict[int, float]] = None,
    ):
        for track in tracks:
            tid = track.track_id
            color = self.COLOR_PALETTE[tid % len(self.COLOR_PALETTE)]
            bx, by, bw, bh = track.to_tlwh()
            cx, cy = track.centroid

            # Update trail
            if tid not in self.track_trails:
                self.track_trails[tid] = deque(maxlen=20)
            self.track_trails[tid].append((cx, cy))

            # Draw motion trail with fading thickness
            pts = list(self.track_trails[tid])
            for i in range(1, len(pts)):
                thickness = int(round(1 + (i / len(pts)) * 3))
                cv2.line(frame, pts[i - 1], pts[i], color, thickness)

            # Draw bounding box
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), color, 2)
            cv2.circle(frame, (cx, cy), 4, color, -1)

            # Header badge
            zone = track_zones.get(tid)
            zone_str = f" [{zone}]" if zone else ""
            dwell_str = ""
            if track_dwells and tid in track_dwells and track_dwells[tid] > 0.5:
                dwell_str = f" ({track_dwells[tid]:.1f}s)"

            label = f"ID #{tid} {track.class_name}{zone_str}{dwell_str}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            label_y1 = max(10, by - lh - 8)
            cv2.rectangle(frame, (bx, label_y1), (bx + lw + 8, label_y1 + lh + 6), color, -1)
            cv2.putText(frame, label, (bx + 4, label_y1 + lh + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

    def draw_active_alerts(self, frame: np.ndarray, current_time_s: float):
        active = [e for e in self.recent_events if current_time_s <= e["display_until"]]
        if not active:
            return

        h, w, _ = frame.shape
        start_y = 70
        for e in active:
            txt = f"VMS EVENT LOGGED: {e['text']}"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(frame, (16, start_y), (24 + tw, start_y + th + 12), (30, 144, 255), -1)
            cv2.putText(frame, txt, (20, start_y + th + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            start_y += th + 20
