from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class QueueTrackStats:
    track_id: int
    entry_time_s: float
    last_seen_s: float
    accumulated_dwell_s: float = 0.0
    threshold_alert_fired: bool = False


class DwellQueueAnalyzer:
    """Competitive VMS Extension: Real-time Queue Occupancy, Dwell Time Analytics & Alerts."""

    def __init__(self, queue_zone_name: str = "QUEUE", dwell_threshold_s: float = 5.0):
        self.queue_zone_name = queue_zone_name
        self.dwell_threshold_s = dwell_threshold_s
        self.active_tracks: Dict[int, QueueTrackStats] = {}
        self.completed_tracks: Dict[int, QueueTrackStats] = {}
        self.occupancy_timeline: Dict[float, int] = {}

    def update(
        self,
        active_zone_tracks: Dict[int, Optional[str]],
        timestamp_s: float,
    ) -> List[Dict]:
        """Update queue dwell tracking and emit DWELL_THRESHOLD alert events."""
        events = []
        current_queue_members = set()

        for track_id, zone in active_zone_tracks.items():
            if zone == self.queue_zone_name:
                current_queue_members.add(track_id)
                if track_id not in self.active_tracks:
                    self.active_tracks[track_id] = QueueTrackStats(
                        track_id=track_id,
                        entry_time_s=timestamp_s,
                        last_seen_s=timestamp_s,
                    )
                else:
                    stats = self.active_tracks[track_id]
                    dt = timestamp_s - stats.last_seen_s
                    stats.accumulated_dwell_s += max(0.0, dt)
                    stats.last_seen_s = timestamp_s

                    # Check for threshold violation event
                    if stats.accumulated_dwell_s >= self.dwell_threshold_s and not stats.threshold_alert_fired:
                        stats.threshold_alert_fired = True
                        events.append({
                            "event_type": "DWELL_THRESHOLD",
                            "track_id": track_id,
                            "dwell_seconds": round(stats.accumulated_dwell_s, 2),
                            "zone": self.queue_zone_name,
                        })

        # Process departures from queue
        departed = set(self.active_tracks.keys()) - current_queue_members
        for track_id in departed:
            self.completed_tracks[track_id] = self.active_tracks.pop(track_id)

        # Record occupancy for this timestamp (rounded to 0.5s intervals)
        bucket = round(timestamp_s * 2) / 2.0
        self.occupancy_timeline[bucket] = len(current_queue_members)

        return events

    @property
    def current_occupancy(self) -> int:
        return len(self.active_tracks)

    def get_track_dwell(self, track_id: int) -> float:
        if track_id in self.active_tracks:
            return self.active_tracks[track_id].accumulated_dwell_s
        if track_id in self.completed_tracks:
            return self.completed_tracks[track_id].accumulated_dwell_s
        return 0.0

    def get_summary_metrics(self) -> Dict:
        all_dwells = [t.accumulated_dwell_s for t in list(self.active_tracks.values()) + list(self.completed_tracks.values())]
        peak = max(self.occupancy_timeline.values()) if self.occupancy_timeline else 0
        avg_dwell = sum(all_dwells) / len(all_dwells) if all_dwells else 0.0
        return {
            "queue_zone": self.queue_zone_name,
            "peak_occupancy": peak,
            "average_dwell_seconds": round(avg_dwell, 2),
            "total_objects_queued": len(all_dwells),
            "dwell_threshold_violations": sum(1 for d in all_dwells if d >= self.dwell_threshold_s),
        }
