from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class TrackJourney:
    track_id: int
    origin_zone: Optional[str] = None
    current_zone: Optional[str] = None
    last_zone: Optional[str] = None
    zone_entry_time: Dict[str, float] = field(default_factory=dict)
    zone_dwell_accum: Dict[str, float] = field(default_factory=dict)
    history: List[str] = field(default_factory=list)
    emitted_events: Set[str] = field(default_factory=set)


class JourneyTracker:
    """Multi-zone trajectory state machine tracking full route journeys and preventing duplicate counts."""

    def __init__(self):
        self.journeys: Dict[int, TrackJourney] = {}

    def update(
        self,
        track_id: int,
        current_zone: Optional[str],
        timestamp_s: float,
    ) -> Optional[Dict]:
        """Update track zone state and emit completion events (e.g. A_TO_B or B_TO_A)."""
        if track_id not in self.journeys:
            self.journeys[track_id] = TrackJourney(track_id=track_id)

        j = self.journeys[track_id]
        event = None

        # Zone transition detection
        if current_zone != j.current_zone:
            # Leaving previous zone
            if j.current_zone is not None:
                dwell = timestamp_s - j.zone_entry_time.get(j.current_zone, timestamp_s)
                j.zone_dwell_accum[j.current_zone] = j.zone_dwell_accum.get(j.current_zone, 0.0) + dwell

            # Entering new zone
            if current_zone is not None:
                j.zone_entry_time[current_zone] = timestamp_s
                if not j.history or j.history[-1] != current_zone:
                    j.history.append(current_zone)

                # Assign origin zone if not yet locked (only primary zones A and B qualify as origins)
                if j.origin_zone is None and current_zone in {"A", "B"}:
                    j.origin_zone = current_zone

            j.last_zone = j.current_zone
            j.current_zone = current_zone

        # Check for destination arrival and journey completion
        if j.origin_zone and current_zone and current_zone != j.origin_zone:
            # Check if this forms a complete primary journey (A -> B or B -> A)
            if j.origin_zone in {"A", "B"} and current_zone in {"A", "B"}:
                event_type = f"{j.origin_zone}_TO_{current_zone}"
                if event_type not in j.emitted_events:
                    j.emitted_events.add(event_type)
                    event = {
                        "event_type": event_type,
                        "source_zone": j.origin_zone,
                        "destination_zone": current_zone,
                        "journey_history": list(j.history),
                        "dwell_stats": dict(j.zone_dwell_accum),
                    }

        return event

    def get_journey(self, track_id: int) -> Optional[TrackJourney]:
        return self.journeys.get(track_id)
