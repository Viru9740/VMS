from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
import csv
import json
import os
import uuid
import cv2
import numpy as np


class EvidenceWriter:
    """Enterprise-grade VMS evidence and audit log writer adhering to NOP evidence contract."""

    def __init__(
        self,
        output_dir: Union[str, Path] if 'Union' in locals() else str,
        camera_id: str = "challenge_cam_01",
        scenario_id: str = "S01_BASIC_GOODS",
        model_name: str = "ByteTrack-NOP-Intelligence",
        model_version: str = "2.0.0",
    ):
        self.output_dir = Path(output_dir)
        self.camera_id = camera_id
        self.scenario_id = scenario_id
        self.model_name = model_name
        self.model_version = model_version

        self.evidence_dir = self.output_dir / "evidence"
        self.evidence_dir.mkdir(parents=True, exist_ok=True)

        self.jsonl_path = self.output_dir / "events.jsonl"
        self.events_csv_path = self.output_dir / "events.csv"
        self.counts_csv_path = self.output_dir / "counts.csv"
        self.queue_json_path = self.output_dir / "queue_analytics.json"

        self.counts: Dict[str, int] = {}
        self.event_records: List[Dict] = []
        self._event_counter = 0

        # Initialize CSV headers
        self._init_csv()

    def _init_csv(self):
        with open(self.events_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "scenario_id", "event_id", "track_id", "event_type",
                "object_type", "observed_at_seconds", "confidence", "evidence_image"
            ])

    def create_composite_snapshot(
        self,
        frame: np.ndarray,
        bbox: List[int],
        track_id: int,
        event_type: str,
        obs_id: str,
        time_s: float,
    ) -> Path:
        """Render tamper-evident visual evidence with metadata overlay and zoomed crop inset."""
        snapshot = frame.copy()
        h, w, _ = snapshot.shape
        bx, by, bw, bh = bbox

        # Draw highlight box around the event subject
        cv2.rectangle(snapshot, (bx, by), (bx + bw, by + bh), (0, 255, 255), 3)

        # Create zoomed crop inset in the top-right corner
        crop_x1, crop_y1 = max(0, bx - 10), max(0, by - 10)
        crop_x2, crop_y2 = min(w, bx + bw + 10), min(h, by + bh + 10)
        crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

        if crop.size > 0:
            inset_h, inset_w = 140, 160
            inset_resized = cv2.resize(crop, (inset_w, inset_h), interpolation=cv2.INTER_LINEAR)
            ix1, iy1 = w - inset_w - 20, 20
            ix2, iy2 = ix1 + inset_w, iy1 + inset_h

            snapshot[iy1:iy2, ix1:ix2] = inset_resized
            cv2.rectangle(snapshot, (ix1, iy1), (ix2, iy2), (0, 255, 255), 2)
            cv2.putText(snapshot, "EVENT TARGET", (ix1 + 6, iy2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        # Draw bottom audit watermark banner
        banner_h = 36
        cv2.rectangle(snapshot, (0, h - banner_h), (w, h), (20, 20, 20), -1)
        meta_str = (
            f"OBS_ID: {obs_id[:8]}... | CAM: {self.camera_id} | SCENARIO: {self.scenario_id} | "
            f"EVENT: {event_type} | TRK: {track_id} | T: {time_s:05.2f}s | {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}"
        )
        cv2.putText(snapshot, meta_str, (16, h - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (230, 230, 230), 1)

        filename = f"evt_{self._event_counter:04d}_{obs_id[:8]}.jpg"
        filepath = self.evidence_dir / filename
        cv2.imwrite(str(filepath), snapshot, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return filepath

    def write_event(
        self,
        frame: np.ndarray,
        track_id: int,
        object_type: str,
        confidence: float,
        event: Dict,
        frame_time_s: float,
        bbox: Optional[List[int]] = None,
        include_in_eval_csv: bool = True,
    ) -> Dict:
        self._event_counter += 1
        event_type = event["event_type"]
        self.counts[event_type] = self.counts.get(event_type, 0) + 1
        obs_id = str(uuid.uuid4())
        event_id = f"evt-{self._event_counter:04d}"

        # Generate composite snapshot
        if bbox is None:
            bbox = [0, 0, frame.shape[1], frame.shape[0]]
        img_path = self.create_composite_snapshot(frame, bbox, track_id, event_type, obs_id, frame_time_s)
        rel_img_path = f"evidence/{img_path.name}"

        # 1. NOP Evidence Contract compliant JSONL record
        record = {
            "observation_id": obs_id,
            "camera_id": self.camera_id,
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "video_time_seconds": round(frame_time_s, 3),
            "object_type": object_type,
            "event_type": event_type,
            "track_id": track_id,
            "confidence": round(float(confidence), 3),
            "attributes": {
                "scenario_id": self.scenario_id,
                "event_id": event_id,
                "source_zone": event.get("source_zone"),
                "destination_zone": event.get("destination_zone"),
                "journey_history": event.get("journey_history", []),
                "dwell_stats": event.get("dwell_stats", {}),
            },
            "evidence": {
                "image_path": rel_img_path,
                "clip_path": "",
            },
            "model": {
                "name": self.model_name,
                "version": self.model_version,
            },
        }

        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        # 2. Append to events.csv (evaluated by tools/evaluate_events.py) if primary event
        if include_in_eval_csv:
            with open(self.events_csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.scenario_id,
                    event_id,
                    track_id,
                    event_type,
                    object_type,
                    f"{frame_time_s:.3f}",
                    f"{confidence:.2f}",
                    rel_img_path,
                ])

        # 3. Update counts.csv
        self._write_counts()
        self.event_records.append(record)
        return record

    def _write_counts(self):
        with open(self.counts_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["event_type", "count"])
            for event_type, count in sorted(self.counts.items()):
                writer.writerow([event_type, count])

    def write_queue_analytics(self, analytics: Dict):
        with open(self.queue_json_path, "w", encoding="utf-8") as f:
            json.dump(analytics, f, indent=2)
