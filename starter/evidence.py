import csv
import json
import os
import uuid
from datetime import datetime, timezone

import cv2


class EvidenceWriter:
    def __init__(self, output_dir, camera_id):
        self.output_dir = output_dir
        self.camera_id = camera_id
        self.evidence_dir = os.path.join(output_dir, "evidence")
        os.makedirs(self.evidence_dir, exist_ok=True)
        self.jsonl_path = os.path.join(output_dir, "events.jsonl")
        self.csv_path = os.path.join(output_dir, "counts.csv")
        self.counts = {}

    def write_event(self, frame, track_id, object_type, confidence, event, frame_time_s):
        event_type = event["event_type"]
        self.counts[event_type] = self.counts.get(event_type, 0) + 1
        obs_id = str(uuid.uuid4())
        image_name = f"{obs_id}.jpg"
        image_path = os.path.join(self.evidence_dir, image_name)
        cv2.imwrite(image_path, frame)

        record = {
            "observation_id": obs_id,
            "camera_id": self.camera_id,
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "video_time_seconds": round(frame_time_s, 3),
            "object_type": object_type,
            "event_type": event_type,
            "track_id": track_id,
            "confidence": float(confidence),
            "attributes": {
                "source_zone": event.get("source_zone"),
                "destination_zone": event.get("destination_zone"),
            },
            "evidence": {
                "image_path": os.path.relpath(image_path, self.output_dir),
                "clip_path": ""
            },
            "model": {
                "name": "baseline-motion-detector",
                "version": "1.0"
            }
        }

        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        self._write_counts()
        return record

    def _write_counts(self):
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["event_type", "count"])
            for event_type, count in sorted(self.counts.items()):
                writer.writerow([event_type, count])
