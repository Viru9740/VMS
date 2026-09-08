from pathlib import Path
from typing import Any, Dict, Optional
import time
import cv2
import numpy as np

from .detector.factory import create_detector
from .tracker.byte_tracker import ByteTracker
from .events.zones import ZoneManager
from .events.journey_tracker import JourneyTracker
from .events.dwell_queue import DwellQueueAnalyzer
from .evidence.writer import EvidenceWriter
from .visualization.annotator import VisualAnnotator


class VideoIntelligencePipeline:
    """Master end-to-end VMS Video Intelligence Pipeline."""

    def __init__(self, config: Dict[str, Any], output_dir: str):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.scenario_id = config.get("scenario_id", "CUSTOM_SCENARIO")
        self.camera_id = config.get("camera_id", "challenge_cam_01")

        # 1. Initialize Detector
        self.detector = create_detector(config)

        # 2. Initialize Tracker
        self.tracker = ByteTracker(
            high_threshold=config.get("tracker_high_threshold", 0.40),
            low_threshold=config.get("tracker_low_threshold", 0.15),
            match_threshold=config.get("tracker_match_threshold", 0.70),
            reid_threshold=config.get("tracker_reid_threshold", 0.55),
            max_lost_frames=config.get("max_lost_frames", 350),
        )

        # 3. Initialize Zone Manager
        zones_def = config.get("zones", {})
        self.zone_manager = ZoneManager(zones_def)

        # 4. Initialize Journey & Event State Machine
        self.journey_tracker = JourneyTracker()

        # 5. Initialize Competitive Extension: Dwell & Queue Analyzer
        self.dwell_analyzer = None
        if "QUEUE" in zones_def:
            self.dwell_analyzer = DwellQueueAnalyzer(
                queue_zone_name="QUEUE",
                dwell_threshold_s=config.get("dwell_threshold_seconds", 5.0),
            )

        # 6. Initialize Evidence & Visual Output
        self.evidence_writer = EvidenceWriter(
            output_dir=self.output_dir,
            camera_id=self.camera_id,
            scenario_id=self.scenario_id,
            model_name=config.get("model_name", "ByteTrack-NOP-VMS"),
            model_version=config.get("model_version", "2.0.0"),
        )
        self.annotator = VisualAnnotator(scenario_id=self.scenario_id)

    def process_video(self, video_path: str, display: bool = False) -> Dict[str, Any]:
        self.tracker.reset()
        self.journey_tracker = JourneyTracker()
        if "QUEUE" in self.config.get("zones", {}):
            self.dwell_analyzer = DwellQueueAnalyzer(
                queue_zone_name="QUEUE",
                dwell_threshold_s=self.config.get("dwell_threshold_seconds", 5.0),
            )

        # Support live webcam index (e.g. 0, "0") or file path / RTSP URL
        if isinstance(video_path, int) or (isinstance(video_path, str) and video_path.strip().isdigit()):
            source = int(video_path)
        else:
            source = str(video_path)

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise FileNotFoundError(f"Failed to open input video stream: {video_path}")

        raw_fps = cap.get(cv2.CAP_PROP_FPS)
        fps = float(raw_fps) if raw_fps and not np.isnan(raw_fps) and raw_fps > 0 else 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        total_frames = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))

        # Auto-configure default zones if not provided
        if not self.config.get("zones"):
            default_zones = {
                "A": [0, 0, width // 3, height],
                "B": [2 * width // 3, 0, width, height],
            }
            self.config["zones"] = default_zones
            self.zone_manager = ZoneManager(default_zones)

        output_video_path = self.output_dir / "annotated.mp4"
        writer = cv2.VideoWriter(
            str(output_video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        frame_idx = 0
        t_start = time.perf_counter()
        t_last = t_start
        fps_realtime = fps

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_idx += 1
            time_s = frame_idx / fps

            # Compute rolling FPS
            now = time.perf_counter()
            dt = now - t_last
            if dt >= 0.2:
                fps_realtime = 1.0 / max(1e-5, dt)
                t_last = now

            # 1. Detection
            raw_frame_copy = frame.copy()
            detections = self.detector.detect(frame, frame_idx)

            # 2. Tracking
            tracks = self.tracker.update(detections, frame_idx, time_s)

            # 3. Zone Localization & Event Processing
            track_zones: Dict[int, Optional[str]] = {}
            active_zone_counts: Dict[str, int] = {}
            track_dwells: Dict[int, float] = {}

            for track in tracks:
                tid = track.track_id
                zone = self.zone_manager.get_zone_for_point(track.centroid)
                track_zones[tid] = zone
                if zone:
                    active_zone_counts[zone] = active_zone_counts.get(zone, 0) + 1

                # Update trajectory state machine
                journey_event = self.journey_tracker.update(tid, zone, time_s)
                if journey_event:
                    self.evidence_writer.write_event(
                        frame=raw_frame_copy,
                        track_id=tid,
                        object_type=track.class_name,
                        confidence=track.confidence,
                        event=journey_event,
                        frame_time_s=time_s,
                        bbox=track.to_tlwh(),
                    )
                    self.annotator.register_event(journey_event["event_type"], tid, time_s)

            # 4. Queue / Dwell Analytics
            queue_occupancy = None
            if self.dwell_analyzer:
                dwell_alerts = self.dwell_analyzer.update(track_zones, time_s)
                queue_occupancy = self.dwell_analyzer.current_occupancy
                for track in tracks:
                    track_dwells[track.track_id] = self.dwell_analyzer.get_track_dwell(track.track_id)
                for alert in dwell_alerts:
                    self.evidence_writer.write_event(
                        frame=raw_frame_copy,
                        track_id=alert["track_id"],
                        object_type="package",
                        confidence=0.95,
                        event=alert,
                        frame_time_s=time_s,
                        include_in_eval_csv=False,
                    )
                    self.annotator.register_event("DWELL_ALERT", alert["track_id"], time_s)

            # 5. Visual Annotations
            self.zone_manager.draw(frame, active_zone_counts)
            self.annotator.draw_tracks(frame, tracks, track_zones, track_dwells)
            self.annotator.draw_active_alerts(frame, time_s)
            self.annotator.draw_hud(
                frame=frame,
                frame_idx=frame_idx,
                fps_realtime=fps_realtime,
                time_s=time_s,
                active_track_count=len(tracks),
                event_counts=self.evidence_writer.counts,
                queue_occupancy=queue_occupancy,
            )

            writer.write(frame)

            if display:
                cv2.imshow(f"NOP VMS - {self.scenario_id}", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        total_time = time.perf_counter() - t_start
        cap.release()
        writer.release()
        if display:
            cv2.destroyAllWindows()

        # Save queue analytics if recorded
        if self.dwell_analyzer:
            self.evidence_writer.write_queue_analytics(self.dwell_analyzer.get_summary_metrics())

        avg_fps = frame_idx / max(1e-4, total_time)
        return {
            "scenario_id": self.scenario_id,
            "frames_processed": frame_idx,
            "total_frames": total_frames,
            "duration_seconds": frame_idx / fps,
            "processing_time_s": round(total_time, 2),
            "average_fps": round(avg_fps, 1),
            "events_detected": dict(self.evidence_writer.counts),
            "output_dir": str(self.output_dir),
            "annotated_video": str(output_video_path),
            "events_csv": str(self.evidence_writer.events_csv_path),
            "events_jsonl": str(self.evidence_writer.jsonl_path),
            "counts_csv": str(self.evidence_writer.counts_csv_path),
        }
