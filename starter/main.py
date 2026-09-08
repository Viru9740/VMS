import argparse
import json
import os

import cv2

from detector import MotionDetector
from evidence import EvidenceWriter
from tracker import CentroidTracker
from zones import ZoneFlow, draw_zones


def parse_args():
    parser = argparse.ArgumentParser(description="NOP AI Developer Challenge starter")
    parser.add_argument("--input", required=True, help="Path to input video")
    parser.add_argument("--output-dir", default="output", help="Directory for challenge outputs")
    parser.add_argument("--config", default="config/example_config.json", help="JSON configuration")
    parser.add_argument("--display", action="store_true", help="Show live preview")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit(f"Unable to open video: {args.input}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_video = os.path.join(args.output_dir, "annotated.mp4")
    writer = cv2.VideoWriter(
        output_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    detector = MotionDetector(min_area=config.get("min_area", 900))
    tracker = CentroidTracker(
        max_match_distance=config.get("max_match_distance", 80),
        max_missed_frames=config.get("max_missed_frames", 12),
    )
    flow = ZoneFlow(config["zones"])
    evidence = EvidenceWriter(args.output_dir, config.get("camera_id", "challenge_cam_01"))

    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_index += 1
        detections = detector.detect(frame)
        tracks = tracker.update(detections)

        draw_zones(frame, config["zones"])

        for track_id, track in tracks.items():
            x, y, w, h = track["bbox"]
            cx, cy = track["centroid"]
            zone, event = flow.update(track_id, track["centroid"])

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"ID {track_id} {track['object_type']}"
            if zone:
                label += f" [{zone}]"
            cv2.putText(frame, label, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

            if event:
                evidence.write_event(
                    frame=frame,
                    track_id=track_id,
                    object_type=track["object_type"],
                    confidence=track["confidence"],
                    event=event,
                    frame_time_s=frame_index / fps,
                )

        writer.write(frame)
        if args.display:
            cv2.imshow("NOP Challenge Starter", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Done. Outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
