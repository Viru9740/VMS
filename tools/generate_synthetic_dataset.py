import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def lerp(a, b, t):
    return a + (b - a) * t


def position_from_object(obj, frame_idx):
    if "waypoints" in obj:
        points = sorted(obj["waypoints"], key=lambda p: p[2])
        if frame_idx < points[0][2] or frame_idx > points[-1][2]:
            return None
        for left, right in zip(points[:-1], points[1:]):
            x1, y1, f1 = left
            x2, y2, f2 = right
            if f1 <= frame_idx <= f2:
                t = 0.0 if f2 == f1 else (frame_idx - f1) / float(f2 - f1)
                return int(lerp(x1, x2, t)), int(lerp(y1, y2, t))
        return None

    sf = obj["start_frame"]
    ef = obj["end_frame"]
    if frame_idx < sf or frame_idx > ef:
        return None
    t = (frame_idx - sf) / float(max(1, ef - sf))
    x = int(lerp(obj["start"][0], obj["end"][0], t))
    y = int(lerp(obj["start"][1], obj["end"][1], t))
    return x, y


def draw_zone(frame, rect, label):
    x1, y1, x2, y2 = rect
    cv2.rectangle(frame, (x1, y1), (x2, y2), (110, 110, 110), 2)
    cv2.putText(frame, label, (x1 + 8, y1 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2)


def generate_one(scenario_path: Path, output_dir: Path):
    scenario = json.loads(scenario_path.read_text())
    output_dir.mkdir(parents=True, exist_ok=True)

    width = int(scenario["width"])
    height = int(scenario["height"])
    fps = int(scenario["fps"])
    total_frames = int(round(float(scenario["duration_seconds"]) * fps))

    video_path = output_dir / f'{scenario["scenario_id"]}.mp4'
    truth_path = output_dir / f'{scenario["scenario_id"]}_frame_truth.csv'

    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {video_path}")

    truth_lines = ["frame,time_seconds,object_id,class,cx,cy,x1,y1,x2,y2,visible"]
    palette = [(70, 170, 255), (120, 220, 120), (230, 150, 80), (180, 100, 230), (80, 210, 210)]

    for frame_idx in range(total_frames):
        frame = np.full((height, width, 3), 28, dtype=np.uint8)
        cv2.putText(frame, scenario["scenario_id"], (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 240), 2)
        cv2.putText(frame, f"t={frame_idx / fps:05.2f}s", (24, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)

        for name, rect in scenario.get("zones", {}).items():
            draw_zone(frame, rect, name)

        objects_to_draw = []
        for idx, obj in enumerate(scenario.get("objects", [])):
            pos = position_from_object(obj, frame_idx)
            if pos is None:
                continue
            cx, cy = pos
            w, h = obj.get("size", [70, 54])
            x1, y1 = int(cx - w / 2), int(cy - h / 2)
            x2, y2 = int(cx + w / 2), int(cy + h / 2)
            objects_to_draw.append((idx, obj, cx, cy, x1, y1, x2, y2))

        # Draw moving objects first. Occluders are drawn after them to create real visual occlusion.
        for idx, obj, cx, cy, x1, y1, x2, y2 in objects_to_draw:
            color = palette[idx % len(palette)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (245, 245, 245), 2)
            cv2.putText(frame, f'{obj["class"]} {obj["id"]}', (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 2)

        for rect in scenario.get("occluders", []):
            x1, y1, x2, y2 = rect
            cv2.rectangle(frame, (x1, y1), (x2, y2), (55, 55, 55), -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (130, 130, 130), 2)
            cv2.putText(frame, "OCCLUDER", (x1 + 8, y1 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 2)

        for _, obj, cx, cy, x1, y1, x2, y2 in objects_to_draw:
            visible = 1
            for ox1, oy1, ox2, oy2 in scenario.get("occluders", []):
                if ox1 <= cx <= ox2 and oy1 <= cy <= oy2:
                    visible = 0
                    break
            truth_lines.append(
                f'{frame_idx},{frame_idx / fps:.3f},{obj["id"]},{obj["class"]},{cx},{cy},{x1},{y1},{x2},{y2},{visible}'
            )

        writer.write(frame)

    writer.release()
    truth_path.write_text("\n".join(truth_lines) + "\n")
    print(f"generated {video_path}")
    print(f"generated {truth_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate deterministic public NOP interview benchmark clips.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scenario", type=Path)
    group.add_argument("--all", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("data/generated"))
    args = parser.parse_args()

    if args.all:
        for scenario_path in sorted(Path("data/scenarios").glob("*.json")):
            generate_one(scenario_path, args.output_dir)
    else:
        generate_one(args.scenario, args.output_dir)


if __name__ == "__main__":
    main()
