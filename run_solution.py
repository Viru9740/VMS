import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

from src.pipeline import VideoIntelligencePipeline


def run_scenario(scenario_path: Path, output_base_dir: Path, detector_override: str = None) -> dict:
    scenario = json.loads(scenario_path.read_text())
    scenario_id = scenario["scenario_id"]
    video_path = Path("data/generated") / f"{scenario_id}.mp4"
    if not video_path.exists():
        raise FileNotFoundError(f"Video {video_path} not found. Please run tools/generate_synthetic_dataset.py --all first.")

    if detector_override:
        scenario["detector_type"] = detector_override

    out_dir = output_base_dir / scenario_id
    pipeline = VideoIntelligencePipeline(config=scenario, output_dir=str(out_dir))
    stats = pipeline.process_video(video_path=str(video_path))
    return stats


def main():
    parser = argparse.ArgumentParser(description="Master Execution Harness for NOP AI Challenge 2026")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Run all 4 public benchmark scenarios and evaluate")
    group.add_argument("--scenario", help="Run a specific scenario (e.g. S01_BASIC_GOODS)")
    group.add_argument("--input", help="Process custom input video file or RTSP stream URL")
    group.add_argument("--camera", type=int, help="Live webcam index (e.g. 0 or 1)")

    parser.add_argument("--detector", choices=["benchmark", "yolo", "onnx"], default=None, help="Detector engine (benchmark, yolo, onnx)")
    parser.add_argument("--config", help="Configuration file for custom input")
    parser.add_argument("--output-dir", default="output", help="Root output directory")
    parser.add_argument("--display", action="store_true", help="Show live visual window during playback")
    args = parser.parse_args()

    out_base = Path(args.output_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    if args.all:
        scenarios = sorted(Path("data/scenarios").glob("*.json"))
        print(f"\n==================================================================")
        print(f" Executing Full NOP Public Benchmark Suite ({len(scenarios)} Scenarios)")
        print(f"==================================================================\n")

        all_events_rows = []
        header = None

        for sc_path in scenarios:
            print(f"--> Processing {sc_path.stem.upper()}...")
            stats = run_scenario(sc_path, out_base, detector_override=args.detector)
            print(f"    Completed {stats['frames_processed']} frames @ {stats['average_fps']} FPS | Events: {stats['events_detected']}")

            csv_path = Path(stats["events_csv"])
            if csv_path.exists():
                with open(csv_path, newline="", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    h = next(reader, None)
                    if header is None:
                        header = h
                    for row in reader:
                        all_events_rows.append(row)

        unified_csv_path = out_base / "all_events.csv"
        with open(unified_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if header:
                writer.writerow(header)
            writer.writerows(all_events_rows)

        print(f"\nSaved combined benchmark events to: {unified_csv_path}")
        print("\n==================================================================")
        print(" Running Official Evaluator (tools/evaluate_events.py)...")
        print("==================================================================\n")
        cmd = [sys.executable, "tools/evaluate_events.py", "--candidate", str(unified_csv_path)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(res.stdout)
        if res.stderr:
            print("Evaluator Stderr:", res.stderr)

    elif args.scenario:
        sc_file = None
        for f in Path("data/scenarios").glob("*.json"):
            sc = json.loads(f.read_text())
            if sc.get("scenario_id") == args.scenario:
                sc_file = f
                break
        if not sc_file:
            sys.exit(f"Error: scenario {args.scenario} not found in data/scenarios.")
        stats = run_scenario(sc_file, out_base, detector_override=args.detector)
        print(f"\nCompleted {args.scenario}:")
        print(json.dumps(stats, indent=2))

    elif args.camera is not None:
        print(f"\n[Live VMS Camera] Opening webcam #{args.camera}...")
        config = {
            "scenario_id": f"LIVE_CAMERA_{args.camera}",
            "camera_id": f"webcam_{args.camera}",
            "detector_type": args.detector or "yolo",
        }
        if args.config:
            with open(args.config, "r", encoding="utf-8") as f:
                config.update(json.load(f))

        out_dir = out_base / f"live_camera_{args.camera}"
        pipeline = VideoIntelligencePipeline(config=config, output_dir=str(out_dir))
        print("Press 'q' in the display window or Ctrl+C in terminal to stop recording.\n")
        stats = pipeline.process_video(video_path=args.camera, display=True)
        print("\nLive Camera Session Complete:")
        print(json.dumps(stats, indent=2))

    elif args.input:
        config = {
            "scenario_id": "CUSTOM_INPUT",
            "camera_id": "custom_cam_01",
            "detector_type": args.detector or "yolo",
        }
        if args.config:
            with open(args.config, "r", encoding="utf-8") as f:
                config.update(json.load(f))

        out_dir = out_base / "custom_input"
        pipeline = VideoIntelligencePipeline(config=config, output_dir=str(out_dir))
        stats = pipeline.process_video(video_path=args.input, display=args.display)
        print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
