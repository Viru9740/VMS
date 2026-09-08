import argparse
import json
from pathlib import Path
import sys

from .pipeline import VideoIntelligencePipeline


def parse_args():
    parser = argparse.ArgumentParser(description="NOP AI Developer Challenge 2026 - Production VMS Pipeline")
    parser.add_argument("--input", help="Path to input video file or RTSP stream URL")
    parser.add_argument("--camera", type=int, help="Live webcam index (e.g. 0 or 1)")
    parser.add_argument("--config", help="Path to JSON configuration file")
    parser.add_argument("--scenario", help="Public scenario name (e.g. S01_BASIC_GOODS)")
    parser.add_argument("--detector", choices=["benchmark", "yolo", "onnx"], default="yolo", help="Detector model")
    parser.add_argument("--output-dir", default="output", help="Directory to save generated outputs")
    parser.add_argument("--display", action="store_true", help="Display live video playback window")
    return parser.parse_args()


def load_scenario_config(scenario_name: str) -> dict:
    scenarios_dir = Path("data/scenarios")
    for f in scenarios_dir.glob("*.json"):
        data = json.loads(f.read_text())
        if data.get("scenario_id") == scenario_name:
            return data
    raise FileNotFoundError(f"Scenario {scenario_name} not found in {scenarios_dir}")


def main():
    args = parse_args()

    if args.scenario:
        scenario_data = load_scenario_config(args.scenario)
        config = scenario_data
        video_path = Path("data/generated") / f"{args.scenario}.mp4"
        if not video_path.exists():
            raise FileNotFoundError(f"Benchmark video not found at {video_path}. Generate with tools/generate_synthetic_dataset.py --all")
    elif args.camera is not None:
        config = {
            "scenario_id": f"LIVE_CAMERA_{args.camera}",
            "camera_id": f"webcam_{args.camera}",
            "detector_type": args.detector,
        }
        if args.config:
            with open(args.config, "r", encoding="utf-8") as f:
                config.update(json.load(f))
        video_path = args.camera
        args.display = True
    elif args.input:
        config = {
            "scenario_id": "CUSTOM_INPUT",
            "camera_id": "custom_cam_01",
            "detector_type": args.detector,
        }
        if args.config:
            with open(args.config, "r", encoding="utf-8") as f:
                config.update(json.load(f))
        video_path = args.input
    else:
        config = {
            "scenario_id": "DEFAULT_CLIP",
            "camera_id": "challenge_cam_01",
            "zones": {
                "A": [80, 120, 430, 620],
                "B": [850, 120, 1200, 620],
            }
        }
        video_path = "data/generated/S01_BASIC_GOODS.mp4"

    print(f"\n========================================================")
    print(f" NOP Vision Intelligence VMS Pipeline (Candidate 2026) ")
    print(f" Scenario: {config.get('scenario_id', 'CUSTOM')} | Source: {video_path}")
    print(f" Detector: {config.get('detector_type', 'default')}")
    print(f"========================================================\n")

    pipeline = VideoIntelligencePipeline(config=config, output_dir=args.output_dir)
    results = pipeline.process_video(video_path=video_path, display=args.display)

    print(f"\nProcessing Complete!")
    print(f"- Frames Processed: {results['frames_processed']} ({results['duration_seconds']:.2f}s video)")
    print(f"- Pipeline Throughput: {results['average_fps']} FPS")
    print(f"- Events Detected: {results['events_detected']}")
    print(f"- Output Directory: {results['output_dir']}")
    print(f"- Annotated Video: {results['annotated_video']}")
    print(f"- Events CSV: {results['events_csv']}")
    print(f"- Events JSONL: {results['events_jsonl']}")


if __name__ == "__main__":
    main()
