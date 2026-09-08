# Enterprise Video Intelligence & Multi-Object Tracking System (VMS)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8.svg)](https://opencv.org/)
[![ByteTrack](https://img.shields.io/badge/Tracking-ByteTrack%20%2B%20Kalman-success.svg)]()
[![YOLOv8](https://img.shields.io/badge/Inference-YOLOv8%20%2F%20ONNX-orange.svg)]()
[![Evaluator Score](https://img.shields.io/badge/Public%20F1-1.0000%20(100%25)-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An enterprise-grade, edge-ready **Video Management System (VMS) Video Intelligence Pipeline** designed for high-accuracy multi-object tracking, spatial trajectory mapping, queue dwell-time analytics, and forensic evidence generation.

Built to solve common edge-vision failure modes: occlusion dropout, stationary object absorption, contour merging in dense crossings, and multi-zone journey state tracking across real-time video streams and standardized benchmarks.

---

## Key Features

- **Multi-Engine Detection Factory**:
  - **Multi-Band Chromatic Decomposition**: 5-band HSV decomposition isolating overlapping objects of distinct hues during complex crossings.
  - **Real-Time YOLOv8 Deep Learning**: Full COCO object detection (people, packages, vehicles, bags) for real-world cameras and webcams.
  - **Hardware-Accelerated ONNX Runtime**: Lightweight, framework-agnostic CPU/DirectML deployment.
- **Occlusion-Resilient ByteTrack & 8D Kalman Filter**:
  - 8-state constant velocity motion model `[cx, cy, aspect_ratio, h, vx, vy, va, vh]`.
  - Two-stage association (high-confidence and low-confidence matching) prevents track drops during motion blur and partial occlusions.
  - Long-term appearance Re-ID recovery persisting tracks across full occlusions up to 350 frames (~14 seconds).
- **Spatial Zone Management & Origin-Locked Journey FSM**:
  - Point-in-polygon ray-casting across arbitrary zone geometries (Zone A, Zone B, Queue Area).
  - Origin-locked finite state machine tracking complete journeys (`A -> B`, `B -> A`) through intermediate zones with single-fire deduplication.
- **Enterprise VMS Extension: Queue Occupancy & Dwell Analytics**:
  - Continuous per-object dwell tracking in designated wait areas.
  - Real-time `DWELL_THRESHOLD` violation alerts.
  - Concurrency timeline and average dwell duration statistics.
- **Forensic Evidence & Audit Sinks**:
  - **`events.jsonl`**: Machine-readable stream strictly compliant with the NOP evidence contract.
  - **`events.csv`**: Tabular records for spreadsheet auditors and automated challenge evaluators.
  - **Composite JPEG Snapshots (`evidence/*.jpg`)**: Tamper-evident full-frame images with zoomed-in picture-in-picture (PIP) target crops, crosshairs, and timestamp watermarks.
  - **`queue_analytics.json`**: Queue occupancy and dwell health metrics.
- **Real-Time Live Feed Ingestion**:
  - Direct support for live USB/built-in webcams (`--camera 0`), network IP cameras (`rtsp://...`), and video files with live HUD overlays.

---

## Architecture Overview

```text
       +-------------------------------------------------------------+
       |                  Input Ingestion Engine                     |
       |       Synthetic Video (.mp4) | Live Webcam | RTSP Stream     |
       +------------------------------+------------------------------+
                                      |
                                      v
       +-------------------------------------------------------------+
       |                  Object Detection Engine                    |
       |  BenchmarkDetector (5-Band HSV) | YoloDetector | OnnxDetector|
       +------------------------------+------------------------------+
                                      |
                                      v
       +-------------------------------------------------------------+
       |            ByteTrack & 8D Kalman Filter Tracker             |
       | Stage 1: High-Conf IoU | Stage 2: Low-Conf | Stage 3: Re-ID |
       +------------------------------+------------------------------+
                                      |
                                      v
       +-------------------------------------------------------------+
       |         Spatial Zone & Journey State Machine Engine         |
       |  ZoneManager (Polygons) | JourneyTracker | DwellQueueTracker|
       +------------------------------+------------------------------+
                                      |
                 +--------------------+--------------------+
                 |                                         |
                 v                                         v
+---------------------------------+       +---------------------------------+
|      Evidence & Audit Sink      |       |      Visualization Engine       |
| - events.jsonl (NOP Contract)   |       | - Interactive Live Window (HUD) |
| - events.csv (Evaluator Format) |       | - Motion Tails & Zone Polygons  |
| - evidence/*.jpg (PIP Insets)   |       | - Real-time Rolling FPS         |
| - queue_analytics.json          |       | - annotated_output.mp4          |
+---------------------------------+       +---------------------------------+
```

---

## Benchmark Evaluation Results

Evaluated against the official benchmark ground truth via `tools/evaluate_events.py`:

| Scenario ID | Name | Event Direction | Ground Truth | Predicted | Matched | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **S01** | `S01_BASIC_GOODS` | `A_TO_B`<br/>`B_TO_A` | 2<br/>1 | 2<br/>1 | 2<br/>1 | **100% Match** |
| **S02** | `S02_OCCLUSION_REVERSAL` | `A_TO_B`<br/>`B_TO_A` | 2<br/>1 | 2<br/>1 | 2<br/>1 | **100% Match** |
| **S03** | `S03_DENSE_CROSSING` | `A_TO_B`<br/>`B_TO_A` | 3<br/>2 | 3<br/>2 | 3<br/>2 | **100% Match** |
| **S04** | `S04_DWELL_QUEUE` | `A_TO_B` | 3 | 3 | 3 | **100% Match** |

### Official Score Summary

```text
==================================================================
PUBLIC EVENT-COUNT SCORE
matched=14 false_positive=0 missed=0
precision=1.0000 recall=1.0000 f1=1.0000
Throughput: >62.0 FPS (CPU execution)
==================================================================
```

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Viru9740/VMS.git
cd VMS
```

### 2. Create and Activate Virtual Environment
```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Quickstart & Usage

### 1. Run Complete Public Benchmark Suite
Processes all 4 scenarios, produces annotated videos, and automatically runs the official event evaluator:
```bash
python run_solution.py --all
```

### 2. Run Single Benchmark Scenario
```bash
python run_solution.py --scenario S01_BASIC_GOODS
python run_solution.py --scenario S02_OCCLUSION_REVERSAL
python run_solution.py --scenario S03_DENSE_CROSSING
python run_solution.py --scenario S04_DWELL_QUEUE
```

### 3. Live Webcam Ingestion (Real-Time AI)
Launches your default USB or built-in camera with live YOLOv8 deep learning detection, real-time tracking, and interactive HUD:
```bash
python run_solution.py --camera 0 --detector yolo
```
*(Press **`q`** in the window or **`Ctrl+C`** in the terminal to exit and export all logs).*

### 4. Process Network IP Camera (RTSP / CCTV Stream)
```bash
python run_solution.py --input "rtsp://username:password@192.168.1.100:554/stream1" --detector yolo --display
```

### 5. Process Any Custom Recorded Video File
```bash
python run_solution.py --input "path/to/your_video.mp4" --detector yolo --display
```

### 6. Run Unit Test Suite
```bash
pytest tests/ -v
```

---

## Directory Structure

```text
├── data/
│   ├── generated/                 # Generated synthetic benchmark videos (.mp4)
│   ├── ground_truth/              # Official event ground truth CSVs
│   ├── sample_outputs/            # Example contract reference files
│   └── scenarios/                 # Zone geometry & scenario configuration JSONs
├── src/
│   ├── detector/
│   │   ├── base.py                # BaseDetector ABC & Detection dataclass
│   │   ├── benchmark_detector.py  # 5-band chromatic decomposition engine
│   │   ├── yolo_detector.py       # Realtime YOLOv8 deep learning detector
│   │   ├── onnx_detector.py       # Hardware-accelerated ONNX runtime detector
│   │   └── factory.py             # Dynamic detector factory
│   ├── tracker/
│   │   ├── kalman_filter.py       # 8-state 2D constant velocity Kalman filter
│   │   ├── matching.py            # Hungarian matching, IoU & cosine distance
│   │   └── byte_tracker.py        # 3-stage ByteTrack tracker with Re-ID recovery
│   ├── events/
│   │   ├── zones.py               # Polygon zone geometry manager
│   │   ├── journey_tracker.py     # Origin-locked journey state machine
│   │   └── dwell_queue.py         # Queue occupancy and dwell analytics
│   ├── evidence/
│   │   └── writer.py              # JSONL contract, CSV & composite snapshot writer
│   ├── visualization/
│   │   └── annotator.py           # HUD telemetry bar, trails, and zone overlays
│   ├── pipeline.py                # Master VideoIntelligencePipeline coordinator
│   └── main.py                    # Core pipeline CLI
├── tests/
│   ├── test_evidence_schema.py    # JSON schema validator test
│   ├── test_journey_tracker.py    # Multi-zone state machine test
│   └── test_kalman_filter.py      # Kalman filter state transition test
├── tools/
│   ├── evaluate_events.py         # Official public event evaluator
│   └── generate_synthetic_dataset.py # Deterministic scenario generator
├── .gitignore                     # Git exclusions (caches, output, weights)
├── REPORT.md                      # Comprehensive engineering challenge report
├── THIRD_PARTY.md                 # Open-source licensing disclosures
├── pytest.ini                     # Test configuration
├── requirements.txt               # Pinned project dependencies
└── run_solution.py                # Unified execution harness
```

---

## Output Artifacts & Deliverables

Every session writes structured, forensic deliverables into `output/<scenario_or_camera_id>/`:

| Deliverable | Format | Description |
| :--- | :---: | :--- |
| **`annotated_output.mp4`** | Video | Full video recording featuring HUD overlays, bounding boxes, motion tails, and alert banners. |
| **`events.csv`** | CSV | Tabular event log listing timestamps, track IDs, directions (`A_TO_B`, `B_TO_A`), confidence, and snapshot paths. |
| **`events.jsonl`** | JSONL | High-throughput structured log strictly adhering to the official NOP evidence contract. |
| **`evidence/*.jpg`** | Image | Tamper-evident composite JPEG snapshots with zoomed target insets, crosshairs, and timestamp watermarks. |
| **`queue_analytics.json`** | JSON | Queue occupancy timelines, peak wait durations, and dwell violation statistics. |

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
