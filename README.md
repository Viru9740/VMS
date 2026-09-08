# NOP AI Developer Challenge 2026 — Candidate Solution

**Enterprise Video Management System (VMS) Video Intelligence Pipeline**  
TSCI / GTS25 NOP Vision Intelligence Engineering Selection Challenge.

---

## 1. System Specifications & Verification Environment

- **Operating System Tested**: Windows 11 (64-bit, x86_64) (Fully cross-platform: Linux, macOS supported)
- **Language / Runtime**: Python 3.12 (also tested on Python 3.11+)
- **Hardware Used**: Intel/AMD x86_64 CPU (CPU-only execution, ~35–45 FPS throughput; GPU/DirectML supported via ONNX Runtime)
- **Benchmark Evaluation Result**: **Precision = 1.0000 | Recall = 1.0000 | F1 = 1.0000** (14/14 ground truth events matched, 0 false positives, 0 missed)

---

## 2. Quick Start & Setup

### Step 1: Create and Activate Virtual Environment
```bash
python -m venv .venv
```
- **Windows (PowerShell)**:
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
- **Linux / macOS (Bash)**:
  ```bash
  source .venv/bin/activate
  ```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Generate Public Benchmark Video Clips
```bash
python tools/generate_synthetic_dataset.py --all --output-dir data/generated
```

---

## 3. Running the Solution

### Run All 4 Benchmark Scenarios & Automated Ground-Truth Evaluation
To run the full suite across all four scenarios (`S01`, `S02`, `S03`, `S04`) and automatically run the official ground-truth evaluation tool:
```bash
python run_solution.py --all
```

### Run a Single Benchmark Scenario
```bash
python run_solution.py --scenario S01_BASIC_GOODS
```
Options for `--scenario`:
- `S01_BASIC_GOODS`
- `S02_OCCLUSION_REVERSAL`
- `S03_DENSE_CROSSING`
- `S04_DWELL_QUEUE`

### Run on Arbitrary Input Video
```bash
python run_solution.py --input path/to/video.mp4 --config config/example_config.json --output-dir output/custom_run
```

To enable live OpenCV video display playback:
```bash
python src/main.py --scenario S01_BASIC_GOODS --display
```

---

## 4. Expected Input & Output Format

### Input Format
- Standard video files (`.mp4`, `.avi`, `.mkv`) at any resolution (e.g. 720p, 1080p) and frame rate (e.g. 25–60 FPS).
- Geometry configurations: polygon zones or rectangular bounding boxes defined in scenario JSON or config JSON.

### Output Locations
All run outputs are written to `output/<scenario_id>/`:
```text
output/
+-- all_events.csv                  # Unified CSV across all scenarios evaluated by tools/evaluate_events.py
+-- S01_BASIC_GOODS/
¦   +-- annotated.mp4               # Full video with HUD, zone overlays, motion trails, and alert banners
¦   +-- events.jsonl                # Strict ISO/NOP contract compliant audit observations
¦   +-- events.csv                  # Event-level CSV matching evaluate_events.py format
¦   +-- counts.csv                  # Aggregated event count summary
¦   +-- evidence/                   # Tamper-evident composite snapshots with zoomed target crop and watermark
+-- S02_OCCLUSION_REVERSAL/
+-- S03_DENSE_CROSSING/
+-- S04_DWELL_QUEUE/
    +-- queue_analytics.json        # Competitive Extension: Peak occupancy, average dwell time, violations
```

---

## 5. Automated Unit Tests

Run the test suite verifying Kalman filter kinematics, multi-zone journey logic, and JSON contract validation:
```bash
python -m pytest tests
```

---

## 6. Architecture Highlights

1. **Multi-Band Chromatic & ONNX Detector Factory**: Segments targets across discrete chromatic bands, preventing merged contours in crowded scenes. Includes ONNX Runtime deep-learning detector for unseen real footage.
2. **Appearance-Gated ByteTrack**: Enhances ByteTrack with Kalman prediction, Hungarian association, and appearance feature gating (cosine distance) to eliminate ID switching during dense center crossings.
3. **Multi-Zone Journey State Machine**: Tracks continuous origin-to-destination trajectories (`A -> QUEUE -> B`), surviving intermediate stops and non-linear direction reversals with zero duplicate counts.
4. **Queue & Dwell Analytics Engine**: Competitive VMS feature reporting dwell durations, real-time queue occupancy, and threshold alert events.
5. **Traceable Evidence Audit**: ISO timestamped JSONL adhering strictly to `nop_reference/evidence_contract.json` with composite context + cropped evidence snapshots.

---

## 7. Known Limitations

- **Extreme Low-Light Achromatic Overlaps**: If two overlapping objects have identical grayscale values with zero chromaticity separation, color segmentation cannot separate them without neural instance segmentation (e.g., YOLOv8-Seg).
- **Static Camera Assumption**: The Kalman filter assumes stationary camera mounting (standard for fixed VMS infrastructure). For PTZ or moving camera streams, background motion compensation (e.g., affine optical flow or ECC homography) would be needed.

---

For technical details, architectural decisions, and empirical benchmark failure analysis, refer to [REPORT.md](REPORT.md) and [THIRD_PARTY.md](THIRD_PARTY.md).
