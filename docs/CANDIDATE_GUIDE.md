# NOP Vision Intelligence Engineering Challenge 2026 - Detailed Candidate Guide

## 1. Goal

Build a useful, reliable video-intelligence capability on top of the public NOP challenge starter. The objective is not to produce the most code or the most attractive demo. The objective is to demonstrate that you can build a working AI/video system, measure it, produce traceable evidence, explain its limitations, and modify it under interview conditions.

TSCI is evaluating execution ability, computer-vision fundamentals, generalisation, product thinking, evidence quality, engineering quality and effective use of modern AI development tools.

## 2. Mandatory outcome

Your submission must:

1. Accept prerecorded video input.
2. Detect at least one useful target/object category.
3. Maintain track identity over time.
4. Use configurable zones, lines, dwell regions or equivalent event geometry.
5. Generate events from tracked behavior/movement rather than raw frame count.
6. Avoid obvious duplicate counting.
7. Produce annotated video or equivalent visual evidence.
8. Produce machine-readable JSON evidence.
9. Produce CSV summary output.
10. Save evidence snapshots for generated events.
11. Include setup/run instructions.
12. Document models, libraries, licenses, AI assistants used and known limitations.

## 3. Platform freedom

Linux, Windows and macOS are all allowed. CPU-only and GPU/accelerator systems are allowed. You may use Python, C++, Rust, JavaScript or another suitable language.

You may use OpenCV, ONNX Runtime, PyTorch, TensorFlow, TensorRT, OpenVINO, DirectML, Core ML or other suitable frameworks/runtimes.

AI-assisted development is explicitly allowed, including ChatGPT, Codex, Claude, Gemini, GitHub Copilot and similar tools. You remain responsible for understanding, validating and modifying your solution.

## 4. Repository map

- `README.md` - entry point and quick start.
- `CHALLENGE.md` - challenge scope and mandatory requirements.
- `RULES.md` - fairness, platform, security and AI-tool rules.
- `SCORING.md` - evaluation model.
- `SUBMISSION.md` - expected submission package.
- `starter/` - deliberately simple cross-platform baseline.
- `config/` - example configuration.
- `nop_reference/` - sanitized NOP-style evidence contract and event names.
- `data/scenarios/` - deterministic public benchmark definitions.
- `data/ground_truth/` - public event-level expected results.
- `data/sample_outputs/` - output examples.
- `tools/generate_synthetic_dataset.py` - creates public benchmark MP4s and frame truth.
- `tools/evaluate_events.py` - basic public event-count evaluator.

## 5. Public benchmark data

Generate all public benchmark clips:

```bash
python tools/generate_synthetic_dataset.py --all --output-dir data/generated
```

The four supplied public scenarios are:

### S01_BASIC_GOODS
Clean A-to-B and B-to-A goods movement. Use it to validate basic detection, tracking, direction and event creation.

### S02_OCCLUSION_REVERSAL
Temporary occlusion, reversal, stop and resume. This tests whether track identity and event-state logic survive imperfect visibility and non-linear motion.

### S03_DENSE_CROSSING
Several objects cross close to one another. This is designed to expose ID switches, fragmented tracks and duplicate counts.

### S04_DWELL_QUEUE
Objects pause inside a central queue/dwell region before continuing. This creates an opportunity to implement dwell time, queue occupancy and more advanced temporal analytics.

Public ground truth is intentionally available so candidates can measure their systems. Finalists may receive unseen footage, so hard-coding public answers will not generalize.

## 6. Evidence contract

Every important event should be traceable. Recommended fields include:

- source/scenario/camera identifier;
- observation time;
- object type;
- event type;
- track ID;
- confidence;
- source/destination zone or other event attributes;
- evidence image and/or clip path;
- model name/version.

Example:

```json
{
  "scenario_id": "S03_DENSE_CROSSING",
  "camera_id": "candidate_cam_01",
  "observed_at_seconds": 15.42,
  "object_type": "package",
  "event_type": "A_TO_B",
  "track_id": 42,
  "confidence": 0.93,
  "attributes": {
    "source_zone": "A",
    "destination_zone": "B"
  },
  "evidence": {
    "image_path": "evidence/evt_0042.jpg"
  },
  "model": {
    "name": "your-model",
    "version": "1.0"
  }
}
```

## 7. Competitive extensions

After meeting the mandatory requirements, add at least one useful improvement. Examples:

- goods/package route intelligence;
- person entry/exit, occupancy or dwell;
- queue analytics;
- vehicle flow or parking duration;
- PPE/restricted-zone safety candidates;
- reference-image search;
- semantic search over observations;
- cross-camera association;
- anomaly/event detection;
- another defensible VMS capability.

A smaller feature that is accurate, measured and well engineered is better than a large feature set that cannot be validated.

## 8. Measurement

Candidates are strongly encouraged to measure:

- event precision/recall/F1;
- count error;
- ID switches/track fragmentation where practical;
- throughput/FPS or processing time;
- evidence completeness;
- known failure cases.

Run the simple public event-count evaluator against a CSV containing `scenario_id` and `event_type` or `direction`:

```bash
python tools/evaluate_events.py --candidate path/to/your_events.csv
```

This public evaluator is not the entire final score.

## 9. Scoring - 100 points

- Working end-to-end system: 20
- Detection/tracking/event quality: 15
- Unseen-video generalisation: 15
- Originality/useful added capability: 10
- Engineering architecture: 10
- Evidence quality/NOP integration thinking: 10
- Performance/resource handling: 5
- Code quality: 5
- Reproducibility: 5
- Live explanation/modification: 5

Selection flow: **12 candidates -> scored submissions -> top 6 -> unseen-video/live-change stage -> final 4.**

## 10. Submission

Do not submit candidate solutions to this public repository. Use a private repository or other private submission method shared only with TSCI.

Recommended package:

```text
candidate-solution/
  README.md
  REPORT.md
  requirements.txt / environment.yml / build instructions
  src/
  config/
  tests/
  output/
    annotated.mp4
    events.jsonl
    events.csv
    evidence/
  THIRD_PARTY.md
```

`REPORT.md` should explain architecture, model/tracker choices, benchmark results, performance, failure cases, improvements you would make, AI tools used and third-party licensing.

## 11. Final interview

Shortlisted candidates may be asked to:

1. demonstrate the submitted solution end to end;
2. run it on unseen footage;
3. implement a small live change;
4. diagnose a false positive, missed event or ID switch;
5. explain how the capability would integrate into a larger VMS.

## 12. Security and licensing

Never commit API keys, passwords, tokens, private camera URLs, confidential company code, customer footage or other sensitive data to this public repository.

For third-party models, code and datasets, disclose the source and license. A model running successfully does not by itself establish unrestricted commercial-use rights.

For people-related analytics, report observable events rather than claiming hidden mental states or intent.

## 13. Suggested first hour

- 0-10 min: clone, read the rules and run the starter.
- 10-20 min: generate and inspect the four public scenarios.
- 20-30 min: choose detector/tracker/event/evidence architecture.
- 30-45 min: make one scenario work end to end.
- 45-60 min: measure your first result and identify the first failure mode.

Then iterate in this order: **correctness -> robustness -> evidence -> measurement -> extra features.**
