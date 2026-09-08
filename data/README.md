# Public Benchmark Data Pack

This folder defines the public evaluation scenarios supplied to every candidate.

The repository intentionally does not store large video corpora. Git hosting should stay lightweight and reproducible. Instead, `tools/generate_synthetic_dataset.py` creates deterministic MP4 clips from the scenario JSON files. Candidates may also test on their own lawful/public footage.

## Generate all public clips

```bash
python tools/generate_synthetic_dataset.py --all --output-dir data/generated
```

## Generated public scenarios

1. `scenario_01_basic_goods.json` - clean A-to-B and B-to-A flows.
2. `scenario_02_occlusion_reversal.json` - temporary occlusion, reversal, stop-and-resume movement.
3. `scenario_03_dense_crossing.json` - multiple simultaneous objects and close crossings.
4. `scenario_04_dwell_queue.json` - dwell-time and queue-style behavior.

The generator writes MP4 video plus a frame-level truth file for each scenario. `data/ground_truth/public_event_truth.csv` provides event-level expected results for the public scenarios.

## Candidate rule

You may train, tune and debug using the public scenarios. Final selection may include unseen footage and live requirement changes, so hard-coding scenario coordinates, object IDs, exact event counts or filenames will not generalize.
