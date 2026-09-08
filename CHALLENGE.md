# Challenge Specification

## NOP Vision Intelligence Engineering Challenge

You are given a deliberately simple cross-platform video-processing baseline. Your job is to turn it into a useful, reliable VMS/video-intelligence capability.

### Mandatory outcome

Your solution must:

1. Accept prerecorded video as input.
2. Detect at least one useful target/object category.
3. Maintain track identity over time.
4. Use configurable event geometry such as zones, lines, dwell regions or equivalent.
5. Generate events from tracked behaviour/movement, not from raw frame count.
6. Avoid obvious duplicate counting.
7. Produce annotated video or equivalent visual evidence.
8. Produce machine-readable JSON evidence.
9. Produce CSV summary output.
10. Save evidence snapshots for generated events.
11. Include clear setup/run instructions and a short technical report.

### Platform freedom

You may use Linux, Windows, macOS or another reasonable development platform. You may use CPU or GPU. You may replace any part of the starter implementation.

### Technology freedom

You may use any suitable language/framework/model, including OpenCV, ONNX Runtime, PyTorch, TensorFlow, TensorRT, OpenVINO, Core ML, CUDA, DirectML, Metal/MPS or others.

AI-assisted development is allowed. You may use ChatGPT, Codex, Claude, Gemini, Copilot or similar tools.

### Build something useful

After completing the mandatory base, add at least one meaningful improvement. Examples include:

- goods/package A-to-B counting;
- person entry/exit and occupancy;
- queue length and dwell analytics;
- vehicle flow and parking duration;
- PPE/restricted-zone safety events;
- object reference-image search;
- semantic search over indexed observations;
- cross-camera association;
- anomaly/event detection;
- another VMS capability you can justify.

### Hidden evaluation

Finalists may be asked to run their solution on unseen footage and make a small live change. Build for generalisation and understand your own code.

### What matters

Accuracy, robustness, evidence quality, engineering judgement, reproducibility, performance, originality, and your ability to explain/debug/modify the solution.