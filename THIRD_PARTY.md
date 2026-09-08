# Third-Party Components & Licensing Disclosures

In compliance with `RULES.md` (Rule 7 & AI-use disclosure) and `SUBMISSION.md`:

| Component / Library | Version | License | Source / Repository | Purpose in System |
| :--- | :--- | :--- | :--- | :--- |
| **OpenCV (`opencv-python`)** | `>=4.8.0` | Apache 2.0 | [OpenCV GitHub](https://github.com/opencv/opencv) | Video I/O, color conversion, morphological operators, contour extraction, drawing |
| **NumPy (`numpy`)** | `>=1.24.0` | BSD-3-Clause | [NumPy GitHub](https://github.com/numpy/numpy) | Matrix operations, feature normalization, spatial transformations |
| **SciPy (`scipy`)** | `>=1.11.0` | BSD-3-Clause | [SciPy GitHub](https://github.com/scipy/scipy) | Bipartite matching via Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) |
| **ONNX Runtime (`onnxruntime`)**| `>=1.16.0` | MIT | [Microsoft ONNX Runtime](https://github.com/microsoft/onnxruntime) | Neural network inference engine for unseen real footage (DirectML/CPU) |
| **JSONSchema (`jsonschema`)** | `>=4.20.0` | MIT | [python-jsonschema](https://github.com/python-jsonschema/jsonschema) | Automated verification of emitted JSON against `nop_reference/evidence_contract.json` |
| **Pytest (`pytest`)** | `>=8.0.0` | MIT | [Pytest GitHub](https://github.com/pytest-dev/pytest) | Automated regression test harness for Kalman filter, ByteTrack matching, and journeys |

### Model Disclosures
- **BenchmarkDetector**: Bespoke algorithmic computer vision detector with zero pretrained neural weights (Apache 2.0 compatible, fully proprietary to candidate).
- **OnnxDetector**: Standard open ONNX execution architecture supporting COCO-pretrained YOLO models (e.g. YOLOv8n under AGPL-3.0 / Enterprise license or proprietary models).
