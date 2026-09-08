import json
from pathlib import Path
import jsonschema


def test_all_emitted_jsonl_against_nop_contract():
    contract = json.loads(Path("nop_reference/evidence_contract.json").read_text(encoding="utf-8"))

    for sc in ["S01_BASIC_GOODS", "S02_OCCLUSION_REVERSAL", "S03_DENSE_CROSSING", "S04_DWELL_QUEUE"]:
        jsonl_path = Path(f"output/{sc}/events.jsonl")
        assert jsonl_path.exists(), f"Missing {jsonl_path}"

        lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) > 0, f"No records in {jsonl_path}"

        for line in lines:
            record = json.loads(line)
            jsonschema.validate(instance=record, schema=contract)
