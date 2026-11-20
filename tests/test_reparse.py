import json

from common.schema import build_record
from tools.reparse import reparse

MODEL = "healthgpt-pro-27b"
META = {"dataset": "brain_metastases", "class": "tumor"}
RAW = '{\n  "modality": "MRI",\n  "diagnosis_name": "tumor"\n}'


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_truncated_think_rows_are_rescored(tmp_path):
    out = tmp_path / MODEL / "eval"
    out.mkdir(parents=True)
    (out / "run_info.json").write_text(json.dumps({"model": MODEL, "mode": "full_json", "started": "t"}))
    record = build_record(MODEL, "molab", "full_json", 0, "010020N__a_img.nii", META, {}, None)  # all values null
    _write(out / f"{MODEL}_brain_metastases_processed.jsonl", [record])
    _write(out / f"{MODEL}_brain_metastases_details.jsonl", [{
        "image_path": "010020N__a_img.nii", "error": None,
        "details": {"raw_values": {}, "raw": RAW, "problems": ["truncated_think"]}}])

    backup = reparse(tmp_path, MODEL, "eval")

    row = json.loads((out / f"{MODEL}_brain_metastases_processed.jsonl").read_text())
    assert row["output"]["parsed_response"]["diagnosis_name"] == "tumor"
    assert row["input"]["metadata"] == META and row["timestamp"] == record["timestamp"]
    detail = json.loads((out / f"{MODEL}_brain_metastases_details.jsonl").read_text())
    assert detail["details"]["problems"] == [] and detail["details"]["raw_values"]["diagnosis_name"] == "tumor"
    assert "truncated_think" in (backup / f"{MODEL}_brain_metastases_details.jsonl").read_text()
    assert json.loads((out / "run_info.json").read_text())["reparsed"]
