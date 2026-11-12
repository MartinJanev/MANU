import json
from pathlib import Path

import pytest

from common.schema import PARSED_KEYS, build_record, full_prompt, normalize_values
from evaluation.check_schema import key_paths

REPO = Path(__file__).resolve().parent.parent


def first_success_row(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["output"]["status"] != "error":
                return row
    raise AssertionError(f"no success row in {path}")


@pytest.mark.parametrize("ds,tag", [("ixi", "healthy"), ("molab", "brain_metastases"), ("ms", "ms")])
def test_record_matches_reference_structure(ds, tag):
    ref = first_success_row(REPO / f"reference/radfm_{tag}_processed.jsonl")
    rec = build_record("lingshu-7b", ds, "full_json", 0, "x.nii.gz", ref["input"]["metadata"],
                       normalize_values(ds, {}), None)
    assert key_paths(rec) == key_paths(ref)


def test_experiment_and_mode():
    rec = build_record("lingshu-7b", "molab", "full_json", 0, "a", {}, {}, None)
    assert rec["experiment_id"] == "lingshu-7b_brain_metastases"
    assert rec["input"]["system_prompt_type"] == "full_json_v1"
    assert rec["input"]["model_requested"] == "lingshu-7b"
    assert rec["api_response"]["model"] == "lingshu-7b"
    assert rec["output"]["parsed_response"]["scan_name"] == "a"


def test_scan_name_only_for_molab():
    assert "scan_name" not in build_record("m", "ms", "full_json", 0, "a", {}, {}, None)["output"]["parsed_response"]


def test_error_record():
    rec = build_record("m", "ixi", "full_json", 3, "a.nii.gz", {"k": 1}, {}, "RuntimeError('OOM')")
    assert rec["output"]["parsed_response"] is None
    assert rec["output"]["status"] == "error"
    assert rec["output"]["error"] == "RuntimeError('OOM')"
    assert rec["processing"]["row_index"] == 3
    assert rec["input"]["metadata"] == {"k": 1}


def test_normalize_values():
    v = normalize_values("molab", {"modality": "mri", "specialized_sequence": "T1 with contrast",
                                   "diagnosis_name": "Tumor", "primary_origin": "overy",
                                   "severity_score": "1.4", "extra": 1})
    assert (v["modality"], v["specialized_sequence"], v["diagnosis_name"], v["primary_origin"],
            v["severity_score"]) == ("MRI", "T1C+", "tumor", "ovary", 1.0)
    assert set(v) == set(PARSED_KEYS["molab"]) - {"scan_name"}
    assert normalize_values("molab", v) == v
    assert normalize_values("ixi", {"diagnosis_name": "brain lesion"})["diagnosis_name"] is None


def test_normalize_synonyms_and_types():
    assert normalize_values("molab", {"primary_origin": "blader"})["primary_origin"] == "bladder"
    assert normalize_values("molab", {"primary_origin": "unknown"})["primary_origin"] == "unknown"
    ms = normalize_values("ms", {"ms_type": "relapsing-remitting", "ms_active_lesions": "true",
                                 "ms_confidence": "abc", "ms_lesion_burden": "High"})
    assert ms["ms_type"] == "SMRR" and ms["ms_active_lesions"] is True
    assert ms["ms_confidence"] is None and ms["ms_lesion_burden"] == "high"
    assert normalize_values("ms", {"ms_type": "secondary progressive"})["ms_type"] == "SMSP"
    assert normalize_values("ms", {"ms_type": "primary progressive"})["ms_type"] == "SMPP"
    assert normalize_values("ms", {"ms_active_lesions": 3})["ms_active_lesions"] is None
    assert normalize_values("ixi", {"severity_score": -0.3})["severity_score"] == 0.0
    assert normalize_values("ixi", {"severity_score": True})["severity_score"] is None
    for seq in ("T1Gd", "T1-weighted with contrast", "T1 post-contrast"):
        assert normalize_values("ixi", {"specialized_sequence": seq})["specialized_sequence"] == "T1C+"


def test_icd10_filter():
    assert normalize_values("ixi", {"icd10_code": "C71.9"})["icd10_code"] == "C71.9"
    assert normalize_values("ixi", {"icd10_code": "I63"})["icd10_code"] == "I63"
    assert normalize_values("ixi", {"icd10_code": "brain tumor"})["icd10_code"] is None


@pytest.mark.parametrize("ds", ["ixi", "molab", "ms", "ct_rate"])
def test_full_prompt_lists_dataset_keys(ds):
    assert all(f'"{k}"' in full_prompt(ds) for k in PARSED_KEYS[ds] if k != "scan_name")


def test_ct_rate_prompt_is_not_head_specific():
    assert "a medical scan (CT or MRI)" in full_prompt("ct_rate")
    assert "head scan" not in full_prompt("ct_rate")
