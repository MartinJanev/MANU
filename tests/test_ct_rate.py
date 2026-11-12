import importlib.util
import json
import sys
from pathlib import Path

import pytest

from common import datasets
from common.datasets import DATASETS, Scan, find_scans, ground_truth, patient_id
from common.schema import build_record
from evaluation.evaluate import ct_rate_table, load_split, split_report

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("prezemi_ct_rate", REPO / "data_tools" / "prezemi_ct_rate.py")
prezemi = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prezemi)
select_files = prezemi.select_files

FILE = "ct_rate/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz"
SPLITS = {"dev": {"ct_rate": ["valid_1"]}, "eval": {"ct_rate": ["valid_2"]}}


@pytest.fixture
def tree(tmp_path):
    p = tmp_path / FILE
    p.parent.mkdir(parents=True)
    p.touch()
    labels = tmp_path / datasets.CT_RATE_LABELS
    labels.parent.mkdir(parents=True)
    labels.write_text("VolumeName,Emphysema,Nodule\nvalid_1_a_1.nii.gz,1,0\nvalid_9_a_1.nii.gz,0,0\n")
    datasets._LABEL_CACHE.clear()
    datasets._WARNED.clear()
    return str(tmp_path)


def test_select_files_prefers_fixed():
    files = ["dataset/valid/valid_1/valid_1_a/valid_1_a_1.nii.gz",
             "dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz",
             "dataset/valid/valid_1/valid_1_a/valid_1_a_2.nii.gz"]
    assert select_files(files, ["valid_1"]) == {"valid_1": files[1]}


def test_select_files_missing_patient_raises():
    files = ["dataset/valid/valid_1/valid_1_a/valid_1_a_1.nii.gz"]
    with pytest.raises(ValueError, match="valid_9"):
        select_files(files, ["valid_1", "valid_9"])


def test_select_files_does_not_confuse_prefix_patients():
    files = ["dataset/valid/valid_12/valid_12_a/valid_12_a_1.nii.gz"]
    with pytest.raises(ValueError, match="valid_1\\b"):
        select_files(files, ["valid_1"])


def test_download_requires_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(SystemExit, match="HF_TOKEN"):
        prezemi.main(["--split", "dev"])


def test_ct_rate_scans(tree):
    (scan,) = find_scans(tree, "ct_rate", "dev", SPLITS)
    assert scan == Scan("ct_rate", "valid_1", "valid_1_a_1.nii.gz", (str(Path(tree) / FILE),))
    assert patient_id("ct_rate", "valid_12_a_1.nii.gz") == "valid_12"
    assert DATASETS["ct_rate"]["experiment"] == "ct_rate"


def test_ct_rate_missing_patient_raises(tree):
    with pytest.raises(ValueError, match="valid_2"):
        find_scans(tree, "ct_rate", "eval", SPLITS)


def test_ct_rate_ground_truth(tree):
    (scan,) = find_scans(tree, "ct_rate", "dev", SPLITS)
    truth = ground_truth(scan, str(REPO / "reference"), tree)
    assert truth["abnormalities"] == ["Emphysema"]
    assert truth["modality"] == "CT" and truth["axial_plane"] == "axial" and truth["dataset"] == "ct_rate"


def test_ct_rate_ground_truth_unknown_is_none_and_warns_once(tree, capsys):
    scan = Scan("ct_rate", "valid_5", "valid_5_a_1.nii.gz", ("x",))
    assert ground_truth(scan, "", tree)["abnormalities"] is None
    assert ground_truth(scan, "", tree)["abnormalities"] is None
    assert capsys.readouterr().out.count("WARNING") == 1
    scan = Scan("ct_rate", "valid_1", "valid_1_a_1.nii.gz", ("x",))
    assert ground_truth(scan, "", str(Path(tree) / "nowhere"))["abnormalities"] is None
    assert capsys.readouterr().out.count("WARNING") == 1


def test_build_record_experiment_id():
    assert build_record("m", "ct_rate", "full_json", 0, "valid_1_a_1.nii.gz", {}, {}, None)["experiment_id"] == "m_ct_rate"


def row(image, abnormalities, **pred):
    meta = {"dataset": "ct_rate", "abnormalities": abnormalities}
    return (image, {"modality": "CT", "plane": "axial", "diagnosis_name": "normal", **pred}, meta)


def test_ct_rate_table():
    runs = {"m": {"ct_rate": [row("valid_1_a_1.nii.gz", ["Emphysema"])]}}
    assert "| m | 1 | 100.0% | 100.0% | 0.0% |" in "\n".join(ct_rate_table(runs))


def test_ct_rate_table_unknown_truth_and_nulls():
    runs = {"m": {"ct_rate": [
        row("valid_1_a_1.nii.gz", ["Emphysema"], diagnosis_name="tumor"),   # agrees
        row("valid_2_a_1.nii.gz", [], diagnosis_name="normal"),             # agrees
        row("valid_3_a_1.nii.gz", None, diagnosis_name="tumor"),            # truth unknown: excluded
        ("valid_4_a_1.nii.gz", None, {"abnormalities": ["Nodule"]}),        # unparsed: wrong everywhere
    ]}, "x": {"ct_rate": [row("valid_3_a_1.nii.gz", None)]}}
    text = "\n".join(ct_rate_table(runs))
    assert "| m | 4 | 75.0% | 75.0% | 66.7% |" in text
    assert "| x | 1 | 100.0% | 100.0% | — |" in text


def test_split_report_appends_ct_rate_table(tmp_path):
    rec = build_record("m", "ct_rate", "full_json", 0, "valid_1_a_1.nii.gz",
                       {"abnormalities": ["Emphysema"]}, {"modality": "CT", "plane": "axial", "diagnosis_name": "other"}, None)
    path = tmp_path / "results" / "m" / "dev" / "m_ct_rate_processed.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(rec) + "\n")
    (tmp_path / "reference").mkdir()
    runs = load_split(tmp_path / "results", tmp_path / "reference", "dev", SPLITS)
    text = "\n".join(split_report(runs, tmp_path / "results", "dev"))
    assert "| m | 1 | 100.0% | 100.0% | 100.0% |" in text
    assert "IXI acc" in text
