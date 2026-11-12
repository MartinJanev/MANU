import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from common import runner
from common.runner import run
from evaluation.check_schema import key_paths

REPO = Path(__file__).resolve().parent.parent
REFERENCE = str(REPO / "reference")
SPLITS = {"dev": {"ixi": ["IXI002"], "molab": ["010005N"], "ms": ["patient05"]}}
FILES = [
    "ixi/T1/IXI002-Guys-0828-T1.nii.gz",
    "ixi/T1/IXI012-HH-1211-T1.nii.gz",
    "molab/nifti/10005_19010618_eTRA-3D-T1Gd_img.nii",
    "molab/nifti/10005_19011011_eTRA-3D-T1Gd_img.nii",
    "open_ms_data/cross_sectional/raw/patient05/T1W.nii.gz",
    "open_ms_data/cross_sectional/raw/patient05/FLAIR.nii.gz",
]


class FakeAdapter:
    name = "fake"
    model_id = "fake/model"
    mode = "full_json"

    def __init__(self):
        self.loads = 0
        self.seen = []  # (image_path, volumes)

    def load(self):
        self.loads += 1

    def predict(self, scan, volumes):
        self.seen.append((scan.image_path, volumes))
        if scan.patient == "IXI002":
            raise RuntimeError("boom")
        return {"modality": "MRI", "diagnosis_name": "tumor"}, {"raw": "x"}


@pytest.fixture
def tmp(tmp_path):
    data = tmp_path / "data"
    rng = np.random.default_rng(0)
    for rel in FILES:
        p = data / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(rng.random((4, 4, 4)).astype(np.float32) + 1, np.eye(4)), str(p))
    return {"data": str(data), "results": str(tmp_path / "results")}


def go(tmp, adapter=None, **kw):
    adapter = adapter or FakeAdapter()
    args = dict(split="dev", datasets=["ixi", "molab", "ms"], data_dir=tmp["data"], reference_dir=REFERENCE,
                results_dir=tmp["results"], splits=SPLITS)
    args.update(kw)
    return adapter, run(adapter, **args)


def rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_dev_run_writes_one_record_per_scan(tmp):
    _, out = go(tmp)
    assert out == Path(tmp["results"]) / "fake" / "dev"
    ixi = rows(out / "fake_healthy_processed.jsonl")
    molab = rows(out / "fake_brain_metastases_processed.jsonl")
    ms = rows(out / "fake_ms_processed.jsonl")
    assert (len(ixi), len(molab), len(ms)) == (1, 2, 1)
    assert [r["processing"]["row_index"] for r in molab] == [0, 1]
    assert all(r["input"]["metadata"]["primary_origin"] == "breast" for r in molab)
    assert molab[0]["output"]["parsed_response"]["modality"] == "MRI"
    ref = rows(REPO / "reference" / "radfm_brain_metastases_processed.jsonl")
    ok_ref = [r for r in ref if r["output"]["status"] != "error"][0]
    assert key_paths(molab[0]) == key_paths(ok_ref)
    assert len(rows(out / "fake_ms_details.jsonl")) == 1


def test_error_scan_is_recorded_and_run_continues(tmp):
    _, out = go(tmp)
    (ixi,) = rows(out / "fake_healthy_processed.jsonl")
    assert ixi["output"]["status"] == "error" and "boom" in ixi["output"]["error"]
    assert all(r["output"]["status"] == "success" for r in rows(out / "fake_ms_processed.jsonl"))
    assert "boom" in rows(out / "fake_healthy_details.jsonl")[0]["error"]


def test_resume_skips_done(tmp):
    adapter, out = go(tmp)
    n = len(adapter.seen)
    go(tmp, adapter)
    assert len(adapter.seen) == n and adapter.loads == 1
    for f in out.glob("*_processed.jsonl"):
        paths = [r["input"]["image_path"] for r in rows(f)]
        assert len(paths) == len(set(paths))


def test_controls_written_once(tmp):
    adapter, out = go(tmp, controls=True)
    adapter2, _ = go(tmp, controls=True)  # all real scans done: only controls would be missing -> none
    assert adapter2.seen == [] and adapter2.loads == 0
    for tag in ("healthy", "brain_metastases", "ms"):
        (rec,) = rows(out / f"fake_{tag}_controls.jsonl")
        assert rec["input"]["image_path"].startswith("BLANK__")
        assert rec["processing"]["row_index"] == -1
    blanks = [(p, v) for p, v in adapter.seen if p.startswith("BLANK__")]
    assert len(blanks) == 3
    assert all(not v.any() and v.shape == (4, 4, 4) for _, vols in blanks for v in vols)


def test_controls_added_after_a_run_without_them(tmp):
    go(tmp)
    adapter, out = go(tmp, controls=True)
    assert adapter.loads == 1
    assert [p for p, _ in adapter.seen if not p.startswith("BLANK__")] == []
    assert len(rows(out / "fake_ms_controls.jsonl")) == 1


def test_ms_flair_name_suffix(tmp):
    adapter, out = go(tmp, ms_flair=True)
    assert out == Path(tmp["results"]) / "fake-flair" / "dev"
    rec = rows(out / "fake-flair_ms_processed.jsonl")[0]
    assert rec["api_response"]["model"] == "fake-flair"


def test_run_info(tmp):
    _, out = go(tmp)
    info = json.loads((out / "run_info.json").read_text())
    assert (info["model"], info["split"], info["mode"]) == ("fake", "dev", "full_json")
    assert info["model_id"] == "fake/model" and info["ms_flair"] is False
    assert "numpy" in info["versions"] and "started" in info and "git_commit" in info


def test_limit(tmp):
    _, out = go(tmp, limit=1)
    for tag in ("healthy", "brain_metastases", "ms"):
        assert len(rows(out / f"fake_{tag}_processed.jsonl")) == 1


def test_main_parses_arguments(tmp, monkeypatch):
    seen = {}
    monkeypatch.setattr(runner, "run", lambda adapter, **kw: seen.update(kw))
    monkeypatch.setattr(runner, "load_splits", lambda p: SPLITS)
    runner.main(FakeAdapter(), ["--split", "dev", "--limit", "2", "--controls", "--ms-flair"])
    assert seen["split"] == "dev" and seen["limit"] == 2 and seen["controls"] and seen["ms_flair"]
    assert seen["datasets"] == ["ixi", "molab", "ms"] and seen["data_dir"] == "/data"
