import math
import os

import nibabel as nib
import numpy as np
import pytest

from common.datasets import Scan
from common.schema import normalize_values
from common.volume import load_canonical
from models.prima import run as prima
from models.prima.run import GROUP_HEADS, NORMAL_THRESHOLD, PrimaAdapter, decide, series_files, sigmoid

pytest.importorskip("pydicom")

HEADS = ["cyst_arachnoid_cyst", "inflammatory_multiple_sclerosis", "tumor_adult_brain_metastasis",
         "tumor_adult_glioma", "tumor_adult_high_grade_glioma", "tumor_adult_low_grade_glioma",
         "tumor_adult_meningioma", "tumor_adult_lymphoma", "vascular_ischemic_lacunar_stroke"]


def _values(**overrides):
    v = {h: -3.0 for h in HEADS}
    v.update(overrides)
    return v


def test_groups_follow_the_brief():
    assert GROUP_HEADS["multiple sclerosis"] == ["inflammatory_multiple_sclerosis"]
    assert set(GROUP_HEADS["tumor"]) == {"tumor_adult_brain_metastasis", "tumor_adult_glioma",
                                         "tumor_adult_high_grade_glioma", "tumor_adult_low_grade_glioma",
                                         "tumor_adult_meningioma"}
    assert NORMAL_THRESHOLD == 0.5


def test_normal_when_every_group_head_is_below_prima_cut_off():
    values, groups = decide(_values(tumor_adult_lymphoma=4.0, vascular_ischemic_lacunar_stroke=2.0))
    assert values["diagnosis_name"] == "normal"  # heads outside the groups do not count
    assert values["diagnosis_confidence"] == pytest.approx(1 - sigmoid(-3.0))
    assert groups == pytest.approx({"tumor": sigmoid(-3.0), "multiple sclerosis": sigmoid(-3.0)})


def test_most_probable_group_wins_with_its_probability():
    values, groups = decide(_values(tumor_adult_meningioma=0.4, inflammatory_multiple_sclerosis=1.5))
    assert values == {"diagnosis_name": "multiple sclerosis", "diagnosis_confidence": pytest.approx(sigmoid(1.5))}
    values, _ = decide(_values(tumor_adult_brain_metastasis=0.0))
    assert values == {"diagnosis_name": "tumor", "diagnosis_confidence": 0.5}  # p = 0.5 is Prima's "present"


def test_only_diagnosis_fields_are_filled():
    values, _ = decide(_values())
    for dataset in ("ixi", "molab", "ms", "ct_rate"):
        filled = {k for k, v in normalize_values(dataset, values).items() if v is not None}
        assert filled == {"diagnosis_name", "diagnosis_confidence"}


def _nifti(path, shape, affine):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = (np.random.default_rng(len(path)).random(shape) * 500 + 10).astype(np.float32)
    nib.save(nib.Nifti1Image(data, affine), path)
    return data


@pytest.fixture
def ms_scan(tmp_path):
    folder = tmp_path / "raw" / "patient09"
    t1 = _nifti(str(folder / "T1W.nii.gz"), (10, 12, 6), np.diag([0.5, 0.5, 1.0, 1.0]))
    flair = _nifti(str(folder / "FLAIR.nii.gz"), (8, 9, 5), np.diag([0.6, 0.6, 1.2, 1.0]))
    scan = Scan("ms", "patient09", "patient09_T1.nii.gz", (str(folder / "T1W.nii.gz"),))
    return scan, t1, flair


def test_ms_study_has_t1_and_flair(ms_scan):
    scan, _, _ = ms_scan
    assert [n for n, _ in series_files(scan)] == ["T1", "FLAIR"]
    assert series_files(scan)[1][1].endswith("patient09/FLAIR.nii.gz")


def test_ixi_and_molab_are_one_series():
    for dataset in ("ixi", "molab"):
        scan = Scan(dataset, "p", "p.nii.gz", ("/data/p.nii.gz",))
        assert series_files(scan) == [(prima.SERIES[dataset][0], "/data/p.nii.gz")]


def _read(folder):
    import SimpleITK as sitk

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(sorted(str(p) for p in folder.glob("*.dcm")))
    return sitk.GetArrayFromImage(reader.Execute())


def test_write_study_keeps_each_file(ms_scan, tmp_path):
    pytest.importorskip("SimpleITK")
    scan, t1, flair = ms_scan
    written = prima.write_study(scan, [load_canonical(scan.files[0])], str(tmp_path / "study"))
    assert [w["series"] for w in written] == ["T1", "FLAIR"]
    assert sorted(os.listdir(tmp_path / "study")) == ["1_T1", "2_FLAIR"]
    for folder, data in (("1_T1", t1), ("2_FLAIR", flair)):
        arr = _read(tmp_path / "study" / folder)  # (k, j, i)
        assert np.abs(arr - data.transpose(2, 1, 0)).max() < 0.02


def test_blank_control_gives_zero_series(ms_scan, tmp_path):
    pytest.importorskip("SimpleITK")
    scan, _, _ = ms_scan
    blank = np.zeros_like(load_canonical(scan.files[0]))
    written = prima.write_study(scan, [blank], str(tmp_path / "study"))
    assert written[1]["shape"] == [8, 9, 5]
    for folder in ("1_T1", "2_FLAIR"):
        assert (_read(tmp_path / "study" / folder) == 0).all()


def test_predict_records_all_heads(monkeypatch, ms_scan, tmp_path):
    scan, _, _ = ms_scan
    adapter = PrimaAdapter()
    adapter.workdir = str(tmp_path)
    adapter.thresholds = {h: -0.4 for h in HEADS}
    seen = {}

    def fake_infer(study_dir, output_dir):
        seen["series"] = sorted(os.listdir(study_dir))
        return {"diagnosis": _values(inflammatory_multiple_sclerosis=0.7),
                "referral": {"neurology": 0.2}, "priority": {"none": 1.0, "low": 0.0, "high": -1.0}}, {"T1": 300}

    monkeypatch.setattr(adapter, "_infer", fake_infer)
    values, details = adapter.predict(scan, [load_canonical(scan.files[0])])
    assert seen["series"] == ["1_T1", "2_FLAIR"]
    assert values == {"diagnosis_name": "multiple sclerosis", "diagnosis_confidence": pytest.approx(sigmoid(0.7))}
    assert details["diagnosis_logit_minus_thresh"] == _values(inflammatory_multiple_sclerosis=0.7)
    assert details["diagnosis_p"]["inflammatory_multiple_sclerosis"] == pytest.approx(1 / (1 + math.exp(-0.7)))
    assert details["normal_threshold"] == 0.5 and details["referral"] == {"neurology": 0.2}
    assert details["diagnosis_thresh"]["cyst_arachnoid_cyst"] == -0.4


def test_pipeline_imports_prima_tools_not_manu_tools(monkeypatch, tmp_path):
    import sys

    import tools as manu_tools  # MANU's tools/ package, as /manu is on PYTHONPATH in the container

    src = tmp_path / "prima"
    (src / "tools").mkdir(parents=True)
    (src / "tools" / "__init__.py").write_text("")
    (src / "tools" / "DicomUtils.py").write_text("WHO = 'prima'\n")
    (src / "end-to-end_inference_pipeline").mkdir()
    (src / "end-to-end_inference_pipeline" / "pipeline.py").write_text(
        "from tools.DicomUtils import WHO\nclass Pipeline:\n    who = WHO\n")
    monkeypatch.setattr(prima, "PRIMA_SRC", str(src))
    monkeypatch.setattr(sys, "path", list(sys.path))
    saved = {k: v for k, v in sys.modules.items() if k == "tools" or k.startswith("tools.")}
    try:
        assert manu_tools.__file__ and "prima" not in manu_tools.__file__
        assert prima._import_pipeline().who == "prima"
    finally:
        for k in [k for k in sys.modules if k == "tools" or k.startswith("tools.")]:
            del sys.modules[k]
        sys.modules.update(saved)
