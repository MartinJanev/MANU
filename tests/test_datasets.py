import json
from pathlib import Path

import pytest

from common.datasets import Scan, find_scans, ground_truth, load_splits, patient_id

REPO = Path(__file__).resolve().parent.parent

FILES = [
    "ixi/T1/IXI002-Guys-0828-T1.nii.gz",
    "ixi/T1/IXI012-HH-1211-T1.nii.gz",
    "molab/nifti/10005_19010618_eTRA-3D-T1Gd_img.nii",
    "molab/nifti/10005_19011011_eTRA-3D-T1Gd_img.nii",
    "molab/nifti/10020_19010822_TRA_3D_T1+GD_img.nii",
    "open_ms_data/cross_sectional/raw/patient05/T1W.nii.gz",
    "open_ms_data/cross_sectional/raw/patient05/FLAIR.nii.gz",
    "open_ms_data/cross_sectional/raw/patient09/T1W.nii.gz",
]


@pytest.fixture
def tree(tmp_path):
    for rel in FILES:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
    return str(tmp_path)


def test_dev_split_resolves_expected_scans(tree):
    dev = {"dev": {"ixi": ["IXI002"], "molab": ["010005N"], "ms": ["patient05"]}}
    assert [s.image_path for s in find_scans(tree, "molab", "dev", dev)] == [
        "010005N__010005_19010618_eTRA-3D-T1Gd_img.nii", "010005N__010005_19011011_eTRA-3D-T1Gd_img.nii"]
    (ms,) = find_scans(tree, "ms", "dev", dev)
    assert (ms.patient, ms.image_path) == ("patient05", "patient05_T1.nii.gz")
    assert ms.files[0].endswith("patient05/T1W.nii.gz")
    (ixi,) = find_scans(tree, "ixi", "dev", dev)
    assert (ixi.patient, ixi.image_path) == ("IXI002", "IXI002-Guys-0828-T1.nii.gz")


def test_ms_flair_variant(tree):
    dev = {"dev": {"ms": ["patient05"]}}
    (ms,) = find_scans(tree, "ms", "dev", dev, ms_flair=True)
    assert ms.files[0].endswith("patient05/FLAIR.nii.gz")
    assert ms.image_path == "patient05_T1.nii.gz"


def test_full_split_returns_all(tree):
    assert len(find_scans(tree, "ixi", "full", {})) == 2
    assert len(find_scans(tree, "molab", "full", {})) == 3
    assert len(find_scans(tree, "ms", "full", {})) == 2


def test_missing_patient_raises(tree):
    with pytest.raises(ValueError, match="IXI999"):
        find_scans(tree, "ixi", "dev", {"dev": {"ixi": ["IXI999"]}})


def test_patient_id():
    assert patient_id("ixi", "IXI012-HH-1211-T1.nii.gz") == "IXI012"
    assert patient_id("molab", "010024N__010024_19001219_TRA_3D_T1+GD_img.nii") == "010024N"
    assert patient_id("ms", "patient10_T1.nii.gz") == "patient10"


def test_real_splits_are_disjoint():
    splits = load_splits(str(REPO / "splits.json"))
    for dataset, dev_ids in splits["dev"].items():
        assert not set(dev_ids) & set(splits["eval"][dataset]), dataset


def test_ground_truth_from_reference(tree):
    ref = str(REPO / "reference")
    dev = {"dev": {"molab": ["010005N"], "ms": ["patient09"], "ixi": ["IXI002"]}}
    (molab, _) = find_scans(tree, "molab", "dev", dev)
    assert ground_truth(molab, ref, tree)["primary_origin"] == "breast"
    (ms,) = find_scans(tree, "ms", "dev", dev)
    assert ground_truth(ms, ref, tree)["ms_type"] == "SMRR"
    (ixi,) = find_scans(tree, "ixi", "dev", dev)
    assert ground_truth(ixi, ref, tree)["class"] == "normal"


def test_ground_truth_unknown_scan_nulls_labels():
    ref = str(REPO / "reference")
    scan = Scan("molab", "099999N", "099999N__099999_x_img.nii", ("/x",))
    meta = ground_truth(scan, ref, "/nowhere")
    assert meta["primary_origin"] is None and meta["dataset"] == "brain_metastases"
