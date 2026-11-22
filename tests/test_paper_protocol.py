"""The protocol of the paper's runs (split "paper"): one test per flaw found in the pilot, plus its split."""
import importlib.util
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from common.datasets import CT_RATE_LABELS, Scan, find_scans, ground_truth
from common.parse import extract_from_report, parse_json_answer
from common.schema import full_prompt, slice_preface
from common.volume import load_canonical, to_rgb_images
from tools import make_split

REPO = Path(__file__).resolve().parent.parent
REFERENCE = str(REPO / "reference")


def touch(root, *rels):
    for rel in rels:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()


# geometry: the pilot stretched IXI brains 1.35x sideways

@pytest.mark.parametrize("shape,zooms", [((150, 256, 20), (1.2, 0.9375, 3.0)),
                                         ((256, 150, 20), (0.9375, 1.2, 3.0)),
                                         ((256, 192, 20), (1.0, 1.0, 1.0))])
def test_round_object_stays_round_in_the_model_images(tmp_path, shape, zooms):
    x, y, _ = np.indices(shape)
    disc = ((x - shape[0] / 2) * zooms[0]) ** 2 + ((y - shape[1] / 2) * zooms[1]) ** 2 < 45 ** 2
    path = tmp_path / "v.nii.gz"
    nib.save(nib.Nifti1Image(disc.astype(np.float32), np.diag([*zooms, 1.0])), str(path))
    vol = load_canonical(str(path))
    assert vol.shape[0] == shape[2] and vol.shape[1] == vol.shape[2]  # depth untouched, square slices
    image = np.asarray(to_rgb_images(vol, [10], 448)[0])[..., 0] > 127
    assert image.any(0).sum() / image.any(1).sum() == pytest.approx(1.0, abs=0.02)


def test_padding_uses_the_volume_minimum(tmp_path):
    data = np.full((4, 8, 3), -1000, np.float32)
    data[1:3] = 50
    path = tmp_path / "ct.nii.gz"
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))
    vol = load_canonical(str(path))
    assert vol.shape == (3, 8, 8) and set(np.unique(vol)) == {-1000.0, 50.0}  # air, not 0 HU (water)


# prompt: the pilot's prompt named the disease of the dataset

def test_prompt_is_identical_for_every_dataset():
    prompts = {full_prompt(d) for d in ("ixi", "molab", "ms", "ct_rate")}
    assert len(prompts) == 1
    prompt = prompts.pop()
    assert "Examples" not in prompt and "head scan" not in prompt
    for field in ("primary_origin", "ms_type", "diagnosis_name", "icd10_code"):
        assert f'"{field}"' in prompt


def test_slice_preface_does_not_give_away_the_slice_count():
    assert not any(c.isdigit() for c in slice_preface().replace("3D", ""))


# labels: the pilot's ms_type came from MSLesSeg

def test_ms_type_comes_from_open_ms_data(tmp_path):
    info = tmp_path / "open_ms_data/cross_sectional/patient_info.csv"
    info.parent.mkdir(parents=True)
    info.write_text("patient_id,age,sex,ms_type,edss,criteria\n2,33,M,CIS,0.0,x\n4,25,M,SP,6.5,x\n"
                    "13,26,M,RR,2.0,x\n15,57,F,PR,6.5,x\n")

    def ms_type(patient):
        scan = Scan("ms", patient, f"{patient}_T1.nii.gz", ("/x",))
        return ground_truth(scan, REFERENCE, str(tmp_path))["ms_type"]

    assert [ms_type(f"patient{i}") for i in ("04", "13", "15")] == ["SMSP", "SMRR", "SMPP"]
    assert ms_type("patient02") is None  # CIS is outside the schema's three types: not scored


def test_ms_type_is_null_without_the_source_file(tmp_path):
    scan = Scan("ms", "patient13", "patient13_T1.nii.gz", ("/x",))
    assert ground_truth(scan, REFERENCE, str(tmp_path))["ms_type"] is None


# parsers

@pytest.mark.parametrize("impression", ["Normal head CT.", "Normal head MRI.", "Normal brain."])
def test_report_normal_impressions(impression):
    report = f"FINDINGS: The ventricles are normal. IMPRESSION: {impression}"
    assert extract_from_report(report, "molab")["diagnosis_name"] == "normal"


def test_report_parenchyma_appears_normal():
    report = "The brain parenchyma appears normal. No mass lesions are seen. No infarcts are seen."
    assert extract_from_report(report, "ms")["diagnosis_name"] == "normal"


def test_report_finding_beats_unremarkable_bone():
    report = "There is an enhancing mass in the left frontal lobe. The brain parenchyma appears normal elsewhere."
    assert extract_from_report(report, "molab")["diagnosis_name"] == "tumor"


def test_one_object_per_slice_takes_the_strict_majority():
    one = '{"diagnosis_name": "%s", "modality": "MRI"}'
    text = "[" + ",".join(one % d for d in ("normal", "tumor", "tumor", "tumor")) + "]"
    obj, problems = parse_json_answer(text)
    assert (obj["diagnosis_name"], obj["modality"]) == ("tumor", "MRI") and "multiple_objects" in problems


def test_one_object_per_slice_abstains_on_a_tie():
    text = '[{"diagnosis_name": "normal", "modality": "MRI"}, {"diagnosis_name": "tumor", "modality": "MRI"}]'
    obj, _ = parse_json_answer(text)
    assert obj["diagnosis_name"] is None and obj["modality"] == "MRI"


# split "paper"

def test_paper_split_keeps_only_the_first_molab_scan(tmp_path):
    touch(tmp_path, "molab/nifti/10005_19011011_eTRA-3D-T1Gd_img.nii",
          "molab/nifti/10005_19010618_eTRA-3D-T1Gd_img.nii")
    splits = {"paper": {"molab": ["010005N"]}, "eval": {"molab": ["010005N"]}}
    (first,) = find_scans(str(tmp_path), "molab", "paper", splits)
    assert "19010618" in first.image_path
    assert len(find_scans(str(tmp_path), "molab", "eval", splits)) == 2  # the pilot split is as it was


def test_paper_split_cohorts(tmp_path):
    touch(tmp_path, *[f"ixi/T1/IXI{i:03d}-HH-0001-T1.nii.gz" for i in range(1, 201)],
          *[f"molab/nifti/{10000 + i}_19010618_eTRA-3D-T1Gd_img.nii" for i in range(1, 8)],
          *[f"open_ms_data/cross_sectional/raw/patient{i:02d}/T1W.nii.gz" for i in range(1, 31)])
    csv_path = tmp_path / CT_RATE_LABELS
    csv_path.parent.mkdir(parents=True)
    csv_path.write_text("VolumeName,Nodule\n" + "".join(f"valid_{i}_a_1.nii.gz,{i % 2}\n" for i in range(1, 121)))
    splits = {"dev": {"ixi": ["IXI001"], "molab": ["010001N"], "ms": ["patient05"], "ct_rate": ["valid_1"]},
              "eval": {"ixi": ["IXI002", "IXI003"], "molab": [], "ms": [], "ct_rate": ["valid_2", "valid_3"]}}
    proposed, _ = make_split.propose_paper(str(tmp_path), splits)
    paper = proposed["paper"]
    assert len(paper["ixi"]) == 120 and not set(paper["ixi"]) & {"IXI001", "IXI002", "IXI003"}
    assert len(paper["molab"]) == 7 and len(paper["ms"]) == 30  # whole cohorts, as in the paper
    ct = paper["ct_rate"]
    abnormal = sum(int(p.split("_")[1]) % 2 for p in ct)
    assert (len(ct), abnormal) == (60, 30) and "valid_1" not in ct
    assert {"valid_2", "valid_3"} <= set(ct)  # already downloaded
    assert proposed["dev"] == splits["dev"] and proposed["eval"] == splits["eval"]
    assert make_split.propose_paper(str(tmp_path), splits)[0] == proposed


def test_ct_download_accepts_the_paper_split(tmp_path):
    spec = importlib.util.spec_from_file_location("prezemi", REPO / "data_tools" / "prezemi_ct_rate.py")
    prezemi = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prezemi)
    path = tmp_path / "s.json"
    path.write_text('{"paper": {"ct_rate": ["valid_7"]}}')
    assert prezemi.split_patients(str(path), "paper") == ["valid_7"]
    assert "paper" in prezemi.SPLITS


def test_runner_accepts_the_paper_split():
    from common import runner

    assert "paper" in runner.SPLIT_NAMES
