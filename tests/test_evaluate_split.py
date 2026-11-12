import json

import pytest

from common.schema import build_record
from evaluation import evaluate
from evaluation.evaluate import blank_agreement, load_split, split_report

SPLITS = {"dev": {"ixi": ["IXI002"], "molab": ["010005N"], "ms": ["patient05"]},
          "eval": {"ixi": ["IXI012"], "molab": ["010020N"], "ms": ["patient09"]}}
IXI_META = {"dataset": "healthy", "class": "normal"}


def rec(model, dataset, image, diag, mode="full_json", **extra):
    values = {"diagnosis_name": diag, "modality": "MRI", "plane": "axial", **extra}
    return build_record(model, dataset, mode, 1, image, IXI_META, values, None)


def write(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


@pytest.fixture
def tmp(tmp_path):
    (tmp_path / "results").mkdir()
    (tmp_path / "reference").mkdir()
    return tmp_path


def test_reference_filtered_to_split(tmp):
    write(tmp / "reference" / "radfm_healthy_processed.jsonl",
          [rec("radfm", "ixi", "IXI002-Guys-0828-T1.nii.gz", "normal", mode=""),
           rec("radfm", "ixi", "IXI012-HH-1211-T1.nii.gz", "normal", mode="")])
    runs = load_split(tmp / "results", tmp / "reference", "dev", SPLITS)
    assert [r[0] for r in runs["radfm (Outputs, mar)"]["ixi"]] == ["IXI002-Guys-0828-T1.nii.gz"]


def test_full_run_filtered(tmp):
    write(tmp / "results" / "radfm" / "full" / "radfm_healthy_processed.jsonl",
          [rec("radfm", "ixi", "IXI012-HH-1211-T1.nii.gz", "normal"),
           rec("radfm", "ixi", "IXI002-Guys-0828-T1.nii.gz", "normal"),
           rec("radfm", "ixi", "IXI099-HH-1211-T1.nii.gz", "normal")])
    runs = load_split(tmp / "results", tmp / "reference", "eval", SPLITS)
    assert list(runs) == ["radfm (full run)"]
    assert [r[0] for r in runs["radfm (full run)"]["ixi"]] == ["IXI012-HH-1211-T1.nii.gz"]


def test_split_dir_wins_over_full(tmp):
    write(tmp / "results" / "m" / "dev" / "m_healthy_processed.jsonl",
          [rec("m", "ixi", "IXI002-Guys-0828-T1.nii.gz", "normal")])
    write(tmp / "results" / "m" / "full" / "m_healthy_processed.jsonl",
          [rec("m", "ixi", "IXI002-Guys-0828-T1.nii.gz", "normal")])
    assert list(load_split(tmp / "results", tmp / "reference", "dev", SPLITS)) == ["m"]


def test_split_report_numbers(tmp):
    d = tmp / "results" / "m" / "dev"
    write(d / "m_healthy_processed.jsonl", [rec("m", "ixi", "IXI002-Guys-0828-T1.nii.gz", "normal")])
    write(d / "m_brain_metastases_processed.jsonl",
          [rec("m", "molab", f"010005N__010005_{i}_img.nii", "tumor") for i in range(4)])
    write(d / "m_ms_processed.jsonl", [rec("m", "ms", "patient05_T1.nii.gz", "stroke")])
    write(d / "m_ct_rate_processed.jsonl", [])  # unknown tag until CT-RATE exists: skipped, no crash
    runs = load_split(tmp / "results", tmp / "reference", "dev", SPLITS)
    text = "\n".join(split_report(runs, tmp / "results", "dev"))
    assert "| m | full_json | 6 | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% | 66.7% |" in text
    assert "blank agreement" in text and "**m**" in text


def test_molab_patient_majority_and_missing_cells(tmp):
    d = tmp / "results" / "m" / "dev"
    diags = ["normal", "normal", "tumor", "tumor", "stroke"]  # 2/5 scans right; 2-2 tie normal/tumor -> wrong
    write(d / "m_brain_metastases_processed.jsonl",
          [rec("m", "molab", f"010005N__010005_{i}_img.nii", x) for i, x in enumerate(diags)])
    runs = load_split(tmp / "results", tmp / "reference", "dev", SPLITS)
    line = [x for x in split_report(runs, tmp / "results", "dev") if x.startswith("| m |")][0]
    assert line.startswith("| m | full_json | 5 | 100.0% | — | 40.0% | 0.0% | — | — | — | — |")


def _patient_cell(tmp, diags):
    d = tmp / "results" / "m" / "dev"
    write(d / "m_brain_metastases_processed.jsonl",
          [rec("m", "molab", f"010005N__010005_{i}_img.nii", x) for i, x in enumerate(diags)])
    runs = load_split(tmp / "results", tmp / "reference", "dev", SPLITS)
    line = [x for x in split_report(runs, tmp / "results", "dev") if x.startswith("| m |")][0]
    return [c.strip() for c in line.strip("|").split("|")][6]


def test_patient_tie_is_wrong_whatever_the_order(tmp):
    assert _patient_cell(tmp, ["tumor", "tumor", "normal", "normal"]) == "0.0%"
    assert _patient_cell(tmp, ["normal", "normal", "tumor", "tumor"]) == "0.0%"


def test_patient_strict_plurality_without_majority_is_correct(tmp):
    assert _patient_cell(tmp, ["normal", "tumor", None, "tumor"]) == "100.0%"
    assert _patient_cell(tmp, [None, None, "tumor", "normal"]) == "0.0%"


def test_blank_agreement_all_null_is_none(tmp):
    d = tmp / "results" / "m" / "dev"
    nulls = dict(diagnosis_name=None, modality=None, plane=None)
    write(d / "m_healthy_processed.jsonl", [build_record("m", "ixi", "full_json", 1, "IXI002-Guys-0828-T1.nii.gz", IXI_META, nulls, None)])
    write(d / "m_healthy_controls.jsonl",
          [build_record("m", "ixi", "full_json", 1, "BLANK__IXI002-Guys-0828-T1.nii.gz", IXI_META, nulls, None)])
    assert blank_agreement(tmp / "results", "m", "dev") is None


def test_blank_agreement_ignores_null_fields(tmp):
    d = tmp / "results" / "m" / "dev"
    real = dict(diagnosis_name=None, modality="MRI", plane=None)
    write(d / "m_healthy_processed.jsonl",
          [build_record("m", "ixi", "full_json", 1, "IXI002-Guys-0828-T1.nii.gz", IXI_META, real, None)])
    write(d / "m_healthy_controls.jsonl",
          [build_record("m", "ixi", "full_json", 1, "BLANK__IXI002-Guys-0828-T1.nii.gz", IXI_META, real, None)])
    assert blank_agreement(tmp / "results", "m", "dev") == 1.0


def test_legend_states_both_rules(tmp):
    d = tmp / "results" / "m" / "dev"
    write(d / "m_healthy_processed.jsonl", [rec("m", "ixi", "IXI002-Guys-0828-T1.nii.gz", "normal")])
    runs = load_split(tmp / "results", tmp / "reference", "dev", SPLITS)
    text = "\n".join(split_report(runs, tmp / "results", "dev"))
    assert "MOLAB patient majority: correct only with a strict plurality for the truth label (ties count as wrong)." in text
    assert "Blank agreement: share of categorical fields with a non-null usual answer" in text


def test_blank_agreement(tmp):
    d = tmp / "results" / "m" / "dev"
    real = rec("m", "ixi", "IXI002-Guys-0828-T1.nii.gz", "normal", specialized_sequence="T1",
               diagnosis_detailed="glioma", icd10_code="G35")
    write(d / "m_healthy_processed.jsonl", [real])
    same = rec("m", "ixi", "BLANK__IXI002-Guys-0828-T1.nii.gz", "normal", specialized_sequence="T1",
               diagnosis_detailed="glioma", icd10_code="G35")
    write(d / "m_healthy_controls.jsonl", [same])
    assert blank_agreement(tmp / "results", "m", "dev") == 1.0
    other = rec("m", "ixi", "BLANK__IXI002-Guys-0828-T1.nii.gz", "tumor", modality="CT", plane="coronal",
                specialized_sequence="FLAIR", diagnosis_detailed="ischemic", icd10_code="I63")
    write(d / "m_healthy_controls.jsonl", [other])
    assert blank_agreement(tmp / "results", "m", "dev") == 0.0
    assert blank_agreement(tmp / "results", "nope", "dev") is None


def test_seconds_and_blank_columns(tmp):
    d = tmp / "results" / "m" / "dev"
    write(d / "m_healthy_processed.jsonl", [rec("m", "ixi", "IXI002-Guys-0828-T1.nii.gz", "normal")])
    write(d / "m_healthy_controls.jsonl", [rec("m", "ixi", "BLANK__IXI002-Guys-0828-T1.nii.gz", "normal")])
    write(d / "m_healthy_details.jsonl", [{"image_path": "IXI002-Guys-0828-T1.nii.gz", "seconds": 4.0},
                                          {"image_path": "BLANK__IXI002-Guys-0828-T1.nii.gz", "seconds": 100.0}])
    runs = load_split(tmp / "results", tmp / "reference", "dev", SPLITS)
    line = [x for x in split_report(runs, tmp / "results", "dev") if x.startswith("| m |")][0]
    assert line.endswith("| 100.0% | 4.0 |")


def test_selftest_still_passes(capsys):
    evaluate.selftest()
    assert "selftest OK" in capsys.readouterr().out
