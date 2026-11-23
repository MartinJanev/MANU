"""evaluation/stats.py: every number of the paper."""
import json

import pytest

from common.schema import PARSED_KEYS, build_record
from evaluation import stats

SPLITS = {"paper": {"ixi": [f"IXI{i:03d}" for i in range(20)], "molab": [f"{i:06d}N" for i in range(1, 11)],
                    "ms": [f"patient{i:02d}" for i in range(1, 6)], "ct_rate": [f"valid_{i}" for i in range(4)]}}
IMAGE = {"ixi": "{}-HH-0001-T1.nii.gz", "molab": "{}__x_img.nii", "ms": "{}_T1.nii.gz",
         "ct_rate": "{}_a_1.nii.gz"}
TAG = {"ixi": "healthy", "molab": "brain_metastases", "ms": "ms", "ct_rate": "ct_rate"}


def meta(dataset, i):
    return {"ixi": {}, "molab": {"primary_origin": "breast" if i < 3 else "lung-NSCLC"},
            "ms": {"ms_type": [None, "SMSP", "SMRR", "SMRR", "SMRR"][i % 5]},
            "ct_rate": {"abnormalities": ["Nodule"] if i % 2 else []}}[dataset]


def write(results, name, answer, datasets=("ixi", "molab", "ms", "ct_rate"), drop=0):
    """answer(dataset, i) -> values dict, or None for an error record."""
    folder = results / name / "paper"
    folder.mkdir(parents=True, exist_ok=True)
    for d in datasets:
        patients = SPLITS["paper"][d]
        with open(folder / f"{name}_{TAG[d]}_processed.jsonl", "w") as f:
            for i, p in enumerate(patients[:len(patients) - drop]):
                values = answer(d, i)
                full = {k: None for k in PARSED_KEYS[d]}
                full.update(values or {})
                rec = build_record(name, d, "full_json", i, IMAGE[d].format(p), meta(d, i), full,
                                   "boom" if values is None else None)
                f.write(json.dumps(rec) + "\n")


RIGHT = {"ixi": "normal", "molab": "tumor", "ms": "multiple sclerosis"}


def perfect(d, i):
    if d == "ct_rate":
        return {"diagnosis_name": "other abnormalities" if i % 2 else "normal"}
    return {"diagnosis_name": RIGHT[d]}


def always_normal(d, i):
    return {"diagnosis_name": "normal"}


def test_outcome_keeps_abstention_apart_from_unparseable():
    assert stats.outcome({"diagnosis_name": "tumor"}, None) == "answered"
    assert stats.outcome({"diagnosis_name": None, "modality": "MRI"}, None) == "abstained"
    assert stats.outcome({"diagnosis_name": None, "modality": None, "scan_name": "x"}, None) == "unparseable"
    assert stats.outcome(None, "PredictionError: no json") == "unparseable"


def test_a_constant_answer_scores_one_third_whatever_the_class_sizes(tmp_path):
    write(tmp_path, "m", always_normal)
    run = stats.load_run(tmp_path, "m", SPLITS)
    d = stats.diagnosis(run)
    assert d["bal_acc"] == pytest.approx(1 / 3) and d["recall"] == {"ixi": 1.0, "molab": 0.0, "ms": 0.0}
    assert d["accuracy"] == pytest.approx(20 / 35)


def test_errors_count_as_wrong(tmp_path):
    write(tmp_path, "m", lambda d, i: None if d == "ms" else perfect(d, i))
    assert stats.diagnosis(stats.load_run(tmp_path, "m", SPLITS))["bal_acc"] == pytest.approx(2 / 3)


def test_interval_contains_the_estimate_and_is_reproducible(tmp_path):
    write(tmp_path, "m", lambda d, i: perfect(d, i) if i % 2 else {"diagnosis_name": "stroke"})
    run = stats.load_run(tmp_path, "m", SPLITS)
    low, high = stats.interval(run)
    assert low < stats.diagnosis(run)["bal_acc"] < high and (low, high) == stats.interval(run)


def test_an_incomplete_run_is_not_scored(tmp_path):
    write(tmp_path, "m", perfect, drop=1)
    assert stats.load_run(tmp_path, "m", SPLITS) == {}
    assert stats.load_run(tmp_path, "absent", SPLITS) == {}


def test_a_scan_outside_the_split_is_refused(tmp_path):
    write(tmp_path, "m", perfect, datasets=("ms",))
    other = {"paper": {**SPLITS["paper"], "ms": [f"patient{i:02d}" for i in range(11, 16)]}}
    with pytest.raises(ValueError, match="split"):
        stats.load_run(tmp_path, "m", other)


def test_paired_test(tmp_path):
    write(tmp_path, "good", perfect)
    write(tmp_path, "bad", always_normal)
    good, bad = (stats.load_run(tmp_path, m, SPLITS) for m in ("good", "bad"))
    diff, p = stats.compare(good, bad)
    assert diff == pytest.approx(2 / 3) and p < 0.01
    assert stats.compare(good, good) == (0.0, 1.0)


def test_holm():
    assert stats.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert stats.holm([]) == []


def test_chest_is_scored_balanced(tmp_path):
    write(tmp_path, "m", lambda d, i: {"diagnosis_name": "tumor"})
    c = stats.chest(stats.load_run(tmp_path, "m", SPLITS))
    assert (c["sensitivity"], c["specificity"], c["bal_acc"]) == (1.0, 0.0, 0.5)


def test_fine_fields_have_a_majority_baseline_and_skip_unknown_truth(tmp_path):
    def answer(d, i):
        return {"molab": {"primary_origin": "lung-NSCLC"}, "ms": {"ms_type": "SMRR"}}.get(d, {"diagnosis_name": "normal"})

    write(tmp_path, "m", answer)
    run = stats.load_run(tmp_path, "m", SPLITS)
    origin, subtype = stats.fine(run, "primary_origin"), stats.fine(run, "ms_type")
    assert origin["accuracy"] == origin["baseline"] == pytest.approx(0.7) and origin["macro_f1"] < 0.5
    assert subtype["n"] == 4 and subtype["baseline"] == pytest.approx(0.75)


def test_expected_sequence():
    assert [stats.expected_sequence(d) for d in ("ixi", "molab", "ms")] == ["T1", "T1C+", "T1"]
    assert stats.expected_sequence("ms", flair=True) == "FLAIR"


def test_tables_show_tbd_until_a_run_is_complete_then_the_number(tmp_path):
    results, out = tmp_path / "results", tmp_path / "tables"
    results.mkdir()
    stats.write_tables(results, out, SPLITS)
    table = (out / "results_diagnosis.tex").read_text()
    assert "\\TBD" in table and "MedGemma 1.5" in table and "Constant answer" in table
    write(results, "medgemma-1.5-4b", perfect)
    write(results, "medgemma-1.5-4b-flair", perfect, datasets=("ms",))
    stats.write_tables(results, out, SPLITS)
    (row,) = [line for line in (out / "results_diagnosis.tex").read_text().splitlines() if line.startswith("MedGemma")]
    assert "\\TBD" not in row and "100.0" in row


def test_the_paper_baseline_does_not_overwrite_the_pilots(tmp_path):
    from tools import site_baseline

    assert site_baseline.output_name("eval") == "site_baseline.md"
    assert site_baseline.output_name("paper") == "site_baseline_paper.md"
    (tmp_path / "site_baseline_paper.md").write_text("| features | accuracy |\n|---|---|\n| all | 88.5% |\n")
    assert stats.acquisition_baseline(tmp_path) == pytest.approx(0.885)


def test_a_model_that_failed_validation_is_marked_and_left_out_of_the_comparisons(tmp_path):
    results, out = tmp_path / "results", tmp_path / "tables"
    for name in ("radfm-qna", "medgemma-1.5-4b", "hlip"):
        write(results, name, perfect)
    stats.write_tables(results, out, SPLITS)
    diagnosis = (out / "results_diagnosis.tex").read_text()
    assert "RadFM$^\\dagger$" in diagnosis and "could not be validated" in diagnosis
    pairs = (out / "results_pairs.tex").read_text()
    assert "RadFM" not in pairs and "MedGemma 1.5" in pairs


def test_every_table_is_written(tmp_path):
    stats.write_tables(tmp_path, tmp_path / "t", SPLITS)
    assert {p.name for p in (tmp_path / "t").glob("results_*.tex")} == {
        "results_diagnosis.tex", "results_confusion.tex", "results_chest.tex", "results_descriptive.tex",
        "results_fine.tex", "results_prompting.tex", "results_prior.tex", "results_failures.tex",
        "results_pairs.tex"}
