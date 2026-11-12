import pytest

from common.parse import extract_from_report, parse_choice, parse_json_answer
from common.schema import PARSED_KEYS


def test_plain_json():
    assert parse_json_answer('{"modality": "MRI"}') == ({"modality": "MRI"}, [])


def test_fenced_answer_tag():
    t = '<think>x</think><answer>```json\n{"plane": "axial",}\n```</answer>'
    assert parse_json_answer(t) == ({"plane": "axial"}, [])


def test_no_json():
    assert parse_json_answer("The image shows") == (None, ["no_json"])


def test_invalid_json():
    assert parse_json_answer('{"modality": MRI}') == (None, ["invalid_json"])


def test_text_around_json_and_nested_braces():
    obj, problems = parse_json_answer('Sure! {"a": {"b": "}"}, "c": 1} done {"x": 2}')
    assert obj == {"a": {"b": "}"}, "c": 1} and problems == []


def test_echoed_placeholder_becomes_null():
    obj, problems = parse_json_answer('{"modality": "<MRI|CT|null>", "plane": "axial"}')
    assert obj == {"modality": None, "plane": "axial"} and "echo:modality" in problems


OPTS = ["Axial", "Sagittal", "Coronal", "Oblique"]


def test_parse_choice():
    assert parse_choice("B. Sagittal", OPTS, "A") == 1
    assert parse_choice("(c) Coronal", OPTS, "a") == 2
    assert parse_choice("sagittal", OPTS, "A") == 1
    assert parse_choice("A. Axial B. Sagittal C. Coronal D. Oblique", OPTS, "A") is None
    assert parse_choice("I don't know", OPTS, "A") is None


def test_parse_choice_bare_letters():
    assert parse_choice("B", OPTS, "A") == 1
    assert parse_choice("B ", OPTS, "A") == 1
    assert parse_choice("(c)", OPTS, "a") == 2
    assert parse_choice("c", OPTS, "a") == 2
    assert parse_choice("D: Oblique", OPTS, "A") == 3


def test_parse_choice_article_is_not_a_letter():
    assert parse_choice("A coronal slice", OPTS, "A") is None
    assert parse_choice("A Axial view", OPTS, "A") == 0


def test_extract_negation():
    v = extract_from_report("No evidence of metastasis. Unremarkable brain MRI.", "molab")
    assert (v["diagnosis_name"], v["modality"]) == ("normal", "MRI")


def test_extract_tumor():
    v = extract_from_report("Enhancing mass in the left frontal lobe consistent with metastasis "
                            "from lung cancer.", "molab")
    assert (v["diagnosis_name"], v["diagnosis_detailed"], v["primary_origin"]) == ("tumor", "carcinoma", "lung-NSCLC")


def test_extract_ms():
    v = extract_from_report("Multiple periventricular hyperintense lesions suggestive of demyelinating disease.", "ms")
    assert (v["diagnosis_name"], v["ms_lesion_location"]) == ("multiple sclerosis", "periventricular")


def test_extract_stroke_and_sequence():
    v = extract_from_report("Axial FLAIR shows an acute infarct. CT is not available.", "ixi")
    assert (v["diagnosis_name"], v["diagnosis_detailed"]) == ("stroke", "ischemic")
    assert (v["specialized_sequence"], v["plane"], v["modality"]) == ("FLAIR", "axial", "CT")


def test_extract_small_cell_and_ms_type():
    assert extract_from_report("Metastasis from small cell lung carcinoma.", "molab")["primary_origin"] == "lung-SCLC"
    assert extract_from_report("Relapsing disease.", "ms")["ms_type"] == "SMRR"


def test_extract_nothing_found():
    v = extract_from_report("Image reviewed.", "ixi")
    assert v["diagnosis_name"] is None and v["modality"] is None


@pytest.mark.parametrize("ds", ["ixi", "molab", "ms", "ct_rate"])
def test_extract_confidences_are_null(ds):
    v = extract_from_report("Enhancing mass with metastasis. MRI.", ds)
    assert set(v) == set(PARSED_KEYS[ds]) - {"scan_name"}
    for k in v:
        if k.endswith("_confidence") or k == "severity_score":
            assert v[k] is None
