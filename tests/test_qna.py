import pytest

from common.qna import BANK, Question, ask, format_mcq, interrogate

PLANE_Q = "Which plane is displayed in the image?"


def test_format_mcq_styles():
    q = BANK["plane"]
    assert format_mcq(q, "A", (0, 1, 2, 3)) == \
        "Which plane is displayed in the image? Choices: A. Axial B. Sagittal C. Coronal D. Oblique"
    assert format_mcq(q, "a", (0, 1, 2, 3)) == \
        "Which plane is displayed in the image? (a) Axial (b) Sagittal (c) Coronal (d) Oblique"


def test_format_mcq_reversed_order():
    assert format_mcq(BANK["plane"], "A", (3, 2, 1, 0)) == \
        "Which plane is displayed in the image? Choices: A. Oblique B. Coronal C. Sagittal D. Axial"


def test_bank_shape():
    flat = []
    for entry in BANK.values():
        flat.extend(entry if isinstance(entry, list) else [entry])
    assert flat and all(isinstance(q, Question) for q in flat)
    assert all(len(q.options) == 4 and len(q.values) == 4 for q in flat)
    assert BANK["severity_score"].values == (0.0, 0.25, 0.5, 0.75)
    assert BANK["diagnosis_detailed"][0].values[2] == "carcinoma"
    assert len(BANK["diagnosis_detailed"]) == 3 and len(BANK["primary_origin"]) == 4


def test_reverse_order_maps_back():
    def fake(prompt):
        return "Sagittal", None
    r = ask(BANK["plane"], fake, "A")
    assert r["value"] == "sagittal" and r["agree"] is True and r["method"] == "option_text"
    assert r["confidence"] is None


def test_position_bias_flagged():
    r = ask(BANK["plane"], lambda p: ("A.", None), "A")
    assert r["value"] == "axial" and r["agree"] is False and r["method"] == "letter"


def test_echo_of_all_options_is_none():
    r = ask(BANK["plane"], lambda p: (p, None), "A")
    assert r["value"] is None and r["method"] == "none" and r["agree"] is False


def test_confidence_is_original_order_probability():
    def fake(prompt):
        return "B.", [0.1, 0.6, 0.2, 0.1]
    r = ask(BANK["plane"], fake, "A")
    assert r["value"] == "sagittal" and r["confidence"] == pytest.approx(0.6)


def test_reversed_probabilities_are_mapped_back():
    def fake(prompt):
        if "A. Axial" in prompt:
            return "A.", [0.7, 0.1, 0.1, 0.1]
        return "D.", [0.1, 0.1, 0.1, 0.7]      # reversed ask, displayed D = Axial
    r = ask(BANK["plane"], fake, "A")
    assert r["agree"] is True
    assert r["answers"][1]["probs"] == pytest.approx([0.7, 0.1, 0.1, 0.1])


def test_ct_skips_sequence():
    def fake(prompt):
        return ("B." if "modality" in prompt else "A."), None
    values, log = interrogate("ixi", fake, "A")
    assert values["modality"] == "CT"
    assert values["specialized_sequence"] is None
    assert "specialized_sequence" not in log
    assert values["icd10_code"] is None


def test_mri_asks_sequence():
    values, log = interrogate("ixi", lambda p: ("A.", None), "A")
    assert values["specialized_sequence"] == "T1" and "specialized_sequence" in log


def test_diagnosis_and_severity_confidence():
    def fake(prompt):
        return "A.", [0.5, 0.2, 0.2, 0.1]
    values, _ = interrogate("ixi", fake, "A")
    assert values["diagnosis_name"] == "normal"
    assert values["diagnosis_confidence"] == pytest.approx(0.5)
    assert values["severity_score"] == 0.0 and values["severity_confidence"] == pytest.approx(0.5)
    assert values["diagnosis_detailed"] is None


def test_tumor_detailed_hierarchy_and_stroke():
    def tumor(prompt):
        if "main finding" in prompt:
            return "B.", None
        if "What type of tumor" in prompt and "Pituitary" in prompt:
            return "D.", None          # R2 -> Other
        if "What type of tumor" in prompt and "Glioma" in prompt:
            return "D.", None          # R1 -> Other
        return "A.", None              # R3 -> Germinoma
    values, _ = interrogate("ixi", tumor, "A")
    assert values["diagnosis_name"] == "tumor" and values["diagnosis_detailed"] == "germinoma"

    def stroke(prompt):
        return ("C." if "main finding" in prompt else "B."), None
    values, _ = interrogate("ixi", stroke, "A")
    assert values["diagnosis_detailed"] == "hemorrhagic"


def test_ms_fields():
    values, log = interrogate("ms", lambda p: ("A.", [0.4, 0.3, 0.2, 0.1]), "A")
    for key in ("ms_type", "ms_lesion_location", "ms_lesion_burden", "ms_confidence"):
        assert values[key] is not None
    assert values["ms_type"] == "SMRR" and values["ms_lesion_location"] == "periventricular"
    assert values["ms_active_lesions"] is None and "ms_active_lesions" not in log
    assert values["ms_confidence"] == pytest.approx(0.1)   # P("Multiple sclerosis"), the fourth option


def test_origin_hierarchy():
    def fake(prompt):
        if "A. Lung" in prompt:             # R1 -> Other, R2 -> Kidney
            return "D.", [0.1, 0.2, 0.3, 0.4]
        return "A.", [0.1, 0.2, 0.3, 0.4]
    values, _ = interrogate("molab", fake, "A")
    assert values["primary_origin"] == "kidney"
    assert values["origin_confidence"] == pytest.approx(0.4 * 0.1)
    assert values["primary_origin_subtype"] is None and values["subtype_confidence"] is None


def test_origin_lung_asks_lung_type():
    def fake(prompt):
        if "small cell or non-small" in prompt:
            return "B.", [0.1, 0.5, 0.2, 0.2]
        return "A.", [0.6, 0.2, 0.1, 0.1]
    values, log = interrogate("molab", fake, "A")
    assert values["primary_origin"] == "lung-SCLC"
    assert values["origin_confidence"] == pytest.approx(0.6 * 0.5)
    assert "primary_origin" in log


def test_origin_confidence_none_without_probabilities():
    values, _ = interrogate("molab", lambda p: ("A.", None), "A")
    assert values["origin_confidence"] is None
