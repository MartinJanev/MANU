import re
from pathlib import Path

import numpy as np
import pytest

from common.datasets import Scan
from common.qna import BANK, ORIGINAL, REVERSED, format_mcq
from common.volume import resize, to01
from models.radfm_qna.run import (MAX_NEW_TOKENS, REPORT_MAX_NEW_TOKENS, REPORT_QUESTION, RadFMQnAAdapter,
                                  add_image_tokens, image_prefix, model_volume, option_probabilities,
                                  shown_options)

REPO = Path(__file__).resolve().parents[2]


def _scan(dataset="molab"):
    return Scan(dataset, "p1", "p1.nii.gz", ["p1.nii.gz"])


def _volume():
    return np.random.default_rng(0).random((40, 64, 64), dtype=np.float32)


def _questions():
    for q in BANK.values():
        yield from (q if isinstance(q, list) else [q])


def _run(monkeypatch, dataset="molab", answer=None, probs=(0.7, 0.1, 0.1, 0.1), report="No abnormality."):
    """answer=None: the fake model answers the first displayed option."""
    adapter = RadFMQnAAdapter()
    calls = []

    def fake_generate(volume, question, options, max_new_tokens):
        calls.append((volume, question, options, max_new_tokens))
        if options is None:
            return report, None
        return (options[0] if answer is None else answer), list(probs)

    monkeypatch.setattr(adapter, "_generate", fake_generate)
    values, details = adapter.predict(_scan(dataset), [_volume()])
    return values, details, calls


@pytest.mark.parametrize("dataset", ["molab", "ct_rate"])
def test_input_is_h_w_d_in_unit_range(monkeypatch, dataset):
    _, details, calls = _run(monkeypatch, dataset)
    volume = calls[0][0]
    assert volume.shape == (256, 256, 64)  # RadFM's [H, W, D]
    assert volume.dtype == np.float32 and volume.flags["C_CONTIGUOUS"]
    assert volume.min() >= 0.0 and volume.max() <= 1.0
    # the canonical stack, scaled by to01 and resized; only depth moved last (no flip)
    np.testing.assert_array_equal(volume, resize(to01(_volume(), dataset), (64, 256, 256)).transpose(1, 2, 0))
    assert all(c[0] is volume for c in calls)  # prepared once per scan
    assert details["input_shape"] == [256, 256, 64]


def test_blank_volume_stays_blank():
    out = model_volume(np.zeros((40, 64, 64), np.float32), "ixi")
    assert out.shape == (256, 256, 64) and not out.any()


def test_questions_are_the_shared_bank_in_m3d_style(monkeypatch):
    _, _, calls = _run(monkeypatch)
    mcq = [c for c in calls if c[2] is not None]
    assert mcq[0][1] == format_mcq(BANK["modality"], "A", ORIGINAL)
    assert mcq[0][2] == ["MRI", "CT", "X-ray", "Ultrasound"]
    assert mcq[1][2] == ["Ultrasound", "X-ray", "CT", "MRI"]  # the reversed ask
    assert all(c[3] == MAX_NEW_TOKENS for c in mcq)
    assert all("Choices: A." in c[1] for c in mcq)


def test_one_report_per_scan_in_details(monkeypatch):
    values, details, calls = _run(monkeypatch, report="Multiple enhancing lesions.")
    reports = [c for c in calls if c[2] is None]
    assert len(reports) == 1
    assert reports[0][1] == REPORT_QUESTION and reports[0][3] == REPORT_MAX_NEW_TOKENS
    assert details["report"] == "Multiple enhancing lesions."
    assert "report" not in values  # read only; values come from the questions


def test_interrogate_wired(monkeypatch):
    values, details, _ = _run(monkeypatch, dataset="ms")
    assert values["modality"] == "MRI" and values["plane"] == "axial"
    assert values["diagnosis_name"] == "normal"
    assert details["modality"]["answers"][0]["raw"] == "MRI"
    assert details["modality"]["confidence"] == pytest.approx(0.7)
    assert details["modality"]["agree"] is False  # reversed ask answered its first option, "Ultrasound"
    assert {"ms_type", "ms_lesion_location", "ms_lesion_burden"} <= set(details)


def test_free_text_answer_parsed_by_option_text(monkeypatch):
    values, _, _ = _run(monkeypatch, dataset="ixi", answer="Coronal")
    assert values["plane"] == "coronal"
    assert values["modality"] is None  # "Coronal" is no modality option


def test_shown_options_inverts_format_mcq_for_every_question():
    for q in _questions():
        for order in (ORIGINAL, REVERSED):
            assert shown_options(format_mcq(q, "A", order)) == [q.options[i] for i in order]
    with pytest.raises(ValueError):
        shown_options("What is the modality? (a) MRI (b) CT (c) X-ray (d) Ultrasound")


def test_option_probabilities():
    probs = option_probabilities([-1.0, -2.0, -3.0, -4.0])
    e = np.exp([-1.0, -2.0, -3.0, -4.0])
    np.testing.assert_allclose(probs, e / e.sum())
    assert option_probabilities([-1.0, -np.inf, -1.0, -np.inf]) == pytest.approx([0.5, 0.0, 0.5, 0.0])
    assert option_probabilities([-np.inf] * 4) is None


def test_image_prefix():
    one = image_prefix(1)
    assert one.startswith("<image><image0><image1>") and one.endswith("<image31></image>")
    assert len(re.findall(r"<image\d+>", one)) == 32
    assert image_prefix(2).count("<image32>") == 1


class _FakeTokenizer:
    def __init__(self):
        self.added = []

    def add_special_tokens(self, special):
        self.added.append(list(special["additional_special_tokens"]))


def test_add_image_tokens_as_quick_demo():
    tok = add_image_tokens(_FakeTokenizer())
    assert len(tok.added) == 100  # called once per image, with the growing list
    assert tok.added[-1][:2] == ["<image>", "</image>"]
    assert len(tok.added[-1]) == 2 + 100 * 32 and tok.added[-1][-1] == "<image3199>"
    assert (tok.pad_token_id, tok.bos_token_id, tok.eos_token_id) == (0, 1, 2)


def test_requirements_match_radfm_dockerfile():
    docker = (REPO / "containers" / "radfm.Dockerfile").read_text()
    pins = {l.split("==")[0]: l for l in re.findall(r"([A-Za-z0-9_.-]+==[\w.]+)", docker)}
    reqs = [l.strip() for l in (REPO / "models" / "radfm_qna" / "requirements.txt").read_text().splitlines()
            if l.strip() and not l.startswith("#")]
    assert reqs and all(pins.get(l.split("==")[0]) == l for l in reqs)
    assert "SIF=containers/radfm.sif" in (REPO / "models" / "radfm_qna" / "env.sh").read_text()
