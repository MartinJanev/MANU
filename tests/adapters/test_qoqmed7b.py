import numpy as np

import models._qwen25vl as qwen25vl
from common.datasets import Scan
from common.schema import full_prompt
from models.qoqmed7b.run import QoQMedAdapter

FENCED = '```json\n{"diagnosis_name": "tumor", "modality": "MRI"}\n```'
# QoQ-Med's trained answer shape (examples/format_prompt/medical_format.jinja in its repo):
# a ```json block of "bbox_2d" boxes, then <think>...</think>, then \boxed{answer}.
BBOX = '```json\n[{"bbox_2d": [10, 20, 200, 220], "label": "lesion"}]\n```\n'
THINK = '<think>A {ring-enhancing} lesion; not {"diagnosis_name": "healthy"}.</think>\n'
ANSWER = '{"diagnosis_name": "tumor", "modality": "MRI"}'


def _scan(dataset="molab"):
    return Scan(dataset, "p1", "p1.nii.gz", ["p1.nii.gz"])


def _volume():
    return np.random.default_rng(0).random((40, 64, 64), dtype=np.float32)


def _run(monkeypatch, output, dataset="molab"):
    adapter = QoQMedAdapter()
    calls = []

    def fake_generate(images, text):
        calls.append((images, text))
        return output

    monkeypatch.setattr(adapter, "_generate", fake_generate)
    values, details = adapter.predict(_scan(dataset), [_volume()])
    return values, details, calls


def test_inputs_are_4_rgb_448(monkeypatch):
    _, details, calls = _run(monkeypatch, FENCED)
    images, text = calls[0]
    assert len(images) == 4
    assert all(im.mode == "RGB" and im.size == (448, 448) for im in images)
    assert text.endswith(full_prompt("molab"))
    assert details["prompt"] == text
    assert len(details["slices"]) == 4


def test_parses_fenced_json(monkeypatch):
    values, details, _ = _run(monkeypatch, FENCED)
    assert values["diagnosis_name"] == "tumor"
    assert details["raw"] == FENCED
    assert details["parsed_from"] == "whole"


def test_boxed_answer_after_think_is_parsed_not_the_bbox(monkeypatch):
    raw = BBOX + THINK + "\\boxed{" + ANSWER + "}"
    values, details, _ = _run(monkeypatch, raw)
    assert values == {"diagnosis_name": "tumor", "modality": "MRI"}
    assert details["parsed_from"] == "boxed"
    assert details["problems"] == []
    assert details["raw"] == raw


def test_plain_json_after_think_is_parsed(monkeypatch):
    values, details, _ = _run(monkeypatch, BBOX + THINK + ANSWER)
    assert values == {"diagnosis_name": "tumor", "modality": "MRI"}
    assert details["parsed_from"] == "after_think"


def test_unclosed_think_is_empty_with_problem(monkeypatch):
    values, details, _ = _run(monkeypatch, BBOX + '<think>Maybe {"diagnosis_name": "tumor"}, but the')
    assert values == {}
    assert details["problems"] == ["truncated_think"]
    assert details["parsed_from"] is None


def test_generates_with_2048_new_tokens(monkeypatch):
    captured = {}

    def fake_generate_images(model, processor, images, text, max_new_tokens):
        captured.update(model=model, processor=processor, text=text, max_new_tokens=max_new_tokens)
        return "answer"

    monkeypatch.setattr(qwen25vl, "generate_images", fake_generate_images)
    adapter = QoQMedAdapter()
    adapter.model, adapter.processor = "model", "processor"
    assert adapter._generate([], "question") == "answer"
    assert captured == {"model": "model", "processor": "processor", "text": "question", "max_new_tokens": 2048}
