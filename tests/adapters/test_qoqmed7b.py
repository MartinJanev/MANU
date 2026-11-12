import numpy as np

from common.datasets import Scan
from common.schema import full_prompt
from models.qoqmed7b.run import QoQMedAdapter

FENCED = '```json\n{"diagnosis_name": "tumor", "modality": "MRI"}\n```'


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
