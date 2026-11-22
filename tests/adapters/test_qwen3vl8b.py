import numpy as np
import pytest

from common import runner
from common.datasets import Scan
from common.schema import full_prompt
from common.volume import pick_slices, to01
from models.healthgptpro27b import run as healthgpt
from models.qwen3vl8b.run import N_SLICES, Qwen3VLAdapter

FENCED = '```json\n{"diagnosis_name": "tumor", "modality": "MRI"}\n```'


def _scan(dataset="molab"):
    return Scan(dataset, "p1", "p1.nii.gz", ("p1.nii.gz",))


def _volume():
    return np.random.default_rng(0).random((40, 64, 64), dtype=np.float32)


def _run(monkeypatch, output, dataset="molab"):
    adapter = Qwen3VLAdapter()
    calls = []

    def fake_generate(frames, text):
        calls.append((frames, text))
        return output

    monkeypatch.setattr(adapter, "_generate", fake_generate)
    values, details = adapter.predict(_scan(dataset), [_volume()])
    return values, details, calls


def test_same_frames_as_healthgpt(monkeypatch):
    assert (N_SLICES, 448) == (healthgpt.N_SLICES, healthgpt.IMAGE_SIZE) == (10, 448)
    _, details, calls = _run(monkeypatch, FENCED)
    frames, text = calls[0]
    assert len(frames) == 10 and all(f.mode == "RGB" and f.size == (448, 448) for f in frames)
    assert details["slices"] == pick_slices(to01(_volume(), "molab"), 10)
    assert text.startswith("The images are axial slices") and text.endswith(full_prompt("molab"))


def test_parses_json(monkeypatch):
    values, details, _ = _run(monkeypatch, FENCED)
    assert values["diagnosis_name"] == "tumor" and details["raw"] == FENCED


def test_failed_parse_is_an_error_with_the_raw_text(monkeypatch):
    with pytest.raises(runner.PredictionError) as e:
        _run(monkeypatch, "I cannot tell.")
    assert e.value.details["raw"] == "I cannot tell." and e.value.details["problems"] == ["no_json"]
    with pytest.raises(runner.PredictionError):
        _run(monkeypatch, '{"diagnosis_name": "<tumor|normal>"}')
