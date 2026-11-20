import base64
import io

import numpy as np
import pytest
from PIL import Image

from common import runner
from common.datasets import Scan
from common.schema import full_prompt
from models.medgemma15.run import (CT_WINDOWS, MAX_SLICE, MedGemma15Adapter, build_messages, encode,
                                   sample_slices, split_prompt, to_rgb_slices)

ANSWER = '```json\n{"modality": "MRI", "diagnosis_name": "Brain tumor", "diagnosis_confidence": 0.8}\n```'


def _scan(dataset="ixi"):
    return Scan(dataset, "p1", "p1.nii.gz", ("p1.nii.gz",))


def _volume(depth=120):
    return np.random.default_rng(0).random((depth, 40, 32), dtype=np.float32) * 900 + 100


def _run(monkeypatch, output, dataset="ixi", volume=None):
    adapter = MedGemma15Adapter()
    calls = []

    def fake_generate(messages):
        calls.append(messages)
        return output

    monkeypatch.setattr(adapter, "_generate", fake_generate)
    values, details = adapter.predict(_scan(dataset), [_volume() if volume is None else volume])
    return values, details, calls


def _decode(uri):
    assert uri.startswith("data:image/jpeg;base64,")
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1]))))


def test_sampling_matches_the_notebook():
    assert sample_slices(40) == list(range(40))
    assert sample_slices(85) == list(range(85))
    idx = sample_slices(300)
    assert len(idx) == MAX_SLICE
    assert idx == [int(round(i / 85 * 299)) for i in range(1, 86)]
    assert idx[-1] == 299 and idx == sorted(set(idx))


def test_mri_is_one_min_max_over_the_volume_grey_in_rgb():
    vol = _volume()
    rgb = to_rgb_slices(vol, "ixi", [0, 60])
    assert rgb.shape == (2, 40, 32, 3) and rgb.dtype == np.uint8
    assert (rgb[..., 0] == rgb[..., 1]).all() and (rgb[..., 1] == rgb[..., 2]).all()
    expected = np.round((vol[[0, 60]] - vol.min()) / (vol.max() - vol.min()) * 255).astype(np.uint8)
    np.testing.assert_array_equal(rgb[..., 0], expected)


def test_blank_mri_gives_black_slices():
    rgb = to_rgb_slices(np.zeros((10, 8, 8), np.float32), "ms", [0, 5])
    assert (rgb == 0).all()


def test_ct_uses_the_three_notebook_windows():
    hu = np.array([[[-2000, -1024, 0, 40, 80, 215, 1024, 3000]]], dtype=np.float32)
    rgb = to_rgb_slices(hu, "ct_rate", [0])[0, 0]
    for channel, (lo, hi) in enumerate(CT_WINDOWS):
        expected = np.round((np.clip(hu[0, 0], lo, hi) - lo) / (hi - lo) * 255).astype(np.uint8)
        np.testing.assert_array_equal(rgb[:, channel], expected)
    assert CT_WINDOWS == [(-1024, 1024), (-135, 215), (0, 80)]


def test_encode_is_a_jpeg_of_the_slice():
    data = np.full((16, 16, 3), 128, np.uint8)
    assert np.abs(_decode(encode(data)).astype(int) - 128).max() <= 2


def test_prompt_split_keeps_every_word_of_the_prompt():
    for dataset in ("ixi", "molab", "ms", "ct_rate"):
        instruction, query = split_prompt(dataset, 85)
        assert instruction.startswith("You are an expert neuroradiologist")
        assert instruction.endswith("The images are 85 axial slices of the same 3D scan, "
                                    "ordered from inferior to superior.")
        assert query.startswith("\n\nOutput ONLY this JSON schema below")
        assert instruction.replace("The images are 85 axial slices of the same 3D scan, ordered from inferior "
                                   "to superior.", "") + query[2:] == full_prompt(dataset)


def test_one_user_message_instruction_slices_schema(monkeypatch):
    _, details, calls = _run(monkeypatch, ANSWER)
    (messages,) = calls
    assert [m["role"] for m in messages] == ["user"]  # no system role
    content = messages[0]["content"]
    assert content[0] == {"type": "text", "text": details["prompt"]["instruction"]}
    assert content[-1] == {"type": "text", "text": details["prompt"]["query"]}
    middle = content[1:-1]
    assert len(middle) == 2 * MAX_SLICE
    assert [c["type"] for c in middle[:4]] == ["image", "text", "image", "text"]
    assert [c["text"] for c in middle[1::2]] == [f"SLICE {n}" for n in range(1, MAX_SLICE + 1)]
    assert _decode(middle[0]["image"]).shape == (40, 32, 3)
    assert details["slices"] == sample_slices(120)


def test_build_messages_numbers_slices_from_one():
    msgs = build_messages(["a", "b"], "I", "Q")
    assert [c.get("text", c.get("image")) for c in msgs[0]["content"]] == ["I", "a", "SLICE 1", "b", "SLICE 2", "Q"]


def test_labels_are_normalised(monkeypatch):
    values, details, _ = _run(monkeypatch, ANSWER)
    assert values["diagnosis_name"] == "tumor"
    assert details["diagnosis_name_raw"] == "Brain tumor"
    values, _, _ = _run(monkeypatch, '{"diagnosis_name": "Normal", "modality": "MRI"}')
    assert values["diagnosis_name"] == "normal"
    assert details["raw"] == ANSWER


def test_no_json_raises_with_the_raw_text(monkeypatch):
    with pytest.raises(runner.PredictionError) as e:
        _run(monkeypatch, "I cannot tell.")
    assert "no_json" in str(e.value)
    assert e.value.details["raw"] == "I cannot tell."


def test_json_without_valid_values_raises(monkeypatch):
    with pytest.raises(runner.PredictionError):
        _run(monkeypatch, '{"diagnosis_name": "<tumor|normal>", "modality": "PET"}')


def test_load_reads_the_local_snapshot_not_the_hub_id(monkeypatch):
    # transformers 4.57.3 asks the Hub about a tokenizer given by repo id (_patch_mistral_regex -> model_info),
    # which fails in an offline job; a local snapshot path skips that call.
    import sys
    import types

    seen = []
    fake_model = types.SimpleNamespace(eval=lambda: None)
    transformers = types.SimpleNamespace(
        AutoProcessor=types.SimpleNamespace(from_pretrained=lambda path, **kw: seen.append(("processor", path))),
        AutoModelForImageTextToText=types.SimpleNamespace(
            from_pretrained=lambda path, **kw: seen.append(("model", path)) or fake_model))
    hub = types.SimpleNamespace(snapshot_download=lambda repo_id: f"/hf_cache/hub/snap/{repo_id}")
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(bfloat16="bf16"))
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    MedGemma15Adapter().load()
    local = "/hf_cache/hub/snap/google/medgemma-1.5-4b-it"
    assert seen == [("processor", local), ("model", local)]
