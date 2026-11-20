import sys
import types

import numpy as np

from common.datasets import Scan
from common.schema import full_prompt, slice_preface
from models.healthgptpro27b.run import HealthGPTProAdapter

# When the chat template's generation prompt ends with an open "<think>\n" (Qwen3.6 thinking mode), the decoded
# output starts inside the reasoning and closes it with </think> before the answer.
REASONING = 'Axial T1 MRI. A draft: {"diagnosis_name": "healthy"}; but a {ring-enhancing} lesion is seen.\n</think>\n\n'
FENCED = '```json\n{"diagnosis_name": "tumor", "modality": "MRI"}\n```'


def _scan(dataset="molab"):
    return Scan(dataset, "p1", "p1.nii.gz", ["p1.nii.gz"])


def _volume():
    return np.random.default_rng(0).random((40, 64, 64), dtype=np.float32)


def _run(monkeypatch, output, dataset="molab", opened=True):
    adapter = HealthGPTProAdapter()
    calls = []

    def fake_generate(frames, text, max_new_tokens):
        calls.append((frames, text, max_new_tokens))
        return output, opened

    monkeypatch.setattr(adapter, "_generate", fake_generate)
    values, details = adapter.predict(_scan(dataset), [_volume()])
    return values, details, calls


class _Inputs(dict):
    """Stands in for the processor's BatchFeature: input_ids attribute and .to(device)."""

    def to(self, device):
        self["moved_to"] = device
        return self

    @property
    def input_ids(self):
        return self["input_ids"]


def _run_generate(monkeypatch, decoded, template="<prompt><think>\n"):
    """predict() through the real _generate, with the processor, model and qwen_vl_utils replaced by recorders."""
    log = {}

    def process_vision_info(messages, **kwargs):
        log["vision"] = (messages, kwargs)
        return None, [("video-tensor", {"fps": 2.0, "frames_indices": list(range(10))})], {"do_sample_frames": False}

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            log["template"] = (messages, kwargs)
            return template

        def __call__(self, **kwargs):
            log["processor"] = kwargs
            return _Inputs(input_ids=[[1, 2, 3]])

        def batch_decode(self, sequences, **kwargs):
            log["decode"] = (sequences, kwargs)
            return [decoded]

    class Model:
        device = "cuda:0"

        def generate(self, **kwargs):
            log["generate"] = kwargs
            return [[1, 2, 3, 7, 8]]

    monkeypatch.setitem(sys.modules, "qwen_vl_utils", types.SimpleNamespace(process_vision_info=process_vision_info))
    adapter = HealthGPTProAdapter()
    adapter.model, adapter.processor = Model(), Processor()
    values, details = adapter.predict(_scan("ms"), [_volume()])
    return values, details, log


def test_ten_frames_as_one_video(monkeypatch):
    _, details, log = _run_generate(monkeypatch, REASONING + FENCED)
    messages, template_kwargs = log["template"]
    assert len(messages) == 1 and messages[0]["role"] == "user"
    content = messages[0]["content"]
    videos = [c for c in content if c["type"] == "video"]
    assert len(videos) == 1
    assert [c["type"] for c in content] == ["video", "text"]  # the card's 3D example: video first, then text
    frames = videos[0]["video"]
    assert len(frames) == 10
    assert all(im.mode == "RGB" and im.size == (448, 448) for im in frames)
    assert videos[0]["sample_fps"] == 2.0
    assert content[1]["text"] == slice_preface(10) + full_prompt("ms") == details["prompt"]
    assert template_kwargs == {"tokenize": False, "add_generation_prompt": True}
    assert log["vision"][0] is messages
    assert len(details["slices"]) == 10


def test_card_video_pipeline_and_greedy_2048(monkeypatch):
    values, details, log = _run_generate(monkeypatch, REASONING + FENCED)
    assert log["vision"][1] == {"image_patch_size": 16, "return_video_kwargs": True, "return_video_metadata": True}
    assert log["processor"] == {
        "text": "<prompt><think>\n",
        "images": None,
        "videos": ["video-tensor"],
        "video_metadata": [{"fps": 2.0, "frames_indices": list(range(10))}],
        "return_tensors": "pt",
        "do_resize": False,
        "do_sample_frames": False,
    }
    generate_kwargs = log["generate"]
    assert generate_kwargs["moved_to"] == "cuda:0"
    assert generate_kwargs["input_ids"] == [[1, 2, 3]]
    assert (generate_kwargs["max_new_tokens"], generate_kwargs["do_sample"], generate_kwargs["num_beams"]) == (
        2048, False, 1)
    assert log["decode"] == ([[7, 8]], {"skip_special_tokens": True, "clean_up_tokenization_spaces": False})
    assert details["raw"] == REASONING + FENCED
    assert values == {"diagnosis_name": "tumor", "modality": "MRI"}


def test_parses_fenced_json(monkeypatch):
    values, details, calls = _run(monkeypatch, REASONING + FENCED)
    assert values == {"diagnosis_name": "tumor", "modality": "MRI"}  # not the draft inside the reasoning
    assert details["problems"] == []
    assert details["raw"] == REASONING + FENCED
    frames, text, max_new_tokens = calls[0]
    assert len(frames) == 10 and max_new_tokens == 2048
    assert text == slice_preface(10) + full_prompt("molab")


def test_reopened_think_is_skipped(monkeypatch):
    raw = "\n</think>\n\n<think>More thought.</think>\n" + FENCED
    values, _, _ = _run(monkeypatch, raw)
    assert values == {"diagnosis_name": "tumor", "modality": "MRI"}


def test_unclosed_think_is_empty_with_problem(monkeypatch):
    values, details, _ = _run(monkeypatch, 'Axial T1. A draft {"diagnosis_name": "tumor"}, but the')
    assert values == {}
    assert details["problems"] == ["truncated_think"]


def test_answer_without_json_is_empty_with_problem(monkeypatch):
    values, details, _ = _run(monkeypatch, "Looked at it.\n</think>\n\nI cannot tell.")
    assert values == {}
    assert details["problems"] == ["no_json"]


def test_prompt_without_think_parses_the_whole_answer(monkeypatch):
    # job 2762: the template did not open <think>, the model answered with plain JSON
    values, details, _ = _run_generate(monkeypatch, '{"diagnosis_name": "tumor", "modality": "MRI"}', template="<prompt>")
    assert values == {"diagnosis_name": "tumor", "modality": "MRI"}
    assert details["problems"] == [] and details["think_opened"] is False


def test_open_think_in_the_prompt_is_reported(monkeypatch):
    _, details, _ = _run_generate(monkeypatch, REASONING + FENCED)
    assert details["think_opened"] is True
