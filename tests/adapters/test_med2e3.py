import json
import os
import re
import sys
import textwrap

import numpy as np
import pytest

from common.datasets import Scan
from common.qna import BANK, ORIGINAL, format_mcq
from common.volume import resize, to01
from models.med2e3.run import (Med2E3Adapter, letter_probabilities, letter_token_ids, local_checkpoint,
                               model_volume, preload_formatters)


def _scan(dataset="molab"):
    return Scan(dataset, "p1", "p1.nii.gz", ["p1.nii.gz"])


def _volume():
    return np.random.default_rng(0).random((40, 64, 64), dtype=np.float32)


def _run(monkeypatch, answer="A", probs=None, dataset="molab"):
    adapter = Med2E3Adapter()
    calls = []

    def fake_generate(volume, prompt):
        calls.append((volume, prompt))
        return answer, probs

    monkeypatch.setattr(adapter, "_generate", fake_generate)
    values, details = adapter.predict(_scan(dataset), [_volume()])
    return values, details, calls


@pytest.mark.parametrize("dataset", ["molab", "ct_rate"])
def test_input_shape(monkeypatch, dataset):
    _, _, calls = _run(monkeypatch, dataset=dataset)
    volume, _ = calls[0]
    assert volume.shape == (1, 32, 256, 256)
    assert volume.dtype == np.float32
    assert volume.min() >= 0.0 and volume.max() <= 1.0
    # the canonical axial stack, scaled by to01 (CT window on CT-RATE) and resized; no flip or axis change
    np.testing.assert_array_equal(volume[0], resize(to01(_volume(), dataset), (32, 256, 256)))
    assert all(v is volume for v, _ in calls)  # prepared once per scan


def test_prompt_prefix_and_style(monkeypatch):
    _, _, calls = _run(monkeypatch)
    assert len(calls) > 2
    for _, prompt in calls:
        assert prompt.startswith("<image3d>\n")
        assert "Choices: A." in prompt
    assert calls[0][1] == "<image3d>\n" + format_mcq(BANK["modality"], "A", ORIGINAL)


def test_interrogate_wired(monkeypatch):
    values, details, _ = _run(monkeypatch, answer="A", probs=[0.7, 0.1, 0.1, 0.1])
    assert values["modality"] == "MRI"
    assert values["plane"] == "axial"
    assert details["modality"]["answers"][0]["raw"] == "A"
    assert details["modality"]["confidence"] == pytest.approx(0.7)


def test_letter_probabilities_renormalised_over_the_four_letters():
    logits = np.full(10, -5.0, dtype=np.float32)
    logits[[3, 4, 5, 6]] = [2.0, 1.0, 0.0, 0.0]
    logits[9] = 50.0  # a non-letter token with most of the mass does not count
    probs = letter_probabilities(logits, [3, 4, 5, 6])
    e = np.exp([2.0, 1.0, 0.0, 0.0])
    np.testing.assert_allclose(probs, e / e.sum(), rtol=1e-6)
    assert sum(probs) == pytest.approx(1.0)
    assert letter_probabilities(np.full(10, -np.inf), [3, 4, 5, 6]) is None


class _PieceTokenizer:
    """Space-prefixed pieces, one id each (like the SentencePiece "▁A" after "ASSISTANT:")."""

    def __init__(self, merge=None):
        self.vocab = {}
        self.merge = merge

    def __call__(self, text, add_special_tokens=True):
        pieces = [text] if text == self.merge else re.findall(r" ?[^ ]+", text)
        ids = [self.vocab.setdefault(p, len(self.vocab) + 10) for p in pieces]
        return type("Encoding", (), {"input_ids": ids})()


def test_letter_token_ids_are_the_first_answer_token_after_assistant():
    tok = _PieceTokenizer()
    ids = letter_token_ids(tok)
    assert ids == [tok.vocab[" A"], tok.vocab[" B"], tok.vocab[" C"], tok.vocab[" D"]]
    with pytest.raises(ValueError):
        letter_token_ids(_PieceTokenizer(merge="ASSISTANT: B"))


def test_local_checkpoint_rewrites_cache_dir_and_attention(tmp_path):
    snapshot = tmp_path / "snapshot"
    (snapshot / "args").mkdir(parents=True)
    (snapshot / "llm").mkdir()
    (snapshot / "llm" / "config.json").write_text("{}")
    config = {"cache_dir_hf": "/mnt/nfs_share/shiym/ckpts/cache_dir_hf", "exp_id": 4,
              "llm_attn_implementation": "flash_attention_2",
              "llm_config": {"_name_or_path": "microsoft/Phi-3-mini-128k-instruct"}}
    (snapshot / "args" / "config.json").write_text(json.dumps(config))

    out = local_checkpoint(str(snapshot), str(tmp_path / "sdpa"), "sdpa")
    written = json.loads((tmp_path / "sdpa" / "args" / "config.json").read_text())
    assert written == dict(config, cache_dir_hf=None, llm_attn_implementation="sdpa")
    assert os.path.realpath(os.path.join(out, "llm")) == os.path.realpath(snapshot / "llm")
    assert json.loads((snapshot / "args" / "config.json").read_text()) == config  # snapshot untouched

    local_checkpoint(str(snapshot), str(tmp_path / "flash"), None)
    kept = json.loads((tmp_path / "flash" / "args" / "config.json").read_text())
    assert kept == dict(config, cache_dir_hf=None)


def test_blank_volume_skips_the_foreground_crop():
    calls = []

    def preprocess(volume):
        calls.append(volume)
        return "cropped and resized"

    blank = np.zeros((1, 4, 8, 8), dtype=np.float32)
    assert model_volume(blank, preprocess) is blank
    assert calls == []
    scan = blank.copy()
    scan[0, 1, 2, 3] = 0.5
    assert model_volume(scan, preprocess) == "cropped and resized"
    assert calls[0] is scan


# The pattern of Med-2E3's src/data/template/: formatter instances as dataclass field defaults.
FORMATTER = """
    from abc import ABC, abstractmethod
    from dataclasses import dataclass


    @dataclass
    class Formatter(ABC):
        slot: str = ""

        @abstractmethod
        def apply(self, **kwargs): ...


    @dataclass
    class EmptyFormatter(Formatter):
        def apply(self, **kwargs):
            return self.slot


    @dataclass
    class StringFormatter(Formatter):
        def apply(self, **kwargs):
            return self.slot.replace("{{content}}", kwargs["content"])
"""
TEMPLATE = """
    from dataclasses import dataclass

    from src.data.template.formatter import EmptyFormatter, Formatter, StringFormatter


    @dataclass
    class PhiTemplate:
        system: "Formatter" = EmptyFormatter(slot="SYSTEM ")
        format_user: "Formatter" = StringFormatter(slot="USER: {{content}} ")
"""


@pytest.fixture
def template_repo(tmp_path, monkeypatch):
    package = tmp_path / "src" / "data" / "template"
    package.mkdir(parents=True)
    (package / "formatter.py").write_text(textwrap.dedent(FORMATTER))
    (package / "phi_template.py").write_text(textwrap.dedent(TEMPLATE))
    (package / "__init__.py").write_text("from src.data.template.phi_template import PhiTemplate\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield str(tmp_path)
    for name in [m for m in sys.modules if m == "src" or m.startswith("src.")]:
        del sys.modules[name]


def test_preload_formatters_lets_the_templates_import(template_repo):
    if sys.version_info >= (3, 11):  # the failure the image's Python 3.12 would show
        with pytest.raises(ValueError, match="mutable default"):
            import src.data.template  # noqa: F401
        for name in [m for m in sys.modules if m == "src" or m.startswith("src.")]:
            del sys.modules[name]
    preload_formatters(template_repo)
    from src.data.template import PhiTemplate

    template = PhiTemplate()
    assert template.system.apply() == "SYSTEM "
    assert template.format_user.apply(content="What?") == "USER: What? "
    assert PhiTemplate().format_user is template.format_user  # one shared default, as on Python 3.10
