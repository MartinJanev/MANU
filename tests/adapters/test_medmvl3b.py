import json
import os
import re
import sys
import textwrap

import numpy as np
import pytest

from common.datasets import Scan
from common.qna import BANK, ORIGINAL, format_mcq
from common.volume import resize, to01, to_pseudo_hu
from models.medmvl3b.run import (MedMVLAdapter, letter_probabilities, letter_token_ids, local_checkpoint, model_input,
                                 model_volume, preload_formatters)


def _scan(dataset="molab"):
    return Scan(dataset, "p1", "p1.nii.gz", ["p1.nii.gz"])


def _volume():
    return np.random.default_rng(0).random((40, 64, 64), dtype=np.float32)


def _run(monkeypatch, answer="A. MRI", probs=None, dataset="molab", volume=None):
    adapter = MedMVLAdapter()
    calls = []

    def fake_generate(volume, prompt):
        calls.append((volume, prompt))
        return (answer(prompt) if callable(answer) else answer), probs

    monkeypatch.setattr(adapter, "_generate", fake_generate)
    values, details = adapter.predict(_scan(dataset), [_volume() if volume is None else volume])
    return values, details, calls


@pytest.mark.parametrize("dataset", ["molab", "ixi", "ms"])
def test_pseudo_hu_range(monkeypatch, dataset):
    _, _, calls = _run(monkeypatch, dataset=dataset)
    volume, _ = calls[0]
    assert volume.shape == (32, 256, 256)
    assert volume.dtype == np.float32
    assert volume.min() >= -1000 and volume.max() <= 1000
    # percentile scaling, resize, then [0, 1] -> [-1000, 1000]; the canonical axial stack, no flip or axis change
    np.testing.assert_array_equal(volume, to_pseudo_hu(resize(to01(_volume(), dataset), (32, 256, 256))))
    assert all(v is volume for v, _ in calls)  # prepared once per scan


def test_ct_rate_gets_clipped_hounsfield_units(monkeypatch):
    hu = np.random.default_rng(0).uniform(-3000, 3000, (40, 64, 64)).astype(np.float32)
    _, _, calls = _run(monkeypatch, dataset="ct_rate", volume=hu)
    volume, _ = calls[0]
    assert volume.shape == (32, 256, 256)
    assert volume.dtype == np.float32
    assert volume.min() >= -1000 and volume.max() <= 1000
    # the real HU, clipped to [-1000, 1000] and resized: not rescaled to pseudo-HU
    np.testing.assert_allclose(volume, resize(np.clip(hu, -1000, 1000), (32, 256, 256)), atol=0.01)
    np.testing.assert_array_equal(model_input(hu, "ct_rate"), volume)


def test_prompt_style(monkeypatch):
    _, _, calls = _run(monkeypatch)
    assert len(calls) > 2
    for _, prompt in calls:
        assert prompt.startswith("<image3d>\n")
        assert "Choices: A." in prompt and " D. " in prompt
        assert "(a) " not in prompt
    # M3D's close-ended format, the one MedM-VL-CT-3B-en was fine-tuned in
    assert calls[0][1] == "<image3d>\n" + format_mcq(BANK["modality"], "A", ORIGINAL)
    assert calls[0][1] == ("<image3d>\nWhat is the modality of this image? Choices: A. MRI B. CT C. X-ray "
                           "D. Ultrasound")


def test_parses_choice_letter(monkeypatch):
    def answer(prompt):
        return "C. Coronal" if "Which plane" in prompt else "A. MRI"

    values, details, _ = _run(monkeypatch, answer=answer, probs=[0.1, 0.2, 0.6, 0.1])
    assert values["plane"] == "coronal"
    assert details["plane"]["method"] == "letter"
    assert details["plane"]["confidence"] == pytest.approx(0.6)
    assert values["modality"] == "MRI"


def test_letter_probabilities_renormalised_over_the_four_letters():
    logits = np.full(10, -5.0, dtype=np.float32)
    logits[[3, 4, 5, 6]] = [2.0, 1.0, 0.0, 0.0]
    logits[9] = 50.0  # a non-letter token with most of the mass does not count
    probs = letter_probabilities(logits, [3, 4, 5, 6])
    e = np.exp([2.0, 1.0, 0.0, 0.0])
    np.testing.assert_allclose(probs, e / e.sum(), rtol=1e-6)
    assert sum(probs) == pytest.approx(1.0)
    assert letter_probabilities(np.full(10, -np.inf), [3, 4, 5, 6]) is None


class _BpeLikeTokenizer:
    """Llama-3-style pieces: "ASSISTANT", ":" and the space-prefixed letter " C" are separate tokens, one id each."""

    def __init__(self, merge=None, split=None):
        self.vocab = {}
        self.merge = merge  # this text becomes one token
        self.split = split  # this letter becomes " " + letter

    def __call__(self, text, add_special_tokens=True):
        pieces = [text] if text == self.merge else re.findall(r" ?\w+|[^\w\s]", text)
        if self.split is not None and pieces[-1] == " " + self.split:
            pieces[-1:] = [" ", self.split]
        ids = [self.vocab.setdefault(p, len(self.vocab) + 10) for p in pieces]
        return type("Encoding", (), {"input_ids": ids})()


def test_letter_token_ids_are_the_first_answer_token_after_assistant():
    tok = _BpeLikeTokenizer()
    ids = letter_token_ids(tok)
    assert ids == [tok.vocab[" A"], tok.vocab[" B"], tok.vocab[" C"], tok.vocab[" D"]]
    with pytest.raises(ValueError):  # a letter merged with the prefix is not one token after "ASSISTANT:"
        letter_token_ids(_BpeLikeTokenizer(merge="ASSISTANT: B"))
    with pytest.raises(ValueError):  # nor is a letter that takes two tokens
        letter_token_ids(_BpeLikeTokenizer(split="C"))


def test_local_checkpoint_rewrites_cache_dir_and_attention(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "model-00001-of-00002.safetensors").write_text("weights")
    (snapshot / "tokenizer.json").write_text("{}")
    config = {"cache_dir_hf": "/mnt/nfs_share/shiym/ckpts/cache_dir_hf", "llm_type": "llama3",
              "llm_name_or_path": "meta-llama/Llama-3.2-3B-Instruct", "llm_attn_implementation": "flash_attention_2",
              "llm_config": {"rope_scaling": {"factor": 32.0, "rope_type": "llama3", "type": "linear"}}}
    (snapshot / "config.json").write_text(json.dumps(config))

    out = local_checkpoint(str(snapshot), str(tmp_path / "sdpa"), "sdpa")
    written = json.loads((tmp_path / "sdpa" / "config.json").read_text())
    assert written == dict(config, cache_dir_hf=None, llm_attn_implementation="sdpa")
    assert not os.path.islink(os.path.join(out, "config.json"))
    for entry in ("model-00001-of-00002.safetensors", "tokenizer.json"):
        assert os.path.realpath(os.path.join(out, entry)) == os.path.realpath(snapshot / entry)
    assert json.loads((snapshot / "config.json").read_text()) == config  # snapshot untouched

    (tmp_path / "flash").mkdir()  # load() passes the empty folder tempfile.mkdtemp made
    local_checkpoint(str(snapshot), str(tmp_path / "flash"), None)
    assert json.loads((tmp_path / "flash" / "config.json").read_text()) == dict(config, cache_dir_hf=None)


def test_blank_volume_skips_the_foreground_crop():
    calls = []

    def processor(volume, mode):
        calls.append((volume, mode))
        return "cropped and resized"

    blank = np.full((4, 8, 8), -1000.0, dtype=np.float32)  # the runner's blank volume after model_input
    out = model_volume(blank, processor)
    assert calls == []
    assert out.shape == (1, 4, 8, 8) and out.dtype == np.float32 and not out.any()  # the processor's min-max of it
    scan = blank.copy()
    scan[1, 2, 3] = 0.0
    assert model_volume(scan, processor) == "cropped and resized"
    assert calls[0][0] is scan and calls[0][1] == "eval"


# The pattern of MedM-VL's lvlm/dataset/template/: formatter instances as dataclass field defaults.
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

    from lvlm.dataset.template.formatter import EmptyFormatter, Formatter, StringFormatter


    @dataclass
    class Llama3Template:
        system: "Formatter" = EmptyFormatter(slot="SYSTEM ")
        format_user: "Formatter" = StringFormatter(slot="USER: {{content}} ")
"""


@pytest.fixture
def template_repo(tmp_path, monkeypatch):
    package = tmp_path / "lvlm" / "dataset" / "template"  # lvlm/ and lvlm/dataset/ have no __init__.py
    package.mkdir(parents=True)
    (package / "formatter.py").write_text(textwrap.dedent(FORMATTER))
    (package / "llama_template.py").write_text(textwrap.dedent(TEMPLATE))
    (package / "__init__.py").write_text("from lvlm.dataset.template.llama_template import Llama3Template\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield str(tmp_path)
    for name in [m for m in sys.modules if m == "lvlm" or m.startswith("lvlm.")]:
        del sys.modules[name]


def test_preload_formatters_lets_the_templates_import(template_repo):
    if sys.version_info >= (3, 11):  # the failure the image's Python 3.12 would show
        with pytest.raises(ValueError, match="mutable default"):
            import lvlm.dataset.template  # noqa: F401
        for name in [m for m in sys.modules if m == "lvlm" or m.startswith("lvlm.")]:
            del sys.modules[name]
    preload_formatters(template_repo)
    from lvlm.dataset.template import Llama3Template

    template = Llama3Template()
    assert template.system.apply() == "SYSTEM "
    assert template.format_user.apply(content="What?") == "USER: What? "
    assert Llama3Template().format_user is template.format_user  # one shared default, as on Python 3.10
