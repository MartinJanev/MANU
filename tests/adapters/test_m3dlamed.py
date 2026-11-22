import subprocess
from pathlib import Path

import numpy as np
import pytest

from common.datasets import Scan
from common.qna import BANK, ORIGINAL, format_mcq
from models._m3dlamed import IMAGE_TOKENS, model_volume
from models.m3dlamed_llama2.run import M3DLlama2Adapter
from models.m3dlamed_phi3.run import M3DPhi3Adapter

REPO = Path(__file__).resolve().parent.parent.parent


def test_volume_is_the_authors_shape_after_cropping_the_empty_border():
    vol = np.zeros((40, 64, 64), np.float32)
    vol[10:30, 16:48, 16:48] = 1.0
    out = model_volume(vol)
    assert out.shape == (1, 32, 256, 256) and out.dtype == np.float32
    assert out.min() == 1.0  # the empty border is cropped before the resize


def test_blank_volume_is_not_cropped_to_nothing():
    out = model_volume(np.zeros((10, 16, 16), np.float32))
    assert out.shape == (1, 32, 256, 256) and not out.any()


@pytest.mark.parametrize("cls", [M3DPhi3Adapter, M3DLlama2Adapter])
def test_each_question_follows_256_image_tokens(monkeypatch, cls):
    adapter, calls = cls(), []

    def fake(volume, prompt, max_new_tokens):
        calls.append(prompt)
        return "A"

    monkeypatch.setattr(adapter, "_generate", fake)
    vol = np.random.default_rng(0).random((20, 32, 32), dtype=np.float32)
    values, details = adapter.predict(Scan("ixi", "p10", "p10.nii.gz", ("p10",)), [vol])
    assert IMAGE_TOKENS == "<im_patch>" * 256
    assert calls[0] == IMAGE_TOKENS + format_mcq(BANK["modality"], "A", ORIGINAL)
    assert values["modality"] == "MRI" and details["modality"]["answers"][0]["raw"] == "A"
    assert values["diagnosis_confidence"] is None  # never the draft's constant 0.6


@pytest.mark.parametrize("folder,version", [("m3dlamed_phi3", "4.43.3"), ("m3dlamed_llama2", "4.39.1")])
def test_each_backbone_has_its_own_transformers(folder, version):
    assert f"transformers=={version}" in (REPO / "models" / folder / "requirements.txt").read_text()


def test_setup_skips_the_duplicate_16_gb_checkpoint():
    p = subprocess.run(["bash", "jobs/setup_env.sh", "m3dlamed_phi3"], cwd=REPO, capture_output=True, text=True,
                       env={"DRY_RUN": "1", "PATH": "/usr/bin:/bin"})
    assert p.returncode == 0, p.stderr
    (download,) = [line for line in p.stdout.splitlines() if "snapshot_download" in line]
    assert "ignore_patterns" in download and "model.bin" in download
