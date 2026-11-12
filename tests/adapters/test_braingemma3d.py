import nibabel as nib
import numpy as np

from common.datasets import Scan
from common.parse import extract_from_report
from common.volume import load_canonical, resize, to01
from models.braingemma3d.run import BrainGemma3DAdapter, to_model_axes

NORMAL_REPORT = "No acute intracranial abnormality."


def _scan(dataset="molab"):
    return Scan(dataset, "p1", "p1.nii.gz", ["p1.nii.gz"])


def _volume():
    return np.random.default_rng(0).random((40, 64, 64), dtype=np.float32)


def _run(monkeypatch, output, dataset="molab", volume=None):
    adapter = BrainGemma3DAdapter()
    calls = []

    def fake_generate(volume, prompt):
        calls.append((volume, prompt))
        return output

    monkeypatch.setattr(adapter, "_generate", fake_generate)
    values, details = adapter.predict(_scan(dataset), [_volume() if volume is None else volume])
    return values, details, calls


def test_volume_is_32_128_128_in_unit_range(monkeypatch):
    _, details, calls = _run(monkeypatch, NORMAL_REPORT)
    volume, _ = calls[0]
    assert volume.shape == (32, 128, 128)
    assert volume.dtype == np.float32
    assert volume.min() >= 0.0 and volume.max() <= 1.0
    assert volume.flags["C_CONTIGUOUS"]  # torch.from_numpy rejects negative strides
    assert details["input_shape"] == [32, 128, 128]


def test_prompt_is_exact(monkeypatch):
    _, _, calls = _run(monkeypatch, NORMAL_REPORT)
    _, prompt = calls[0]
    # CANONICAL_PROMPT of braingemma3d_architecture.py, trailing " \n" included (the training prompt)
    assert prompt == "Generate a radiology report for this brain MRI FLAIR scan. \n"


def test_report_goes_through_shared_extractor(monkeypatch):
    values, details, _ = _run(monkeypatch, NORMAL_REPORT, dataset="ms")
    assert values["diagnosis_name"] == "normal"
    assert values == extract_from_report(NORMAL_REPORT, "ms")
    assert details["raw_report"] == NORMAL_REPORT


def test_volume_is_in_model_loader_axes(monkeypatch, tmp_path):
    # A BraTS-like LPS affine, so the reorientation to RAS is not the identity.
    data = np.random.default_rng(1).random((6, 5, 4), dtype=np.float32)
    path = str(tmp_path / "lps.nii.gz")
    nib.save(nib.Nifti1Image(data, np.diag([-1.0, -1.0, 1.0, 1.0])), path)
    # What braingemma3d_architecture.load_nifti_volume does to a BraTS file before scaling and resizing.
    expected = np.transpose(nib.as_closest_canonical(nib.load(path)).get_fdata(dtype=np.float32), (2, 1, 0))
    ours = load_canonical(path)
    np.testing.assert_array_equal(to_model_axes(ours), expected)

    _, _, calls = _run(monkeypatch, NORMAL_REPORT, volume=ours)
    np.testing.assert_array_equal(calls[0][0], to_model_axes(resize(to01(ours, "molab"), (32, 128, 128))))
