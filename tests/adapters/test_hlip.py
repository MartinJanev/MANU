import nibabel as nib
import numpy as np
import pytest

from common.datasets import Scan
from common.schema import normalize_values
from common.volume import load_canonical, to01
from models.hlip.run import CLASS_GROUPS, HLIPAdapter, preprocess, prompt, to_model_axes


def _scan(dataset="molab"):
    return Scan(dataset, "p1", "p1.nii.gz", ["p1.nii.gz"])


def _volume():
    return np.random.default_rng(0).random((40, 64, 64), dtype=np.float32)


def _probabilities():
    """Normal is the single most likely class, but the tumor classes together are the most likely group."""
    groups = {"normal": 0.30, "tumor": 0.45, "stroke": 0.20, "multiple sclerosis": 0.05}
    probs = {}
    for group, total in groups.items():
        members = [c for c, g in CLASS_GROUPS.items() if g == group]
        probs.update({c: total / len(members) for c in members})
    return probs


def _run(monkeypatch, probabilities, dataset="molab"):
    adapter = HLIPAdapter()
    calls = []

    def fake_score(volume):
        calls.append(volume)
        return probabilities

    monkeypatch.setattr(adapter, "_score", fake_score)
    values, details = adapter.predict(_scan(dataset), [_volume()])
    return values, details, calls


def test_preprocess_shape():
    out = preprocess(to01(_volume(), "molab"))
    assert out.shape == (48, 224, 224)
    assert out.dtype == np.float32
    assert out.flags["C_CONTIGUOUS"]  # torch.from_numpy rejects negative strides


def test_predict_scores_the_preprocessed_volume(monkeypatch):
    _, details, calls = _run(monkeypatch, _probabilities(), dataset="ct_rate")
    np.testing.assert_array_equal(calls[0], preprocess(to01(_volume(), "ct_rate")))
    assert details["input_shape"] == [48, 224, 224]


def test_groups_have_equal_weight_whatever_their_number_of_classes(monkeypatch):
    probs = _probabilities()  # tumor's 0.45 is spread over 8 classes, normal's 0.30 is one class
    values, details, _ = _run(monkeypatch, probs)
    means = {"normal": 0.30, "tumor": 0.45 / 8, "stroke": 0.20 / 5, "multiple sclerosis": 0.05}
    total = sum(means.values())
    assert values["diagnosis_name"] == "normal"  # a sum over classes would say tumor
    assert values["diagnosis_confidence"] == pytest.approx(0.30 / total)
    assert details["group_probabilities"] == pytest.approx({g: m / total for g, m in means.items()})
    assert details["class_probabilities"] == probs


def test_an_uninformative_model_does_not_favour_tumor(monkeypatch):
    _, details, _ = _run(monkeypatch, {c: 1 / len(CLASS_GROUPS) for c in CLASS_GROUPS})
    assert list(details["group_probabilities"].values()) == pytest.approx([0.25] * 4)


def test_other_fields_null(monkeypatch):
    for dataset in ("ixi", "molab", "ms", "ct_rate"):
        values, _, _ = _run(monkeypatch, _probabilities(), dataset=dataset)
        assert set(values) == {"diagnosis_name", "diagnosis_confidence"}
        normalized = normalize_values(dataset, values)
        assert {k for k, v in normalized.items() if v is not None} == {"diagnosis_name", "diagnosis_confidence"}


def test_classes_and_prompts():
    assert set(CLASS_GROUPS.values()) == {"normal", "tumor", "stroke", "multiple sclerosis"}
    assert CLASS_GROUPS["no significant abnormalities"] == "normal"
    for c in ("metastasis", "glioma", "meningioma", "Metastatic malignant neoplasm to brain"):
        assert CLASS_GROUPS[c] == "tumor"
    assert CLASS_GROUPS["acute stroke"] == "stroke"
    assert CLASS_GROUPS["Demyelinating disease of central nervous system"] == "multiple sclerosis"
    assert prompt("glioma") == "This study shows: glioma identified."
    # zeroshot_metadata_pubbrain5.TEMPLATES[1] words the normal class without "identified"
    assert prompt("no significant abnormalities") == "This study shows: no significant abnormalities."


def test_volume_is_in_hlip_loader_axes(tmp_path):
    # An LPS affine, so the reorientation is not the identity.
    data = np.random.default_rng(1).random((6, 6, 4), dtype=np.float32)
    path = str(tmp_path / "lps.nii.gz")
    nib.save(nib.Nifti1Image(data, np.diag([-1.0, -1.0, 1.0, 1.0])), path)
    # load_nifti_file of HLIP's data/pub_brain_5/*/process.py (tag v1.0): SimpleITK DICOMOrient to 'RPI', then
    # GetArrayFromImage = (z, y, x) with x -> patient right, y -> posterior, z -> inferior. The same array via
    # nibabel's RAS stack (checked against SimpleITK 2.5.7 on six affines):
    ras = nib.as_closest_canonical(nib.load(path)).get_fdata(dtype=np.float32)
    expected = np.transpose(ras[:, ::-1, ::-1], (2, 1, 0))
    np.testing.assert_array_equal(to_model_axes(load_canonical(path)), expected)


def test_depth_runs_head_to_feet():
    # Our depth axis runs feet -> head; HLIP's scans (and the example studies in its docs/) start at the vertex.
    ramp = np.broadcast_to(np.linspace(0.0, 1.0, 40, dtype=np.float32)[:, None, None], (40, 64, 64))
    out = preprocess(ramp)
    assert out[0, 112, 112] > out[-1, 112, 112]


# Reference values from torch 2.6.0 (CPU) running the steps of PubBrain5Dataset.__getitem__ in
# src/hlip_test/zeroshot_pubbrain5.py (pad to square, bilinear 256x256, nearest-exact (48, 256, 256),
# crop 16:240, Normalize) on vol[::-1, :, ::-1], vol = default_rng(1).random(shape, dtype=float32).
TORCH_REFERENCE = {
    (30, 40, 50): {"slice_means": [0.000676, 0.034335, 0.07861],
                   "voxels": [0.251366, -0.15396, -0.559287, -0.964613]},
    (100, 300, 260): {"slice_means": [0.201334, 0.205054, 0.211567],
                      "voxels": [0.439294, 0.328625, 0.5171, 1.29457]},
}


@pytest.mark.parametrize("shape", sorted(TORCH_REFERENCE))
def test_preprocess_matches_torch_steps(shape):
    ref = TORCH_REFERENCE[shape]
    out = preprocess(np.random.default_rng(1).random(shape, dtype=np.float32))
    np.testing.assert_allclose(out.mean(axis=(1, 2))[[0, 13, 47]], ref["slice_means"], atol=1e-5)
    np.testing.assert_allclose(out[24, 100:104, 57], ref["voxels"], atol=1e-5)
