import nibabel as nib
import numpy as np

from tools.site_baseline import features, loo_accuracy, report

Y = ["ixi"] * 5 + ["molab"] * 5 + ["ms"] * 5


def test_separable_datasets_are_found():
    rng = np.random.default_rng(0)
    X = np.concatenate([rng.normal(c, 0.1, (5, 13)) for c in (0, 5, 10)])
    acc, confusion = loo_accuracy(X, Y)
    assert acc == 1.0 and confusion["ms"]["ms"] == 5


def test_identical_datasets_give_no_signal():
    acc, _ = loo_accuracy(np.ones((15, 13)), Y)
    assert acc <= 1 / 3 + 1e-9


def test_features_and_report(tmp_path):
    vol = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
    path = tmp_path / "v.nii.gz"
    nib.save(nib.Nifti1Image(vol, np.diag([1.0, 2.0, 3.0, 1.0])), str(path))
    f = features(str(path))
    assert list(f[:6]) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] and 0 < f[6] < f[-1] <= 1
    text = report(np.tile(f, (15, 1)) + np.repeat([[0], [1], [2]], 5, axis=0), Y)
    assert "| all | 100.0% |" in text and "| ixi (n=5) | 5 | 0 | 0 |" in text
