import nibabel as nib
import numpy as np

from common.volume import (
    ct_to01, foreground_range, load_canonical, normalize01, pick_slices,
    resize, to01, to_pseudo_hu, to_rgb_images,
)


def save(data, affine, tmp_path, name="v.nii.gz"):
    path = str(tmp_path / name)
    nib.save(nib.Nifti1Image(data, affine), path)
    return path


def test_canonical_axes(tmp_path):
    data = np.zeros((4, 5, 6), np.float32)
    data[0, 4, 5] = 1  # most left, most anterior, most superior
    out = load_canonical(save(data, np.eye(4), tmp_path))
    assert out.dtype == np.float32
    assert out.shape == (6, 5, 4) and out[5, 0, 3] == 1


def test_sagittal_affine_gives_axial_stack(tmp_path):
    ras = np.random.default_rng(0).random((4, 5, 6)).astype(np.float32)
    sag = ras.transpose(1, 2, 0)  # sag[i, j, k] = ras[k, i, j]
    affine = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float)
    a = load_canonical(save(sag, affine, tmp_path, "sag.nii.gz"))
    b = load_canonical(save(ras, np.eye(4), tmp_path, "ras.nii.gz"))
    np.testing.assert_array_equal(a, b)


def test_nan_and_4d(tmp_path):
    data = np.ones((3, 4, 5, 1), np.float32)
    data[0, 0, 0, 0] = np.nan
    out = load_canonical(save(data, np.eye(4), tmp_path))
    assert out.shape == (5, 4, 3) and not np.isnan(out).any()
    assert out.sum() == 3 * 4 * 5 - 1


def test_normalize01_range():
    v = np.random.default_rng(1).normal(size=(10, 10, 10))
    out = normalize01(v)
    assert out.min() == 0 and out.max() == 1
    assert (normalize01(np.full((4, 4, 4), 7.0)) == 0).all()


def test_resize_exact_shape():
    vol = np.random.default_rng(2).random((10, 20, 30)).astype(np.float32)
    assert resize(vol, (32, 64, 64)).shape == (32, 64, 64)


def test_foreground_range_empty_gives_whole_volume():
    assert foreground_range(np.zeros((12, 4, 4), np.float32)) == (0, 11)


def test_pick_slices_inside_foreground():
    vol = np.zeros((40, 8, 8), np.float32)
    vol[10:30] = 1.0
    idx = pick_slices(vol, 4)
    assert len(idx) == 4 and all(10 <= i <= 29 for i in idx) and idx == sorted(set(idx))


def test_pick_slices_tight_range_stays_distinct():
    vol = np.zeros((40, 8, 8), np.float32)
    vol[10:14] = 1.0  # exactly 4 foreground slices, 4 wanted
    assert pick_slices(vol, 4) == [10, 11, 12, 13]


def test_to_rgb_images():
    vol = np.random.default_rng(3).random((5, 16, 16)).astype(np.float32)
    imgs = to_rgb_images(vol, [1, 3], 448)
    assert len(imgs) == 2
    assert all(im.mode == "RGB" and im.size == (448, 448) for im in imgs)


def test_hu_mappings():
    assert to_pseudo_hu(np.array([0.0, 1.0])).tolist() == [-1000.0, 1000.0]
    assert ct_to01(np.array([-2000.0, 0.0, 3000.0])).tolist() == [0.0, 0.5, 1.0]


def test_to01_dispatch():
    v = np.array([-2000.0, 0.0, 3000.0])
    assert to01(v, "ct_rate").tolist() == [0.0, 0.5, 1.0]
    assert to01(v, "ixi").tolist() == normalize01(v).tolist()
