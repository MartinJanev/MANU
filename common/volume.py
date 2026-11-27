"""Volume preprocessing shared by every model adapter: one canonical axial array.

Only numpy, nibabel, scipy and Pillow are imported (no torch).
"""
import nibabel as nib
import numpy as np
from PIL import Image
from scipy import ndimage


def load_canonical(path):
    """NIfTI file -> float32 (D, H, W), slices with square pixels and a square frame.

    D = axial slices feet->head, H = rows front->back, W = columns patient right->left
    (the way a radiologist views an axial slice), whatever orientation the file stores.
    In-plane only: resampled to the finer of the two spacings, then padded with the volume minimum (air). Every
    adapter resizes a slice to a square, so without this a brain is stretched by an amount that differs per dataset
    (1.35x on IXI). The depth axis is left alone: each model has its own slice count.
    """
    # RAS first, so sagittal (IXI) and axial files end up with the same layout.
    img = nib.as_closest_canonical(nib.load(path))
    vol = np.nan_to_num(np.asarray(img.dataobj, dtype=np.float32))
    if vol.ndim == 4:  # take the first volume of a 4D file
        vol = vol[..., 0]
    # Same as load_volume of the earlier RadFM runner (git show 4a2317c:models/radfm/run.py): (x, y, z) -> (rows, cols, depth) ...
    vol = vol[::-1, ::-1, :].transpose(1, 0, 2)
    # ... then depth moved to axis 0.
    dx, dy = img.header.get_zooms()[:2]
    vol = _square_pixels(vol.transpose(2, 0, 1), float(dy), float(dx))
    return np.ascontiguousarray(vol, dtype=np.float32)


def _square_pixels(vol, dy, dx):
    """(D, H, W) with row spacing dy and column spacing dx -> square pixels, padded to H == W."""
    if dy > 0 and dx > 0 and abs(dy - dx) > 0.01 * min(dy, dx):
        fine = min(dy, dx)
        vol = ndimage.zoom(vol, (1, dy / fine, dx / fine), order=1)
    h, w = vol.shape[1:]
    if h == w:
        return vol
    size = max(h, w)
    top, left = (size - h) // 2, (size - w) // 2
    return np.pad(vol, ((0, 0), (top, size - h - top), (left, size - w - left)), constant_values=vol.min())


def normalize01(vol, lo=0.5, hi=99.5):
    """Clip to the [lo, hi] percentiles and scale to [0, 1]; a constant volume gives zeros."""
    vol = np.asarray(vol, dtype=np.float32)
    p_lo, p_hi = np.percentile(vol, (lo, hi))
    if p_hi - p_lo < 1e-6:
        return np.zeros_like(vol)
    return np.clip((vol - p_lo) / (p_hi - p_lo), 0.0, 1.0)


def resize(vol, shape):
    """Trilinear resize to exactly `shape`."""
    vol = np.asarray(vol, dtype=np.float32)
    shape = tuple(int(s) for s in shape)
    out = ndimage.zoom(vol, [t / s for t, s in zip(shape, vol.shape)], order=1)
    if out.shape != shape:  # zoom rounds its output size; pad/crop the odd voxel
        fixed = np.zeros(shape, dtype=out.dtype)
        sl = tuple(slice(0, min(a, b)) for a, b in zip(out.shape, shape))
        fixed[sl] = out[sl]
        out = fixed
    return out


def foreground_range(vol01, thr=0.1, min_frac=0.05):
    """First and last slice where at least `min_frac` of the voxels exceed `thr`.

    The whole range if no slice qualifies.
    """
    frac = (np.asarray(vol01) > thr).reshape(len(vol01), -1).mean(axis=1)
    hits = np.flatnonzero(frac >= min_frac)
    if len(hits) == 0:
        return 0, len(vol01) - 1
    return int(hits[0]), int(hits[-1])


def pick_slices(vol01, n):
    """n slice indices spread evenly inside the foreground range (ends excluded).

    Sorted and distinct whenever the range holds at least n slices.
    """
    a, b = foreground_range(vol01)
    idx = np.round(np.linspace(a, b, n + 2)[1:-1]).astype(int)
    if b - a + 1 >= n:  # rounding can collide on a tight range: force strictly increasing
        for i in range(1, n):
            idx[i] = max(idx[i], idx[i - 1] + 1)
        idx[-1] = min(idx[-1], b)
        for i in range(n - 2, -1, -1):
            idx[i] = min(idx[i], idx[i + 1] - 1)
    return [int(i) for i in idx]


def to_rgb_images(vol01, indices, size):
    """Axial slices at `indices` -> RGB uint8 PIL images of size x size."""
    images = []
    for i in indices:
        grey = (np.clip(vol01[i], 0.0, 1.0) * 255).round().astype(np.uint8)
        img = Image.fromarray(grey, mode="L").resize((size, size), Image.BILINEAR)
        images.append(img.convert("RGB"))
    return images


def to_pseudo_hu(vol01):
    """[0, 1] -> [-1000, 1000], for models that expect CT Hounsfield units."""
    return np.asarray(vol01) * 2000 - 1000


def ct_to01(vol_hu):
    """CT window: clip to [-1000, 1000] HU and scale to [0, 1]."""
    return (np.clip(vol_hu, -1000.0, 1000.0) + 1000.0) / 2000.0


def to01(vol, dataset):
    """CT window for CT-RATE, percentile scaling for every MRI dataset."""
    return ct_to01(vol) if dataset == "ct_rate" else normalize01(vol)
