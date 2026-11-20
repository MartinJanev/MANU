import nibabel as nib
import numpy as np
import pytest

from common.dicom import native_from_canonical, write_dicom_series
from common.volume import load_canonical

sitk = pytest.importorskip("SimpleITK")
pytest.importorskip("pydicom")

AFFINES = {
    "ras": np.diag([0.9, 1.1, 1.3, 1.0]),
    "lps_sagittal": np.array([[0, 0, -1.2, 80], [-0.94, 0, 0, 120], [0, 0.94, 0, -90], [0, 0, 0, 1.0]]),
    "oblique": np.array([[0.98, -0.17, 0.0, -100], [0.17, 0.98, 0.0, -110], [0.0, 0.0, 2.0, -40], [0, 0, 0, 1.0]]),
    "flipped_z": np.diag([1.0, 1.0, -1.5, 1.0]),
}


def _nifti(tmp_path, affine, shape=(12, 10, 7), seed=0):
    data = (np.random.default_rng(seed).random(shape) * 1000 - 200).astype(np.float32)
    path = str(tmp_path / "vol.nii.gz")
    nib.save(nib.Nifti1Image(data, affine), path)
    return path, data


def _read_series(folder):
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(sorted(str(p) for p in folder.glob("*.dcm")))  # name order, as Prima's natsort
    return reader.Execute()


@pytest.mark.parametrize("name", list(AFFINES))
def test_series_has_the_nifti_geometry_and_values(tmp_path, name):
    path, data = _nifti(tmp_path, AFFINES[name])
    write_dicom_series(data, AFFINES[name], str(tmp_path / "s1"), "T1", 1, "study")
    # Compared in one orientation (LPS, as Prima reads it): a left-handed NIfTI is stored last slice first.
    dcm = sitk.DICOMOrient(_read_series(tmp_path / "s1"), "LPS")
    ref = sitk.DICOMOrient(sitk.ReadImage(path), "LPS")
    assert dcm.GetSize() == ref.GetSize()
    np.testing.assert_allclose(dcm.GetSpacing(), ref.GetSpacing(), atol=1e-6)
    np.testing.assert_allclose(dcm.GetOrigin(), ref.GetOrigin(), atol=1e-4)
    np.testing.assert_allclose(dcm.GetDirection(), ref.GetDirection(), atol=1e-6)
    a, b = sitk.GetArrayFromImage(dcm), sitk.GetArrayFromImage(ref)
    assert np.abs(a - b).max() <= (data.max() - data.min()) / 65535 + 1e-3
    # every voxel at the same physical point
    for index in [(0, 0, 0), (11, 9, 6), (3, 7, 2)]:
        np.testing.assert_allclose(dcm.TransformIndexToPhysicalPoint(index),
                                   ref.TransformIndexToPhysicalPoint(index), atol=1e-4)


def test_same_physical_layout_after_prima_reorientation(tmp_path):
    # Prima's reader: sitk.DICOMOrient(image, 'LPS'); the NIfTI read the same way must give the same array.
    path, data = _nifti(tmp_path, AFFINES["lps_sagittal"])
    write_dicom_series(data, AFFINES["lps_sagittal"], str(tmp_path / "s1"), "T1", 1, "study")
    a = sitk.GetArrayFromImage(sitk.DICOMOrient(_read_series(tmp_path / "s1"), "LPS"))
    b = sitk.GetArrayFromImage(sitk.DICOMOrient(sitk.ReadImage(path), "LPS"))
    assert np.abs(a - b).max() < 0.05


def test_native_from_canonical_round_trip(tmp_path):
    for affine in AFFINES.values():
        path, data = _nifti(tmp_path, affine)
        np.testing.assert_array_equal(native_from_canonical(load_canonical(path), nib.load(path)), data)


def test_blank_volume_and_tags(tmp_path):
    import pydicom

    write_dicom_series(np.zeros((8, 6, 3)), AFFINES["ras"], str(tmp_path / "s2"), "FLAIR", 2, "study",
                       study_description="MRI BRAIN")
    files = sorted((tmp_path / "s2").glob("*.dcm"))
    assert [f.name for f in files] == ["0001.dcm", "0002.dcm", "0003.dcm"]
    ds = pydicom.dcmread(str(files[0]))
    assert ds.SeriesDescription == ds.ProtocolName == "FLAIR" and ds.StudyDescription == "MRI BRAIN"
    assert (ds.Rows, ds.Columns) == (6, 8)
    assert (sitk.GetArrayFromImage(_read_series(tmp_path / "s2")) == 0).all()


def test_series_of_one_study_share_the_study_uid(tmp_path):
    import pydicom

    write_dicom_series(np.ones((4, 4, 2)), AFFINES["ras"], str(tmp_path / "a"), "T1", 1, "p1")
    write_dicom_series(np.ones((4, 4, 2)), AFFINES["ras"], str(tmp_path / "b"), "FLAIR", 2, "p1")
    a, b = (pydicom.dcmread(str(next((tmp_path / d).glob("*.dcm")))) for d in "ab")
    assert a.StudyInstanceUID == b.StudyInstanceUID and a.SeriesInstanceUID != b.SeriesInstanceUID
