"""NIfTI -> DICOM series, for models that read DICOM studies (Prima).

One NIfTI volume becomes one series folder of single-frame images (MR, or CT for the CT-RATE control), slices along
the file's own third voxel axis (its acquisition plane is kept: IXI stays sagittal), with the file's geometry:
ImagePositionPatient, ImageOrientationPatient, PixelSpacing and the slice step all come from the NIfTI affine (RAS)
converted to DICOM's LPS patient coordinates. Voxel values are stored as uint16 with RescaleSlope /
RescaleIntercept, so readers that apply the rescale (SimpleITK, pydicom.pixels.apply_rescale) get the original values
back to 1/65535 of the range.

numpy, nibabel and pydicom only (pydicom is imported inside the writer, so this module imports without it).
"""
import os

import nibabel as nib
import numpy as np

SOP_CLASS = {"MR": "1.2.840.10008.5.1.4.1.1.4", "CT": "1.2.840.10008.5.1.4.1.1.2"}  # MR / CT Image Storage
LPS = np.diag([-1.0, -1.0, 1.0])  # RAS (NIfTI) -> LPS (DICOM)


def native_from_canonical(vol, img):
    """Invert common.volume.load_canonical: (D, H, W) canonical array -> the file's own voxel layout (I, J, K).

    The blank control is the canonical zeros array, so going through this keeps it a volume of the file's shape."""
    ras = np.asarray(vol).transpose(1, 2, 0)  # (rows, cols, depth) = ras[::-1, ::-1, :].transpose(1, 0, 2)
    ras = ras.transpose(1, 0, 2)[::-1, ::-1, :]
    ornt = nib.io_orientation(img.affine)  # as_closest_canonical applies this to the file's array
    back = nib.orientations.ornt_transform(nib.orientations.axcodes2ornt(("R", "A", "S")), ornt)
    return np.ascontiguousarray(nib.orientations.apply_orientation(ras, back))


def _ds(x):
    """A DICOM decimal string (at most 16 characters)."""
    text = f"{float(x):.10g}"
    return text if len(text) <= 16 else f"{float(x):.6g}"


def _uid(*parts):
    from pydicom.uid import generate_uid

    return generate_uid(entropy_srcs=[str(p) for p in parts])  # deterministic: same input, same UID


def geometry(affine):
    """(row_cosine, col_cosine, pixel_spacing [row, col], slice_step, slice_positions(k)) in LPS.

    Slice k shows data[:, :, k].T: DICOM rows follow the second voxel axis (j), columns the first (i)."""
    a = np.asarray(affine, dtype=np.float64)
    di, dj, dk = (LPS @ a[:3, c] for c in range(3))
    si, sj = np.linalg.norm(di), np.linalg.norm(dj)
    origin = LPS @ a[:3, 3]

    def position(k):
        return origin + k * dk

    return di / si, dj / sj, [sj, si], dk, position


def write_dicom_series(data, affine, out_dir, series_description, series_number, study_key,
                       study_description="", modality="MR"):
    """Write data (I, J, K) with its NIfTI affine as one DICOM series in out_dir (files 0001.dcm, ... along the
    slice normal). study_key identifies the study: series written with the same key share the Study and
    Frame-of-reference UIDs."""
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian

    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 3:
        raise ValueError(f"expected a 3D volume, got shape {data.shape}")
    os.makedirs(out_dir, exist_ok=True)
    row_cos, col_cos, spacing, step, position = geometry(affine)
    lo, hi = float(data.min()), float(data.max())
    slope = float(_ds((hi - lo) / 65535.0)) if hi > lo else 1.0
    intercept = float(_ds(lo))
    stored = np.clip(np.round((data - intercept) / slope), 0, 65535).astype("<u2")
    study_uid, frame_uid = _uid(study_key, "study"), _uid(study_key, "frame")
    series_uid = _uid(study_key, series_number, series_description)
    thickness = float(np.linalg.norm(step))
    # Readers (SimpleITK / GDCM) take the slice direction as row x column, so the files must follow that normal:
    # a left-handed affine (slice step against it) is written last slice first. Same physical volume either way.
    order = range(data.shape[2])
    if np.dot(step, np.cross(row_cos, col_cos)) < 0:
        order = order[::-1]

    for n, k in enumerate(order):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = SOP_CLASS[modality]
        meta.MediaStorageSOPInstanceUID = _uid(series_uid, n)
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        path = os.path.join(out_dir, f"{n + 1:04d}.dcm")
        ds = FileDataset(path, {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID = SOP_CLASS[modality]
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.FrameOfReferenceUID = study_uid, series_uid, frame_uid
        ds.Modality = modality
        ds.PatientID = ds.PatientName = "ANON"
        ds.StudyDescription = study_description
        ds.SeriesDescription = ds.ProtocolName = series_description
        ds.SeriesNumber, ds.InstanceNumber = int(series_number), n + 1
        ds.ImagePositionPatient = [_ds(v) for v in position(k)]
        ds.ImageOrientationPatient = [_ds(v) for v in (*row_cos, *col_cos)]
        ds.PixelSpacing = [_ds(v) for v in spacing]
        ds.SliceThickness = ds.SpacingBetweenSlices = _ds(thickness)
        ds.Rows, ds.Columns = data.shape[1], data.shape[0]
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
        ds.RescaleIntercept, ds.RescaleSlope, ds.RescaleType = _ds(intercept), _ds(slope), "US"
        ds.PixelData = np.ascontiguousarray(stored[:, :, k].T).tobytes()
        if int(pydicom.__version__.split(".")[0]) >= 3:
            ds.save_as(path, enforce_file_format=True)
        else:  # pydicom 2.x (Prima pins 2.4.4)
            ds.is_little_endian, ds.is_implicit_VR = True, False
            ds.save_as(path, write_like_original=False)
    return out_dir
