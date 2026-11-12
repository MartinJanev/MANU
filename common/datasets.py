"""Which scans belong to a split, and the ground truth that goes with each one.

Standard library only: this module must import without numpy, nibabel or torch.
"""
import glob
import json
import os
import re
from dataclasses import dataclass

DATASETS = {
    "ixi": {"glob": "ixi/T1/*.nii.gz", "tag": "healthy", "experiment": "healthy"},
    "molab": {"glob": "molab/nifti/*_img.nii*", "tag": "brain_metastases", "experiment": "brain_metastases"},
    "ms": {
        "glob": "open_ms_data/cross_sectional/raw/patient*/T1W.nii.gz",
        "tag": "ms",
        "experiment": "multiple_sclerosis",
    },
}


@dataclass(frozen=True)
class Scan:
    dataset: str
    patient: str
    image_path: str  # the colleagues' spelling, used as the record key
    files: tuple  # absolute paths fed to the model


def load_splits(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def molab_name(name):
    """figshare '10005_19010618_eTRA-3D-T1Gd_img.nii' -> '010005N__010005_19010618_eTRA-3D-T1Gd_img.nii'."""
    if re.match(r"\d+N__", name):  # already in the colleague spelling
        return name
    patient, rest = name.split("_", 1)
    patient = patient.zfill(6)
    return f"{patient}N__{patient}_{rest}"


def patient_id(dataset, image_path):
    """'IXI012-HH-1211-T1.nii.gz' -> 'IXI012'; '010024N__...' -> '010024N'; 'patient10_T1.nii.gz' -> 'patient10'."""
    if dataset == "ixi":
        return image_path.split("-", 1)[0]
    if dataset == "molab":
        return image_path.split("__", 1)[0]
    if dataset == "ms":
        return image_path.split("_", 1)[0]
    raise KeyError(dataset)


def _scan_from_file(data_dir, dataset, path, ms_flair):
    if dataset == "ixi":
        name = os.path.basename(path)
        return Scan(dataset, patient_id(dataset, name), name, (path,))
    if dataset == "molab":
        name = molab_name(os.path.basename(path))
        return Scan(dataset, patient_id(dataset, name), name, (path,))
    folder = os.path.dirname(path)  # ms: one patient folder
    patient = os.path.basename(folder)
    image = os.path.join(folder, "FLAIR.nii.gz" if ms_flair else "T1W.nii.gz")
    return Scan(dataset, patient, f"{patient}_T1.nii.gz", (image,))


def find_scans(data_dir, dataset, split, splits, ms_flair=False):
    """All scans of a split (split="full": every scan), sorted by image_path.

    Raises ValueError naming every split patient that has no file, so a model is never
    silently evaluated on fewer patients than intended."""
    pattern = os.path.join(data_dir, DATASETS[dataset]["glob"])
    scans = [_scan_from_file(data_dir, dataset, p, ms_flair) for p in sorted(glob.glob(pattern))]
    if split != "full":
        wanted = list(splits[split][dataset])
        found = {s.patient for s in scans}
        missing = [p for p in wanted if p not in found]
        if missing:
            raise ValueError(f"{dataset}: no scan files for {split} patients {missing} under {pattern}")
        scans = [s for s in scans if s.patient in set(wanted)]
    return sorted(scans, key=lambda s: s.image_path)


_REFERENCE_CACHE = {}


def load_reference(reference_dir, dataset):
    """image_path -> metadata, from the colleague output file for this dataset (cached)."""
    key = (os.path.abspath(reference_dir), dataset)
    if key not in _REFERENCE_CACHE:
        path = os.path.join(reference_dir, f"radfm_{DATASETS[dataset]['tag']}_processed.jsonl")
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        _REFERENCE_CACHE[key] = {r["input"]["image_path"]: r["input"]["metadata"] for r in rows}
    return _REFERENCE_CACHE[key]


def metadata_for(image_path, reference):
    """Ground-truth metadata for one scan. MOLAB labels are per patient (first 6 digits),
    so a scan the colleagues did not have takes the metadata of the same patient."""
    if image_path in reference:
        return reference[image_path]
    for name, meta in reference.items():
        if re.match(r"\d{6}N__", image_path) and name[:6] == image_path[:6]:
            return meta
    print(f"WARNING: no reference metadata for {image_path}; labels set to null", flush=True)
    meta = dict(next(iter(reference.values())))  # constant fields (dataset, class, modality ...)
    for key in ("primary_origin", "ms_type"):
        if key in meta:
            meta[key] = None
    return meta


def ground_truth(scan, reference_dir, data_dir):
    """Metadata for the record's input.metadata. data_dir is unused for these datasets
    (CT-RATE reads its labels from data_dir)."""
    return dict(metadata_for(scan.image_path, load_reference(reference_dir, scan.dataset)))
