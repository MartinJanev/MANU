"""Which scans belong to a split, and the ground truth that goes with each one.

Standard library only: this module must import without numpy, nibabel or torch.
"""
import csv
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
    "ct_rate": {
        "glob": "ct_rate/dataset/valid*/valid_*/valid_*_a/valid_*_a_1.nii.gz",
        "tag": "ct_rate",
        "experiment": "ct_rate",
    },
}

MS_PATIENT_INFO = "open_ms_data/cross_sectional/patient_info.csv"
MS_TYPES = {"RR": "SMRR", "SP": "SMSP", "PR": "SMPP"}  # CIS and N/A are outside the schema: not scored
CT_RATE_LABELS = "ct_rate/dataset/multi_abnormality_labels/valid_predicted_labels.csv"


@dataclass(frozen=True)
class Scan:
    dataset: str
    patient: str
    image_path: str  # the colleagues' spelling, used as the record key
    files: tuple  # absolute paths fed to the model
    prior: bool = False  # the prior-knowledge condition: the prompt states the diagnosis (runner --task prior)


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
    if dataset == "ct_rate":
        return re.match(r"(valid_\d+)_", image_path).group(1)
    raise KeyError(dataset)


def _scan_from_file(dataset, path, ms_flair):
    if dataset == "ms":  # one patient folder
        folder = os.path.dirname(path)
        patient = os.path.basename(folder)
        image = os.path.join(folder, "FLAIR.nii.gz" if ms_flair else "T1W.nii.gz")
        return Scan(dataset, patient, f"{patient}_T1.nii.gz", (image,))
    name = os.path.basename(path)
    if dataset == "molab":
        name = molab_name(name)
    return Scan(dataset, patient_id(dataset, name), name, (path,))


def find_scans(data_dir, dataset, split, splits, ms_flair=False):
    """All scans of a split (split="full": every scan), sorted by image_path.

    Raises ValueError naming every split patient that has no file, so a model is never
    silently evaluated on fewer patients than intended."""
    pattern = os.path.join(data_dir, DATASETS[dataset]["glob"])
    scans = [_scan_from_file(dataset, p, ms_flair) for p in sorted(glob.glob(pattern))]
    if dataset == "ct_rate":  # the same volume may sit under dataset/valid and dataset/valid_fixed
        by_name = {}
        for s in scans:
            if s.image_path not in by_name or "valid_fixed" in s.files[0]:
                by_name[s.image_path] = s
        scans = list(by_name.values())
    if split != "full":
        wanted = list(splits[split][dataset])
        found = {s.patient for s in scans}
        missing = [p for p in wanted if p not in found]
        if missing:
            raise ValueError(f"{dataset}: no scan files for {split} patients {missing} under {pattern}")
        scans = [s for s in scans if s.patient in set(wanted)]
    scans = sorted(scans, key=lambda s: s.image_path)
    if split == "paper":  # one scan per patient: the first by date (the file names sort by it)
        first = {}
        for s in scans:
            first.setdefault(s.patient, s)
        scans = list(first.values())
    return scans


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
    if re.match(r"\d{6}N__", image_path):
        for name, meta in reference.items():
            if name[:6] == image_path[:6]:
                return meta
    print(f"WARNING: no reference metadata for {image_path}; labels set to null", flush=True)
    meta = dict(next(iter(reference.values())))  # constant fields (dataset, class, modality ...)
    for key in ("primary_origin", "ms_type"):
        if key in meta:
            meta[key] = None
    return meta


_LABEL_CACHE = {}
_WARNED = set()


def _warn_once(message):
    if message not in _WARNED:
        _WARNED.add(message)
        print(f"WARNING: {message}", flush=True)


def load_ct_rate_labels(data_dir):
    """VolumeName -> [abnormality columns equal to 1]; None when the labels CSV is missing (cached)."""
    path = os.path.join(data_dir, CT_RATE_LABELS)
    key = os.path.abspath(path)
    if key not in _LABEL_CACHE:
        labels = None
        if os.path.isfile(path):
            labels = {}
            with open(path, encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = row.pop("VolumeName")
                    labels[name] = [c for c, v in row.items() if str(v).strip() in ("1", "1.0")]
        else:
            _warn_once(f"CT-RATE labels CSV not found: {path}; abnormalities set to null")
        _LABEL_CACHE[key] = labels
    return _LABEL_CACHE[key]


def _ct_rate_truth(scan, data_dir):
    labels = load_ct_rate_labels(data_dir)
    abnormalities = None if labels is None else labels.get(scan.image_path)
    if labels is not None and abnormalities is None:
        _warn_once(f"CT-RATE labels CSV has no row for {scan.image_path}; abnormalities set to null")
    return {"dataset": "ct_rate", "class": None, "modality": "CT", "axial_plane": "axial",
            "modality_subtype": None, "icd10_code": None, "abnormalities": abnormalities}


def _ms_type(patient, data_dir):
    path = os.path.join(data_dir, MS_PATIENT_INFO)
    if not os.path.isfile(path):
        _warn_once(f"MS patient info not found: {path}; ms_type set to null")
        return None
    with open(path, encoding="utf-8-sig", newline="") as f:
        types = {int(r["patient_id"]): r["ms_type"] for r in csv.DictReader(f) if r["patient_id"].strip().isdigit()}
    return MS_TYPES.get(types.get(int(re.search(r"\d+", patient).group())))


def ground_truth(scan, reference_dir, data_dir):
    """Metadata for the record's input.metadata. IXI/MOLAB/MS come from the colleagues' reference files, except
    ms_type: theirs is MSLesSeg's clinical table joined on the patient number (other patients), so it is read from
    open_ms_data's own patient_info.csv. CT-RATE reads its abnormality labels from data_dir."""
    if scan.dataset == "ct_rate":
        return _ct_rate_truth(scan, data_dir)
    meta = dict(metadata_for(scan.image_path, load_reference(reference_dir, scan.dataset)))
    if scan.dataset == "ms":
        meta["ms_type"] = _ms_type(scan.patient, data_dir)
    return meta
