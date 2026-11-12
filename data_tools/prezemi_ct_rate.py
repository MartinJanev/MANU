#!/usr/bin/env python3
"""Download the few CT-RATE chest-CT volumes used as a control (gated dataset, 21.3 TB in total).

Only the first scan of each split patient (valid_N_a_1.nii.gz) and three small CSV files are fetched, by
their known paths (no repository listing): dataset/valid_fixed/... first, dataset/valid/... if that file
does not exist. Needs only huggingface_hub (imported lazily; works with 0.16.4) and a token in HF_TOKEN from
an account that accepted the dataset terms.

  export HF_TOKEN=...
  python3 data_tools/prezemi_ct_rate.py [--split dev|eval|all] [--data data] [--splits splits.json]
"""
import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

REPO_ID = "ibrahimhamamci/CT-RATE"
REPO_ROOT = Path(__file__).resolve().parent.parent
LABELS = "dataset/multi_abnormality_labels/valid_predicted_labels.csv"
EXTRA_FILES = (LABELS,
               "dataset/radiology_text_reports/validation_reports.csv",
               "dataset/metadata/validation_metadata.csv")
PARENTS = ("dataset/valid_fixed", "dataset/valid")


def select_files(repo_files, patients):
    """patient -> repo path of its valid_N_a_1.nii.gz, preferring a valid_fixed path when both exist."""
    chosen = {}
    for patient in patients:
        pattern = re.compile(rf"^dataset/valid[^/]*/{patient}/{patient}_a/{patient}_a_1\.nii\.gz$")
        hits = sorted(f for f in repo_files if pattern.match(f))
        fixed = [f for f in hits if f.startswith("dataset/valid_fixed/")]
        if not hits:
            raise ValueError(f"no {patient}_a_1.nii.gz in the repository listing (patient {patient} missing)")
        chosen[patient] = (fixed or hits)[0]
    return chosen


def split_patients(splits_path, split):
    with open(splits_path, encoding="utf-8") as f:
        splits = json.load(f)
    names = ("dev", "eval") if split == "all" else (split,)
    return [p for name in names for p in splits[name]["ct_rate"]]


def _class_names(exc):
    return {c.__name__ for c in type(exc).__mro__}


def _status(exc):
    return getattr(getattr(exc, "response", None), "status_code", None)


def _is_not_found(exc):
    """True when the requested file does not exist in the repository (EntryNotFoundError or HTTP 404).
    LocalEntryNotFoundError (huggingface_hub's 'connection down and not cached') and a repository that is
    not found are not a missing file."""
    names = _class_names(exc)
    if names & {"LocalEntryNotFoundError", "RepositoryNotFoundError"}:
        return False
    return "EntryNotFoundError" in names or _status(exc) == 404


def _is_access_error(exc):
    return _status(exc) in (401, 403) or bool(_class_names(exc) & {"GatedRepoError", "RepositoryNotFoundError"})


def fetch(repo_path, local_dir, token, download=None):
    """Download one repo file to local_dir/<repo_path> as a real file (no symlink into the HF cache)."""
    if download is None:
        from huggingface_hub import hf_hub_download as download  # lazy: the module imports without it
    kwargs = {"repo_id": REPO_ID, "repo_type": "dataset", "filename": repo_path, "local_dir": str(local_dir),
              "token": token}
    try:
        try:
            return download(**kwargs, local_dir_use_symlinks=False)  # 0.16.4 symlinks big files otherwise
        except TypeError as e:
            if "local_dir_use_symlinks" not in str(e):
                raise
            return download(**kwargs)  # newer huggingface_hub without that keyword: always real files
    except Exception as e:
        if _is_access_error(e):
            print(f"Access to {REPO_ID} denied: accept the dataset terms on https://huggingface.co/datasets/"
                  f"{REPO_ID} with the account whose token is in HF_TOKEN, and check that the token is valid.",
                  file=sys.stderr, flush=True)
        raise


def download(patients, data_dir, token, hf_download=None):
    """Download each patient's <P>_a_1.nii.gz (valid_fixed, else valid) and the three CSV files, keeping repo
    paths under data_dir/ct_rate. No repository listing: the paths are known. Returns patient -> repo path;
    raises ValueError naming every patient found in neither folder (after trying all of them)."""
    local_dir = Path(data_dir) / "ct_rate"
    selected, missing = {}, []
    for patient in patients:
        for parent in PARENTS:
            repo_path = f"{parent}/{patient}/{patient}_a/{patient}_a_1.nii.gz"
            print(f"downloading {repo_path}", flush=True)
            try:
                fetch(repo_path, local_dir, token, hf_download)
            except Exception as e:  # noqa: BLE001 - only "file not found" moves on to the next folder
                if not _is_not_found(e):
                    raise
                print(f"  not in the repository: {repo_path}", flush=True)
                continue
            selected[patient] = repo_path
            break
        else:
            missing.append(patient)
    for repo_path in EXTRA_FILES:
        print(f"downloading {repo_path}", flush=True)
        fetch(repo_path, local_dir, token, hf_download)
    if missing:
        raise ValueError(f"no <P>_a_1.nii.gz under {' or '.join(PARENTS)} for patient(s) {', '.join(missing)}: "
                         "replace them in splits.json")
    return selected


def positive_labels(labels_csv, volume):
    with open(labels_csv, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["VolumeName"] == volume:
                return sum(1 for k, v in row.items() if k != "VolumeName" and str(v).strip() in ("1", "1.0"))
    return None


def summarize(selected, data_dir):
    try:
        import nibabel as nib
    except ImportError:
        nib = None
    root = Path(data_dir) / "ct_rate"
    for patient, repo_path in selected.items():
        positives = positive_labels(root / LABELS, Path(repo_path).name) if (root / LABELS).exists() else None
        line = f"{patient}: {root / repo_path}  positive labels: {positives}"
        if nib is not None:
            vol = nib.load(str(root / repo_path)).get_fdata()
            line += f"  shape {vol.shape}  min/max {vol.min():.0f}/{vol.max():.0f}"
        print(line)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", choices=("dev", "eval", "all"), default="all")
    p.add_argument("--data", default=str(REPO_ROOT / "data"))
    p.add_argument("--splits", default=str(REPO_ROOT / "splits.json"))
    args = p.parse_args(argv)
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN is not set: accept the CT-RATE terms on Hugging Face and export your access token.")
    patients = split_patients(args.splits, args.split)
    selected = download(patients, args.data, token)
    summarize(selected, args.data)


if __name__ == "__main__":
    main()
