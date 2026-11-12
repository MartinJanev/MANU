#!/usr/bin/env python3
"""Download the few CT-RATE chest-CT volumes used as a control (gated dataset, 21.3 TB in total).

Only the first scan of each split patient (valid_N_a_1.nii.gz) and three small CSV files are fetched.
Needs only huggingface_hub (imported lazily) and a token in HF_TOKEN from an account that accepted the
dataset terms.

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


def list_repo(api, patients):
    """Repo file paths under the candidate folders of each patient (never the whole repository)."""
    files = []
    for patient in patients:
        for parent in PARENTS:
            path = f"{parent}/{patient}"
            try:
                if hasattr(api, "list_repo_tree"):
                    entries = api.list_repo_tree(REPO_ID, repo_type="dataset", path_in_repo=path, recursive=True)
                    files += [getattr(e, "path", None) or e.rfilename for e in entries]
                else:  # very old huggingface_hub: no tree listing
                    files += [f for f in api.list_repo_files(REPO_ID, repo_type="dataset")
                              if f.startswith(path + "/")]
            except Exception as e:  # noqa: BLE001 - a folder that does not exist is expected
                status = getattr(getattr(e, "response", None), "status_code", None)
                if type(e).__name__ != "EntryNotFoundError" and status != 404:
                    raise
    return files


def download(patients, data_dir, token):
    from huggingface_hub import HfApi, hf_hub_download  # lazy: the module imports without huggingface_hub

    selected = select_files(list_repo(HfApi(token=token), patients), patients)
    local = {}
    for repo_path in [*selected.values(), *EXTRA_FILES]:
        print(f"downloading {repo_path}", flush=True)
        local[repo_path] = hf_hub_download(REPO_ID, repo_path, repo_type="dataset",
                                           local_dir=str(Path(data_dir) / "ct_rate"), token=token)
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
