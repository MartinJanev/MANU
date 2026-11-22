#!/usr/bin/env python3
"""Site-shortcut baseline: can IXI, MOLAB and MS scans be told apart without looking at anatomy?

Every normal scan is IXI, every tumor MOLAB, every MS open_ms_data, so a model can score by recognising the
dataset (scanner, resolution, contrast) instead of the disease. This script gives a deliberately dumb classifier
only acquisition features of one scan per eval patient (the T1 the models get):
  geometry   voxel spacing and volume shape (from the NIfTI, after reorientation to RAS)
  intensity  percentiles 1/5/25/50/75/95/99 of the voxel values, divided by the 99.9th percentile
z-scored, nearest class centroid, leave-one-patient-out. About 33% means the datasets look alike to it; close to
100% means the dataset is recognisable without anatomy, so a model's accuracy alone cannot show it sees disease.

Usage (needs numpy and nibabel; on the cluster inside the image, see README):
    python3 tools/site_baseline.py [--data data] [--split eval] [--results results]  -> results/site_baseline.md (eval, the pilot)
                                                  or results/site_baseline_<split>.md
"""
import argparse
import sys
from pathlib import Path

import numpy as np

MANU = Path(__file__).resolve().parent.parent
if str(MANU) not in sys.path:
    sys.path.insert(0, str(MANU))

from common.datasets import find_scans, load_splits  # noqa: E402

CLASSES = ("ixi", "molab", "ms")
PERCENTILES = (1, 5, 25, 50, 75, 95, 99)
GEOMETRY, INTENSITY = slice(0, 6), slice(6, 6 + len(PERCENTILES))


def features(path):
    import nibabel as nib

    img = nib.as_closest_canonical(nib.load(path))
    vol = np.nan_to_num(np.asarray(img.dataobj, dtype=np.float32))
    if vol.ndim == 4:
        vol = vol[..., 0]
    top = float(np.percentile(vol, 99.9)) or 1.0
    return np.array([*img.header.get_zooms()[:3], *vol.shape[:3], *(np.percentile(vol, PERCENTILES) / top)],
                    dtype=np.float64)


def loo_accuracy(X, y):
    """(accuracy, confusion[truth][pred]) of z-score + nearest centroid, each sample left out once."""
    X, y = np.asarray(X, dtype=np.float64), np.asarray(y)
    labels = sorted(set(y))
    confusion = {t: {p: 0 for p in labels} for t in labels}
    for i in range(len(y)):
        train = np.arange(len(y)) != i
        mean, std = X[train].mean(0), X[train].std(0)
        std[std == 0] = 1.0
        Z = (X - mean) / std
        centroids = {c: Z[train & (y == c)].mean(0) for c in labels if (train & (y == c)).any()}
        pred = min(centroids, key=lambda c: np.linalg.norm(Z[i] - centroids[c]))
        confusion[y[i]][pred] += 1
    return sum(confusion[c][c] for c in labels) / len(y), confusion


def report(X, y):
    lines = ["# Site-shortcut baseline", "",
             "Nearest-centroid on acquisition features only, leave-one-patient-out, one T1 per eval patient. "
             "33% = the datasets look alike; ~100% = the dataset (and so the diagnosis) is recognisable without "
             "anatomy.", "", "| features | accuracy |", "|---|---|"]
    X = np.asarray(X)
    for name, cols in (("all", slice(None)), ("geometry (spacing, shape)", GEOMETRY), ("intensity percentiles", INTENSITY)):
        acc, _ = loo_accuracy(X[:, cols], y)
        lines.append(f"| {name} | {acc:.1%} |")
    _, confusion = loo_accuracy(X, y)
    labels = list(confusion)
    lines += ["", "Confusion, all features (rows = dataset, columns = predicted):", "",
              "| dataset | " + " | ".join(labels) + " |", "|---" * (len(labels) + 1) + "|"]
    lines += [f"| {t} (n={sum(confusion[t].values())}) | " + " | ".join(str(confusion[t][p]) for p in labels) + " |"
              for t in labels]
    return "\n".join(lines) + "\n"


def output_name(split):
    return "site_baseline.md" if split == "eval" else f"site_baseline_{split}.md"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=str(MANU / "data"))
    ap.add_argument("--splits", default=str(MANU / "splits.json"))
    ap.add_argument("--split", default="eval")
    ap.add_argument("--results", default=str(MANU / "results"))
    a = ap.parse_args()
    splits = load_splits(a.splits)
    X, y = [], []
    for dataset in CLASSES:
        seen = set()
        for scan in find_scans(a.data, dataset, a.split, splits):  # sorted by image_path: first scan per patient
            if scan.patient not in seen:
                seen.add(scan.patient)
                X.append(features(scan.files[0]))
                y.append(dataset)
        print(f"{dataset}: {len(seen)} patients", flush=True)
    text = report(X, y)
    out = Path(a.results) / output_name(a.split)
    out.write_text(text, encoding="utf-8")
    print(text + f"written to {out}")


if __name__ == "__main__":
    main()
