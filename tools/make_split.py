#!/usr/bin/env python3
"""Propose a larger eval split: keep the current eval patients, add randomly chosen ones up to N per dataset.

Run on the cluster login node (standard library only), next to data/. Dev patients are never picked. Writes
results/splits_proposed.json (fetch_results.sh brings it to the Mac) and prints patients and scans per dataset.
Keeping the old eval patients means scans that already have results are resumed, not rerun.

Usage: python3 tools/make_split.py [--data data] [--per-dataset 30] [--seed 0] [--out results/splits_proposed.json]
"""
import argparse
import collections
import json
import random
import sys
from pathlib import Path

MANU = Path(__file__).resolve().parent.parent
if str(MANU) not in sys.path:
    sys.path.insert(0, str(MANU))

from common.datasets import DATASETS, find_scans, load_splits  # noqa: E402


def propose(data_dir, splits, per_dataset, seed):
    """(new splits dict, {dataset: (patients, scans)}). Datasets with fewer patients take all of them."""
    proposed = json.loads(json.dumps(splits))
    counts = {}
    for dataset in DATASETS:
        scans = find_scans(str(data_dir), dataset, "full", splits)
        per_patient = collections.Counter(s.patient for s in scans)
        keep = list(splits["eval"].get(dataset, []))
        dev = set(splits["dev"].get(dataset, []))
        pool = sorted(p for p in per_patient if p not in dev and p not in keep)
        random.Random(f"{seed}:{dataset}").shuffle(pool)  # one stream per dataset: N for one does not move another
        chosen = keep + pool[:max(0, per_dataset - len(keep))]
        proposed["eval"][dataset] = sorted(chosen)
        counts[dataset] = (len(chosen), sum(per_patient[p] for p in chosen))
    return proposed, counts


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=str(MANU / "data"))
    ap.add_argument("--splits", default=str(MANU / "splits.json"))
    ap.add_argument("--per-dataset", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(MANU / "results" / "splits_proposed.json"))
    a = ap.parse_args()
    proposed, counts = propose(a.data, load_splits(a.splits), a.per_dataset, a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(proposed, indent=2) + "\n", encoding="utf-8")
    for dataset, (patients, scans) in counts.items():
        print(f"{dataset}: {patients} patients, {scans} scans")
    print(f"written to {a.out}; review it, then copy it over splits.json on the Mac and commit")


if __name__ == "__main__":
    main()
