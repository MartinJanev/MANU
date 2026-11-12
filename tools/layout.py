"""Check that a MANU checkout (local or cluster) has the expected layout.

Usage:  python3 tools/layout.py [--cluster] [ROOT]
Prints missing entries, exits 1 if any. An entry ending in "/" is a directory.
"""
import argparse
import sys
from pathlib import Path

CODE_LAYOUT = ["README.md", "splits.json", ".gitignore", "common/__init__.py",
               "models/radfm/run.py", "jobs/job_radfm.sh", "containers/radfm.Dockerfile",
               "data_tools/prezemi_podatoci.sh", "data_tools/proveri_podatoci.py",
               "evaluation/evaluate.py", "evaluation/check_schema.py", "reference/",
               "results/", "tools/layout.py", "tests/"]

CLUSTER_LAYOUT = CODE_LAYOUT + ["data/ixi/T1/", "data/molab/nifti/",
                                "data/open_ms_data/cross_sectional/raw/",
                                "containers/radfm.sif", "hf_cache/", "pyuser/radfm/",
                                "third_party/RadFM/Quick_demo/", "results/slurm/",
                                "results/radfm/full/"]


def check(root: Path, layout: list[str]) -> list[str]:
    """Return the entries of `layout` missing under `root`, in layout order."""
    root = Path(root)
    missing = []
    for entry in layout:
        p = root / entry
        if not (p.is_dir() if entry.endswith("/") else p.is_file()):
            missing.append(entry)
    return missing


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root", nargs="?", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--cluster", action="store_true", help="also require cluster-only data/containers")
    a = ap.parse_args()
    missing = check(Path(a.root), CLUSTER_LAYOUT if a.cluster else CODE_LAYOUT)
    for m in missing:
        print(m)
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
