#!/usr/bin/env python3
"""Re-score saved answers after a parser fix, without the GPU.

Reads details.raw of every row in results/<model>/<split>/*_details.jsonl, parses it again with the adapter's
current code and rewrites the processed, controls and details files. The directory is copied to
results/archive/<model>/<split>_reparse_<time>/ first; run_info.json lists the re-score under "reparsed".

Usage: python3 -I tools/reparse.py healthgpt-pro-27b|medgemma-1.5-4b dev|eval [--results results]
       (needs the adapters' imports: numpy, nibabel, scipy, Pillow)
"""
import argparse
import datetime
import json
import shutil
import subprocess
import sys
from pathlib import Path

MANU = Path(__file__).resolve().parent.parent
if str(MANU) not in sys.path:  # -I drops the script directory from sys.path
    sys.path.insert(0, str(MANU))

from common.datasets import DATASETS  # noqa: E402
from common.runner import PredictionError, _read_jsonl  # noqa: E402
from common.schema import build_record, normalize_values  # noqa: E402


def _healthgpt(raw, dataset, details):
    from models.healthgptpro27b.run import answer_values

    # The saved eval/dev answers (jobs 2734 dev, 2762 eval) are plain JSON: the template had not opened a <think>
    obj, problems = answer_values(raw, opened=False)
    details["problems"] = problems
    return obj or {}


def _medgemma(raw, dataset, details):
    from models.medgemma15.run import answer_values

    return answer_values(raw, dataset, details)


REPARSERS = {"healthgpt-pro-27b": _healthgpt, "medgemma-1.5-4b": _medgemma}


def _write_jsonl(path, rows):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False, default=repr) + "\n" for r in rows)
    tmp.replace(path)


def _commit():
    try:
        return subprocess.run(["git", "-C", str(MANU), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def reparse(results, model, split):
    """Re-score results/<model>/<split>; returns the backup directory."""
    reparser = REPARSERS[model]
    out_dir = Path(results) / model / split
    if not out_dir.is_dir():
        raise FileNotFoundError(out_dir)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = Path(results) / "archive" / model / f"{split}_reparse_{stamp}"
    shutil.copytree(out_dir, backup)
    info = json.loads((out_dir / "run_info.json").read_text(encoding="utf-8"))
    mode = info["mode"]

    for dataset, spec in DATASETS.items():
        details_path = out_dir / f"{model}_{spec['tag']}_details.jsonl"
        if not details_path.exists():
            continue
        records = {}  # image_path -> (file, index) of its processed or controls row
        files = {}
        for kind in ("processed", "controls"):
            path = out_dir / f"{model}_{spec['tag']}_{kind}.jsonl"
            files[path], _ = _read_jsonl(str(path))
            for i, row in enumerate(files[path]):
                records[row["input"]["image_path"]] = (path, i)
        details_rows, _ = _read_jsonl(str(details_path))
        changed = 0
        for detail in details_rows:
            image_path = detail["image_path"]
            if image_path not in records:
                continue
            adapter_details = {k: v for k, v in detail["details"].items() if k != "raw_values"}
            error = None
            try:
                raw_values = reparser(adapter_details.get("raw", ""), dataset, adapter_details)
                values = normalize_values(dataset, raw_values)
            except PredictionError as e:
                raw_values, values, error = {}, {}, f"{type(e).__name__}: {e}"
                adapter_details = e.details
            path, i = records[image_path]
            old = files[path][i]
            new = build_record(model, dataset, mode, old["processing"]["row_index"], image_path,
                               old["input"]["metadata"], values, error)
            new["timestamp"] = old["timestamp"]
            changed += new["output"] != old["output"]
            files[path][i] = new
            detail["details"] = {"raw_values": raw_values, **adapter_details}
            detail["error"] = error
        for path, rows in files.items():
            if path.exists():
                _write_jsonl(path, rows)
        _write_jsonl(details_path, details_rows)
        print(f"{dataset}: {changed} of {len(details_rows)} rows changed", flush=True)

    info.setdefault("reparsed", []).append({"time": datetime.datetime.now().isoformat(), "git_commit": _commit()})
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return backup


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model", choices=sorted(REPARSERS))
    ap.add_argument("split")
    ap.add_argument("--results", default=str(MANU / "results"))
    a = ap.parse_args()
    backup = reparse(a.results, a.model, a.split)
    print(f"previous files copied to {backup}")


if __name__ == "__main__":
    main()
