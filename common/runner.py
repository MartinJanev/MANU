"""The run loop shared by every model: scan selection, resume, error isolation, blank controls, output files.

A model script implements the small Adapter below and calls main(adapter). The runner owns everything
else, so every model is scored from identical records.

Output: <results>/<name>/<split>/
  <name>_<tag>_processed.jsonl   records in the colleagues' format (common.schema.build_record)
  <name>_<tag>_details.jsonl     raw values, adapter details, timing and errors per scan
  <name>_<tag>_controls.jsonl    with --controls: one blank-volume record per dataset (row_index -1)
  run_info.json                  model, split, versions, host, git commit
"""
import argparse
import dataclasses
import datetime
import importlib
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Protocol

import numpy as np

from common.datasets import DATASETS, Scan, find_scans, ground_truth, load_splits
from common.schema import build_record, normalize_values
from common.volume import load_canonical

MODES = ("full_json", "qna_per_field", "findings_then_extract", "clip_or_heads")


class Adapter(Protocol):
    name: str
    model_id: str
    mode: str

    def load(self) -> None: ...

    def predict(self, scan: Scan, volumes: list) -> tuple:
        """volumes: load_canonical arrays of scan.files. Returns (raw values, JSON-serialisable details)."""
        ...


def _read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _append(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=repr) + "\n")


def _versions():
    versions = {}
    for module in ("torch", "transformers", "numpy", "nibabel"):
        try:
            versions[module] = importlib.import_module(module).__version__
        except Exception:  # not installed (or broken): leave it out
            pass
    return versions


def _write_run_info(out_dir, adapter, name, split, datasets, ms_flair):
    info = {
        "model": name,
        "model_id": adapter.model_id,
        "mode": adapter.mode,
        "split": split,
        "datasets": list(datasets),
        "ms_flair": ms_flair,
        "started": datetime.datetime.now().isoformat(),
        "host": socket.gethostname(),
        "git_commit": os.environ.get("MANU_GIT_COMMIT"),
        "versions": _versions(),
    }
    with open(out_dir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)


def _predict(adapter, name, scan, volumes_fn, row_index, image_path, metadata):
    """One scan through predict -> normalize -> build_record. Never raises: a failure becomes an error record."""
    start = time.time()
    raw, details, values, error = {}, {}, {}, None
    try:
        raw, details = adapter.predict(scan, volumes_fn())
        values = normalize_values(scan.dataset, raw)
    except Exception as e:  # one bad scan or an OOM must not kill the job
        error = repr(e)
        if "torch" in sys.modules and sys.modules["torch"].cuda.is_available():
            sys.modules["torch"].cuda.empty_cache()
    seconds = round(time.time() - start, 1)
    record = build_record(name, scan.dataset, adapter.mode, row_index, image_path, metadata, values, error)
    detail = {"image_path": image_path, "patient": scan.patient, "files": list(scan.files),
              "seconds": seconds, "details": {"raw_values": raw, **(details if isinstance(details, dict) else {"adapter_details": details})}, "error": error}
    return record, detail, seconds


def run(adapter, split, datasets, data_dir, reference_dir, results_dir, splits,
        limit=None, controls=False, ms_flair=False):
    name = adapter.name + ("-flair" if ms_flair else "")
    out_dir = Path(results_dir) / name / split
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_run_info(out_dir, adapter, name, split, datasets, ms_flair)
    loaded = False

    for dataset in datasets:
        tag = DATASETS[dataset]["tag"]
        processed = str(out_dir / f"{name}_{tag}_processed.jsonl")
        details_path = str(out_dir / f"{name}_{tag}_details.jsonl")
        controls_path = str(out_dir / f"{name}_{tag}_controls.jsonl")
        scans = find_scans(data_dir, dataset, split, splits, ms_flair=ms_flair)
        done_rows = _read_jsonl(processed)
        done = {r["input"]["image_path"] for r in done_rows}
        todo = [s for s in scans if s.image_path not in done][:limit]
        need_control = bool(controls and scans and not _read_jsonl(controls_path))
        print(f"{dataset}: {len(scans)} scans, {len(done)} already done, {len(todo)} to go"
              f"{', blank control to write' if need_control else ''}", flush=True)
        if not todo and not need_control:
            continue
        if not loaded:
            adapter.load()
            loaded = True

        def volumes_of(scan):
            return lambda: [load_canonical(p) for p in scan.files]

        for row_index, scan in enumerate(todo, len(done_rows)):
            metadata = ground_truth(scan, reference_dir, data_dir)
            record, detail, seconds = _predict(adapter, name, scan, volumes_of(scan), row_index,
                                               scan.image_path, metadata)
            _append(processed, record)
            _append(details_path, detail)
            shown = record["output"]["error"] or (record["output"]["parsed_response"] or {}).get("diagnosis_name")
            print(f"[{row_index + 1}/{len(scans)}] {scan.image_path} {seconds}s {shown}", flush=True)

        if need_control:
            first = scans[0]
            blank_path = "BLANK__" + first.image_path

            def blank_volumes():
                return [np.zeros_like(v) for v in volumes_of(first)()]

            metadata = ground_truth(first, reference_dir, data_dir)
            blank_scan = dataclasses.replace(first, image_path=blank_path)
            record, detail, seconds = _predict(adapter, name, blank_scan, blank_volumes, -1, blank_path, metadata)
            _append(controls_path, record)
            _append(details_path, detail)
            shown = record["output"]["error"] or (record["output"]["parsed_response"] or {}).get("diagnosis_name")
            print(f"[control] {blank_path} {seconds}s {shown}", flush=True)
    return out_dir


def main(adapter, argv=None):
    p = argparse.ArgumentParser(description=f"Run {adapter.name} over a dataset split.")
    p.add_argument("--split", required=True, choices=("dev", "eval", "full"))
    p.add_argument("--datasets", default=",".join(DATASETS), help="comma-separated keys of DATASETS")
    p.add_argument("--limit", type=int, default=None, help="only the first N unfinished scans per dataset")
    p.add_argument("--controls", action="store_true", help="also write one blank-volume control per dataset")
    p.add_argument("--ms-flair", action="store_true", help="feed FLAIR instead of T1 for MS (results under <model>-flair)")
    p.add_argument("--data", default="/data")
    p.add_argument("--reference", default="/reference")
    p.add_argument("--results", default="/results")
    p.add_argument("--splits", default="/manu/splits.json")
    args = p.parse_args(argv)
    datasets = [d for d in args.datasets.split(",") if d]
    unknown = [d for d in datasets if d not in DATASETS]
    if unknown:
        p.error(f"unknown datasets {unknown}; choose from {list(DATASETS)}")
    run(adapter, split=args.split, datasets=datasets, data_dir=args.data, reference_dir=args.reference,
        results_dir=args.results, splits=load_splits(args.splits), limit=args.limit,
        controls=args.controls, ms_flair=args.ms_flair)
