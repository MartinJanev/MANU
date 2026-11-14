"""The run loop shared by every model: scan selection, resume, error isolation, blank controls, output files.

A model script implements the small Adapter below and calls main(adapter). The runner owns everything
else, so every model is scored from identical records.

Output: <results>/<name>/<split>/
  <name>_<tag>_processed.jsonl   records in the colleagues' format (common.schema.build_record)
  <name>_<tag>_details.jsonl     raw values, adapter details, timing, peak VRAM and errors per scan
  <name>_<tag>_controls.jsonl    with --controls: one blank-volume record per dataset (row_index -1)
  run_info.json                  model, split, versions, host, git commit

Resume: a rerun skips the scans (and controls) already recorded without an error. Error rows are removed
from the three files first and those scans are run again; a cut-off last line (a job killed mid-write) is
dropped the same way. Resuming continues a run of the same code: after changing what a model outputs,
move results/<name>/<split>/ to results/archive/ first.
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
from common.torch_compat import alias_transform_get_item_to_index

# Before any adapter imports transformers: the NGC 24.12 image's torch needs it (see common/torch_compat.py)
alias_transform_get_item_to_index()

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
    """(rows, clean): clean is False when the last line does not parse (a cut-off write; that line is left out)
    or the file does not end with a newline. Any other unparseable line raises."""
    if not os.path.exists(path):
        return [], True
    with open(path, encoding="utf-8") as f:
        text = f.read()
    lines = [line for line in text.splitlines() if line.strip()]
    rows = []
    for i, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except ValueError:
            if i < len(lines) - 1:
                raise
            print(f"{path}: dropped a cut-off last line", flush=True)
            return rows, False
    return rows, not text or text.endswith("\n")


def _is_error(row):
    """An error row of a processed/controls file (output.error) or of a details file (error)."""
    return bool(row["output"].get("error") if "output" in row else row.get("error"))


def _without_errors(path):
    """The rows of path that are not errors; the file is rewritten (temp file + os.replace) if anything was
    dropped, so the next append starts on a fresh line."""
    rows, clean = _read_jsonl(path)
    kept = [r for r in rows if not _is_error(r)]
    if len(kept) < len(rows) or not clean:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False, default=repr) + "\n" for r in kept)
        os.replace(tmp, path)
        if len(kept) < len(rows):
            print(f"{os.path.basename(path)}: {len(rows) - len(kept)} error rows removed, to be retried", flush=True)
    return kept


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


def _cuda():
    """torch.cuda if the adapter has imported torch and a GPU is there, else None (the runner never imports it)."""
    try:
        torch = sys.modules.get("torch")
        return torch.cuda if torch is not None and torch.cuda.is_available() else None
    except Exception:
        return None


def _predict(adapter, name, scan, volumes_fn, row_index, image_path, metadata):
    """One scan through predict -> normalize -> build_record. Never raises: a failure becomes an error record."""
    cuda, peak_vram_gb, measured = _cuda(), None, False
    if cuda is not None:
        try:  # best effort: the peak is only reported
            cuda.reset_peak_memory_stats()
            measured = True
        except Exception:
            pass
    start = time.time()
    raw, details, values, error = {}, {}, {}, None
    try:
        raw, details = adapter.predict(scan, volumes_fn())
        values = normalize_values(scan.dataset, raw)
    except Exception as e:  # one bad scan or an OOM must not kill the job
        error = repr(e)
    seconds = round(time.time() - start, 1)
    if measured:
        try:
            peak_vram_gb = round(cuda.max_memory_allocated() / 2**30, 1)
        except Exception:
            pass
    if error and cuda is not None:
        cuda.empty_cache()
    record = build_record(name, scan.dataset, adapter.mode, row_index, image_path, metadata, values, error)
    detail = {"image_path": image_path, "patient": scan.patient, "files": list(scan.files),
              "seconds": seconds, "peak_vram_gb": peak_vram_gb,
              "details": {"raw_values": raw, **(details if isinstance(details, dict) else {"adapter_details": details})},
              "error": error}
    return record, detail, seconds


def run(adapter, split, datasets, data_dir, reference_dir, results_dir, splits,
        limit=None, controls=False, ms_flair=False):
    if ms_flair and list(datasets) != ["ms"]:
        raise ValueError(f"--ms-flair needs the datasets to be exactly ['ms'], got {list(datasets)}")
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
        done_rows = _without_errors(processed)  # error rows (and a cut-off last line) are retried
        _without_errors(details_path)
        done_controls = _without_errors(controls_path)
        done = {r["input"]["image_path"] for r in done_rows}
        todo = [s for s in scans if s.image_path not in done][:limit]
        need_control = bool(controls and scans and not done_controls)
        next_index = max((r["processing"]["row_index"] for r in done_rows), default=-1) + 1
        print(f"{dataset}: {len(scans)} scans, {len(done)} already done, {len(todo)} to go"
              f"{', blank control to write' if need_control else ''}", flush=True)
        if not todo and not need_control:
            continue
        if not loaded:
            adapter.load()
            loaded = True

        def volumes_of(scan):
            return lambda: [load_canonical(p) for p in scan.files]

        for k, scan in enumerate(todo):
            metadata = ground_truth(scan, reference_dir, data_dir)
            record, detail, seconds = _predict(adapter, name, scan, volumes_of(scan), next_index + k,
                                               scan.image_path, metadata)
            _append(processed, record)
            _append(details_path, detail)
            shown = record["output"]["error"] or (record["output"]["parsed_response"] or {}).get("diagnosis_name")
            print(f"[{len(done) + k + 1}/{len(scans)}] {scan.image_path} {seconds}s {shown}", flush=True)

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
    if args.ms_flair and datasets != ["ms"]:
        p.error("--ms-flair needs --datasets ms (FLAIR exists only for MS; results go under <model>-flair)")
    run(adapter, split=args.split, datasets=datasets, data_dir=args.data, reference_dir=args.reference,
        results_dir=args.results, splits=load_splits(args.splits), limit=args.limit,
        controls=args.controls, ms_flair=args.ms_flair)
