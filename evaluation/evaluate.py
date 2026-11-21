#!/usr/bin/env python3
"""
Per-split evaluation of every model: results/<model>/<split>/ (new runs) and reference/*_processed.jsonl
(colleagues' MedGemma, M3D-Phi3/Llama2, RadFM from March), filtered to the split's patients.

Main metric: the diagnosis on IXI+MOLAB+MS as one 3-class task. A model that always gives the same answer has
balanced accuracy 33.3%, the "no signal from the image" line.

Usage:  python3 -I evaluation/evaluate.py --split dev|eval [--results results] [--reference reference]
                                          [--splits splits.json]   -> results/report_<split>.md
        python3 -I evaluation/evaluate.py --selftest
(The older all-scans report of the first RadFM runner, eval_summary.md, was removed after 45b8448.)
"""
import argparse
import collections
import json
import re
import statistics
import sys
from pathlib import Path

MANU = Path(__file__).resolve().parent.parent
if str(MANU) not in sys.path:  # the script runs with -I, which drops the script directory from sys.path
    sys.path.insert(0, str(MANU))

from common.datasets import DATASETS, load_splits, patient_id  # noqa: E402
from common.schema import NUMERIC_KEYS, PARSED_KEYS  # noqa: E402

TRUTH_DIAG = {"ixi": "normal", "molab": "tumor", "ms": "multiple sclerosis"}
DIAG_VOCAB = ("normal", "tumor", "multiple sclerosis", "stroke", "other abnormalities")
FIELDS = ("modality", "specialized_sequence", "plane", "diagnosis_name",
          "diagnosis_detailed", "primary_origin", "ms_type")


# ---------------------------------------------------------------- нормализација (дел 4.5)
def norm(v):
    if v is None:
        return None
    s = str(v).strip().lower().replace("_", " ")
    return None if s in ("", "null", "none", "n/a") else s


def norm_diag(v):
    s = norm(v)
    if s is None or s in DIAG_VOCAB:
        return s
    for label, key in (("tumor", "tumo"), ("tumor", "metasta"), ("multiple sclerosis", "sclerosis"),
                       ("stroke", "stroke"), ("normal", "normal")):
        if key in s:
            return label
    return None  # надвор од речникот -> null


def norm_origin(v):
    s = norm(v)
    return {"overy": "ovary", "blader": "bladder", "bone sarcoma": "bone-sarcoma"}.get(s, s)


def norm_ms(v):
    s = re.sub(r"[\s-]", "", norm(v) or "")
    for code, keys in (("SMRR", ("rr", "relaps")), ("SMSP", ("sp", "secondary")), ("SMPP", ("pp", "primary"))):
        if any(k in s for k in keys):
            return code
    return None


NORMALIZE = {"diagnosis_name": norm_diag, "primary_origin": norm_origin, "ms_type": norm_ms}


# ---------------------------------------------------------------- вчитување
def describe(path):
    """Colleagues' file name -> (model, dataset)."""
    n = path.name.lower()
    dataset = ("molab" if re.search(r"metasta|molab", n) else
               "ms" if re.search(r"multiple_sclerosis|_ms", n) else "ixi")
    model = {"medgemma": "medgemma-1.5", "phi3": "m3d-phi3", "llama2": "m3d-llama2",
             "radfm": "radfm (Outputs, mar)"}[n.split("_")[0]]
    return model + (" +prior" if "predznaenje" in n else ""), dataset


# ---------------------------------------------------------------- метрики
def class_metrics(truth, pred):
    """Точност, balanced accuracy и macro-F1 (просек по класите во вистината)."""
    recalls, f1s = [], []
    for c in sorted(set(truth)):
        tp = sum(t == c and p == c for t, p in zip(truth, pred))
        n_t, n_p = truth.count(c), pred.count(c)
        rec, prec = tp / n_t, (tp / n_p if n_p else 0.0)
        recalls.append(rec)
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    acc = sum(t == p for t, p in zip(truth, pred)) / len(truth)
    return {"acc": acc, "bal_acc": statistics.mean(recalls), "macro_f1": statistics.mean(f1s)}


def field(pred, name):
    return NORMALIZE.get(name, norm)((pred or {}).get(name))


def pct(x):
    return "—" if x is None else f"{100 * x:.1f}%"


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    return out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]


# ---------------------------------------------------------------- делови од извештајот (споделени)
def confusion_lines(runs, corner, caption):
    L = []
    for model, per_ds in runs.items():
        counts = {d: collections.Counter(str(field(p, "diagnosis_name")) for _, p, _ in rs)
                  for d, rs in per_ds.items()}
        cols = sorted({c for cnt in counts.values() for c in cnt})
        L += [f"**{model}** ({caption})", ""]
        L += table([corner, *cols],
                   [[f"{d} ({TRUTH_DIAG[d]})", *(counts[d][c] for c in cols)] for d in counts]) + [""]
    return L


def collapse_lines(runs, field_header):
    L = []
    for model, per_ds in runs.items():
        rows = []
        for f in FIELDS:
            cells = []
            for d in TRUTH_DIAG:
                rs = per_ds.get(d)
                if not rs:
                    cells.append("—")
                    continue
                c = collections.Counter(str(field(p, f)) for _, p, _ in rs)
                top, n = c.most_common(1)[0]
                cells.append(f"{len(c)} · {100 * n / len(rs):.0f}% ({top})")
            rows.append([f, *cells])
        L += [f"**{model}**", ""] + table([field_header, "IXI", "MOLAB", "MS"], rows) + [""]
    return L


# ---------------------------------------------------------------- per-split evaluation (new model runs)
SPLIT_ORDER = ("ixi", "molab", "ms", "ct_rate")
FULL_SUFFIX = " (full run)"
COLLEAGUES_MODE = "full_json (colleagues)"


def _tag_dataset(name):
    """'<model>_<tag>_processed.jsonl' (or _controls / _details) -> dataset key, None for an unknown tag."""
    for suffix in ("_processed.jsonl", "_controls.jsonl", "_details.jsonl"):
        if name.endswith(suffix):
            stem = name[:-len(suffix)]
            break
    else:
        return None
    for dataset, spec in sorted(DATASETS.items(), key=lambda kv: -len(kv[1]["tag"])):
        if stem.endswith("_" + spec["tag"]):
            return dataset
    return None


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _run_files(directory, suffix="_processed.jsonl"):
    """dataset -> path of the run files in one results/<model>/<split>/ directory."""
    found = {}
    for path in sorted(directory.glob("*" + suffix)):
        dataset = _tag_dataset(path.name)
        if dataset is not None:
            found[dataset] = path
    return found


def _rows(path, keep=None):
    rows = [(r["input"]["image_path"], r["output"].get("parsed_response"), r["input"]["metadata"])
            for r in _read_jsonl(path)]
    return rows if keep is None else [r for r in rows if keep(r[0])]


def load_split(results, reference, split, splits):
    """model -> dataset -> [(image_path, parsed_response, metadata)] for one split (datasets ixi, molab, ms, ct_rate).

    Sources: results/<m>/<split>/; results/<m>/full/ filtered to the split's patients when the model has no
    <split> directory (named "<m> (full run)"); reference/ filtered to the split's patients."""
    results, reference = Path(results), Path(reference)
    runs = collections.defaultdict(dict)

    def add(model, dataset, path, filtered):
        patients = set(splits.get(split, {}).get(dataset, []))
        keep = (lambda image: patient_id(dataset, image) in patients) if filtered else None
        rows = _rows(path, keep)
        if rows:
            runs[model][dataset] = rows

    for model_dir in sorted(p for p in results.iterdir() if p.is_dir()) if results.is_dir() else []:
        if (model_dir / split).is_dir():
            for dataset, path in _run_files(model_dir / split).items():
                add(model_dir.name, dataset, path, False)
        elif (model_dir / "full").is_dir():
            for dataset, path in _run_files(model_dir / "full").items():
                add(model_dir.name + FULL_SUFFIX, dataset, path, True)
    for path in sorted(reference.glob("*_processed.jsonl")) if reference.is_dir() else []:
        model, dataset = describe(path)
        add(model, dataset, path, True)
    return {m: {d: per[d] for d in SPLIT_ORDER if d in per} for m, per in runs.items()}


def _model_dir(results, model, split):
    """Directory of a model's run files for this split, or None for a reference model."""
    if model.endswith(FULL_SUFFIX):
        path = Path(results) / model[:-len(FULL_SUFFIX)] / "full"
    else:
        path = Path(results) / model / split
    return path if path.is_dir() else None


def blank_agreement(results, model, split):
    """Per dataset, share of the categorical fields whose most common real value is non-null where the
    blank-volume control gives that same value; averaged over the datasets that contribute. A dataset
    with no such field contributes nothing; None when no dataset contributes."""
    directory = Path(results) / model / split
    if not directory.is_dir():
        return None
    shares = []
    processed_files = _run_files(directory)
    for dataset, path in _run_files(directory, "_controls.jsonl").items():
        processed = processed_files.get(dataset)
        controls = [r["output"]["parsed_response"] for r in _read_jsonl(path) if r["output"].get("parsed_response")]
        real = [p for _, p, _ in _rows(processed) if p] if processed else []
        if not controls or not real:
            continue
        keys = [k for k in PARSED_KEYS[dataset] if k not in NUMERIC_KEYS and k != "scan_name"]
        hits = []
        for key in keys:
            common = collections.Counter(json.dumps(p.get(key)) for p in real).most_common(1)[0][0]
            if common != "null":
                hits.append(json.dumps(controls[0].get(key)) == common)
        if hits:
            shares.append(sum(hits) / len(hits))
    return statistics.mean(shares) if shares else None


def blank_diagnoses(results, model, split):
    """Diagnosis of the blank-volume control per dataset, 'IXI · MOLAB · MS' ('null' when it gave none,
    '—' when that dataset has no control); None when the model has no controls at all."""
    directory = Path(results) / model / split
    files = _run_files(directory, "_controls.jsonl") if directory.is_dir() else {}
    if not files:
        return None
    shown = []
    for dataset in TRUTH_DIAG:
        rows = _read_jsonl(files[dataset]) if dataset in files else []
        if not rows:
            shown.append("—")
            continue
        shown.append(field(rows[0]["output"].get("parsed_response"), "diagnosis_name") or "null")
    return " · ".join(shown)


def _median_seconds(directory):
    if directory is None:
        return None
    seconds = [r["seconds"] for path in _run_files(directory, "_details.jsonl").values() for r in _read_jsonl(path)
               if r.get("seconds") is not None and not str(r.get("image_path", "")).startswith("BLANK__")]
    return statistics.median(seconds) if seconds else None


def _mode(directory):
    """Prompt mode from system_prompt_type ('full_json_v1' -> 'full_json'); colleagues' files have ''."""
    for path in _run_files(directory).values() if directory else []:
        for r in _read_jsonl(path):
            kind = r["input"].get("system_prompt_type") or ""
            return kind[:-3] if kind.endswith("_v1") else (kind or COLLEAGUES_MODE)
    return COLLEAGUES_MODE


def _is_valid(parsed):
    return bool(parsed) and any(v is not None for k, v in parsed.items() if k != "scan_name")


def _diag_acc(rows, dataset):
    return sum(field(p, "diagnosis_name") == TRUTH_DIAG[dataset] for _, p, _ in rows) / len(rows)


def _molab_patient_acc(rows):
    """Majority vote of the diagnosis over each patient's scans (correct only with a strict plurality
    for the truth label; any tie for first place, nulls counted as a label, is wrong)."""
    votes = collections.defaultdict(list)
    for image, p, _ in rows:
        votes[patient_id("molab", image)].append(field(p, "diagnosis_name"))
    wins = []
    for v in votes.values():
        counts = collections.Counter(v)
        mine = counts.pop(TRUTH_DIAG["molab"], 0)
        wins.append(mine > max(counts.values(), default=0))
    return sum(wins) / len(wins)


def _predicted_abnormal(pred):
    """True when the normalized diagnosis_name is non-null and not 'normal', False for 'normal', None when it
    is null (counts as wrong). An answer outside the vocabulary (e.g. 'emphysema') normalizes to null, as the
    runner already stores it."""
    diagnosis = field(pred, "diagnosis_name")
    return None if diagnosis is None else diagnosis != "normal"


def ct_rate_table(runs):
    """Markdown lines: per model, how a chest-CT control is read (modality, plane, normal vs abnormal).
    Truth abnormal = at least one positive label; predicted abnormal = normalized diagnosis_name non-null and
    not 'normal'; a null diagnosis (including an out-of-vocabulary answer) is wrong. Scans with unknown truth
    (abnormalities null) are left out of the last column's denominator ('—' when none is left)."""
    rows = []
    for model, per_ds in runs.items():
        rs = per_ds.get("ct_rate")
        if not rs:
            continue
        known = [(p, m["abnormalities"]) for _, p, m in rs if m.get("abnormalities") is not None]
        agree = pct(sum(_predicted_abnormal(p) == bool(a) for p, a in known) / len(known)) if known else "—"
        rows.append([model, len(rs),
                     pct(sum(field(p, "modality") == "ct" for _, p, _ in rs) / len(rs)),
                     pct(sum(field(p, "plane") == "axial" for _, p, _ in rs) / len(rs)), agree])
    if not rows:
        return []
    return ["## CT-RATE (chest CT control)", "",
            "Share of scans where the answer is right: modality CT, plane axial, normal/abnormal agrees with "
            "the labels (truth abnormal = at least one positive label; predicted abnormal = a diagnosis other "
            "than normal; a null diagnosis, including an answer outside the vocabulary, is wrong; scans with "
            "unknown labels are excluded from the last column).", ""] \
        + table(["model", "n", "modality = CT", "plane = axial", "normal/abnormal agrees"], rows) + [""]


def split_report(runs, results, split):
    """Markdown lines: main table per model, then the confusion and collapse sections."""
    scored = {m: {d: rs for d, rs in per.items() if d in TRUTH_DIAG and rs} for m, per in runs.items()}
    rows = []
    for model, per_ds in runs.items():
        directory = _model_dir(results, model, split)
        all_rows = [r for rs in per_ds.values() for r in rs]
        mine = scored[model]
        if set(mine) == set(TRUTH_DIAG):
            t = [TRUTH_DIAG[d] for d, rs in mine.items() for _ in rs]
            p = [field(pr, "diagnosis_name") for rs in mine.values() for _, pr, _ in rs]
            pooled = pct(class_metrics(t, p)["bal_acc"])
        else:
            pooled = "—"
        blank, blank_diag = None, None
        if directory is not None:
            blank = blank_agreement(results, directory.parent.name, directory.name)
            blank_diag = blank_diagnoses(results, directory.parent.name, directory.name)
        seconds = _median_seconds(directory)
        rows.append([model, _mode(directory), len(all_rows),
                     pct(sum(_is_valid(p) for _, p, _ in all_rows) / len(all_rows)) if all_rows else "—",
                     pct(_diag_acc(mine["ixi"], "ixi")) if "ixi" in mine else "—",
                     pct(_diag_acc(mine["molab"], "molab")) if "molab" in mine else "—",
                     pct(_molab_patient_acc(mine["molab"])) if "molab" in mine else "—",
                     pct(_diag_acc(mine["ms"], "ms")) if "ms" in mine else "—",
                     pooled, pct(blank), blank_diag or "—", "—" if seconds is None else f"{seconds:.1f}"])
    L = [f"# Evaluation: {split} split", "",
         "Accuracy is the share of scans with the expected diagnosis (IXI normal, MOLAB tumor, MS multiple "
         "sclerosis); MOLAB (patient majority) votes over each patient's scans. Pooled bal.acc covers "
         "IXI+MOLAB+MS as one 3-class task (33.3% = no signal). Blank agreement = share of categorical "
         "fields where the blank-volume control equals the most common real answer (high = image ignored).", ""]
    L += table(["model", "mode", "records", "valid", "IXI acc", "MOLAB acc (scan)", "MOLAB acc (patient majority)",
                "MS acc", "pooled bal.acc", "blank agreement", "blank → diagnosis (IXI · MOLAB · MS)",
                "median s/scan"], rows) + [""]
    L += ["MOLAB patient majority: correct only with a strict plurality for the truth label (ties count as wrong). "
          "Blank agreement: share of categorical fields with a non-null usual answer where the blank-image "
          "control gives that same answer (high = the model ignores the image). Blank → diagnosis: what the "
          "blank-volume control was diagnosed as; the same label as on every real scan means the image is ignored, "
          "null means the model declined to answer.", ""]
    site = results / "site_baseline.md"  # tools/site_baseline.py, eval patients
    if split == "eval" and site.exists():
        L += ["#" + site.read_text(encoding="utf-8").strip(), ""]
    shown = {m: per for m, per in scored.items() if per}
    L += ["## Confusion matrices (diagnosis)", ""]
    L += confusion_lines(shown, "truth \\ model", "rows = truth, columns = predicted")
    L += ["## Collapse: how many distinct answers does the model give?", "",
          "Cell = number of distinct values · share of the most common (most common value).", ""]
    L += collapse_lines(shown, "field")
    return L + ct_rate_table(runs)


# ---------------------------------------------------------------- self-check
def selftest():
    t = ["normal"] * 4 + ["tumor"] * 4 + ["multiple sclerosis"] * 4
    assert abs(class_metrics(t, ["normal"] * 12)["bal_acc"] - 1 / 3) < 1e-9
    assert class_metrics(t, t) == {"acc": 1.0, "bal_acc": 1.0, "macro_f1": 1.0}
    assert norm_diag("Brain tumor") == "tumor" and norm_diag("Normal") == "normal" and norm_diag("Brain") is None
    assert norm_diag("multiple_sclerosis") == "multiple sclerosis"
    assert norm_ms("SMRR") == "SMRR" and norm_ms("secondary progressive") == "SMSP" and norm_ms("null") is None
    assert norm_origin("overy") == "ovary" and norm_origin("lung-NSCLC") == "lung-nsclc"
    assert describe(Path("medgemma_15_predznaenje_brain_metastases_processed.jsonl")) == ("medgemma-1.5 +prior", "molab")
    assert describe(Path("radfm_ms_processed.jsonl")) == ("radfm (Outputs, mar)", "ms")
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default=str(MANU / "reference"))
    ap.add_argument("--results", default=str(MANU / "results"))
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--split", choices=("dev", "eval"), help="per-split report of the model runs (required)")
    ap.add_argument("--reference", help="colleagues' files (default: --outputs)")
    ap.add_argument("--splits", default=str(MANU / "splits.json"))
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.split:
        ap.error("--split dev|eval is required")
    results = Path(a.results)
    runs = load_split(results, Path(a.reference or a.outputs), a.split, load_splits(a.splits))
    if not runs:
        sys.exit(f"No runs for split {a.split} in {a.results} or {a.reference or a.outputs}")
    text = "\n".join(split_report(runs, results, a.split)) + "\n"
    results.mkdir(parents=True, exist_ok=True)
    (results / f"report_{a.split}.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
