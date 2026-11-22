#!/usr/bin/env python3
"""Every number of the paper, from results/<model>/paper/ only: LaTeX tables, none typed by hand.

A dataset of a run is scored only when it holds exactly the split's patients (one record each), otherwise its cells
are \\TBD. An error record or a null answer is a wrong answer, never a dropped one. The rules are fixed in
evaluation/PROTOCOL.md; change them there first.

Usage:  python3 -I evaluation/stats.py [--results results] [--out paper/tables] [--splits splits.json]
"""
import argparse
import collections
import itertools
import math
import re
import sys
from pathlib import Path

import numpy as np

MANU = Path(__file__).resolve().parent.parent
if str(MANU) not in sys.path:  # the script runs with -I
    sys.path.insert(0, str(MANU))

from common.datasets import load_splits, patient_id  # noqa: E402
from evaluation.evaluate import TRUTH_DIAG, _read_jsonl, _run_files, class_metrics, field, norm  # noqa: E402

BRAIN = ("ixi", "molab", "ms")
SEED, N_RESAMPLES = 0, 10000
# results/<name> -> the paper's name, in the order of the tables
MODELS_3D = {"radfm-qna": "RadFM", "m3d-lamed-phi3": "M3D-LaMed-Phi-3", "m3d-lamed-llama2": "M3D-LaMed-Llama-2",
             "braingemma3d": "BrainGemma3D", "med-2e3": "Med-2E3"}
MODELS_SLICES = {"medgemma-1.5-4b": "MedGemma 1.5", "healthgpt-pro-27b": "HealthGPT-Pro", "lingshu-7b": "Lingshu",
                 "qoq-med-vl-7b": "QoQ-Med-VL", "fleming-vl-8b": "Fleming-VL", "qwen3-vl-8b": "Qwen3-VL"}
MODELS_REFERENCE = {"hlip": "HLIP (reference)"}
MODELS = {**MODELS_3D, **MODELS_SLICES, **MODELS_REFERENCE}
QNA_NATIVE = ("radfm-qna", "m3d-lamed-phi3", "m3d-lamed-llama2", "med-2e3")
TBD = "\\TBD"

Row = collections.namedtuple("Row", "patient pred meta error")


def load_run(results, name, splits, split="paper"):
    """dataset -> rows sorted by patient, only for the datasets that are complete."""
    folder, run = Path(results) / name / split, {}
    if not folder.is_dir():
        return run
    for dataset, path in _run_files(folder).items():
        wanted = set(splits[split][dataset])
        rows = [Row(patient_id(dataset, r["input"]["image_path"]), r["output"].get("parsed_response"),
                    r["input"]["metadata"], r["output"].get("error")) for r in _read_jsonl(path)]
        patients = collections.Counter(r.patient for r in rows)
        if set(patients) - wanted or any(n > 1 for n in patients.values()):
            raise ValueError(f"{path}: records outside the split {split}, or two for one patient")
        if set(patients) == wanted:
            run[dataset] = sorted(rows, key=lambda r: r.patient)
    return run


def outcome(pred, error):
    if error or not pred or all(v is None for k, v in pred.items() if k != "scan_name"):
        return "unparseable"
    return "answered" if field(pred, "diagnosis_name") is not None else "abstained"


def complete(run):
    return set(BRAIN) <= set(run)


def correct(rows, dataset):
    return np.array([field(r.pred, "diagnosis_name") == TRUTH_DIAG[dataset] for r in rows])


def diagnosis(run):
    recall = {d: float(correct(run[d], d).mean()) for d in BRAIN}
    truth = [TRUTH_DIAG[d] for d in BRAIN for _ in run[d]]
    pred = [field(r.pred, "diagnosis_name") for d in BRAIN for r in run[d]]
    f1 = {}
    for c in set(truth):
        tp = sum(t == c == p for t, p in zip(truth, pred))
        total = truth.count(c) + pred.count(c)
        f1[c] = 2 * tp / total if total else 0.0
    m = class_metrics(truth, pred)
    return {"recall": recall, "bal_acc": m["bal_acc"], "accuracy": m["acc"], "macro_f1": m["macro_f1"],
            "weighted_f1": sum(f1[c] * truth.count(c) for c in f1) / len(truth)}


def interval(run):
    """95% bootstrap interval of the balanced accuracy, patients resampled within each dataset."""
    rng, total = np.random.default_rng(SEED), 0.0
    for d in BRAIN:
        ok = correct(run[d], d)
        # resampling n zeros and ones with replacement is a binomial draw
        total = total + rng.binomial(len(ok), ok.mean(), N_RESAMPLES) / len(ok) / len(BRAIN)
    low, high = np.percentile(total, (2.5, 97.5))
    return float(low), float(high)


def compare(a, b):
    """(balanced-accuracy difference a - b, two-sided p) on the same patients: paired sign-flip permutation."""
    rng, observed, flipped = np.random.default_rng(SEED), 0.0, np.zeros(N_RESAMPLES)
    for d in BRAIN:
        diff = correct(a[d], d).astype(int) - correct(b[d], d).astype(int)
        weight = 1 / (len(diff) * len(BRAIN))
        observed += diff.sum() * weight
        k = int(np.count_nonzero(diff))  # only the patients on which the two runs differ can flip
        if k:
            flipped += (2 * rng.binomial(k, 0.5, N_RESAMPLES) - k) * weight
    p = (np.count_nonzero(np.abs(flipped) >= abs(observed) - 1e-12) + 1) / (N_RESAMPLES + 1)
    return float(observed), float(p)


def holm(ps):
    adjusted, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(sorted(range(len(ps)), key=lambda i: ps[i])):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adjusted[i] = running
    return adjusted


def chest(run):
    tp = tn = positive = negative = 0
    for r in run["ct_rate"]:
        if r.meta.get("abnormalities") is None:
            continue
        answer = field(r.pred, "diagnosis_name")
        if r.meta["abnormalities"]:
            positive += 1
            tp += answer not in (None, "normal")
        else:
            negative += 1
            tn += answer == "normal"
    sensitivity = tp / positive if positive else 0.0
    specificity = tn / negative if negative else 0.0
    return {"sensitivity": sensitivity, "specificity": specificity, "bal_acc": (sensitivity + specificity) / 2,
            "n": positive + negative}


def fine(run, key):
    """primary_origin (MOLAB) or ms_type (MS): patients with a known truth, next to the majority answer."""
    rows = run["molab" if key == "primary_origin" else "ms"]
    pairs = [(field(r.meta, key), field(r.pred, key)) for r in rows if field(r.meta, key) is not None]
    if not pairs:
        return None
    truth, pred = [t for t, _ in pairs], [p for _, p in pairs]
    m = class_metrics(truth, pred)
    return {"n": len(pairs), "accuracy": m["acc"], "macro_f1": m["macro_f1"],
            "baseline": collections.Counter(truth).most_common(1)[0][1] / len(truth)}


def expected_sequence(dataset, flair=False):
    return "FLAIR" if flair else {"ixi": "T1", "molab": "T1C+", "ms": "T1"}[dataset]


def field_accuracy(rows, key, expected):
    return sum(field(r.pred, key) == norm(expected) for r in rows) / len(rows)


def position_agreement(results, name):
    """Share of scans whose diagnosis is the same in both option orders."""
    folder, agree = Path(results) / name / "paper", []
    for dataset, path in _run_files(folder, "_details.jsonl").items() if folder.is_dir() else ():
        if dataset not in BRAIN:
            continue
        for r in _read_jsonl(path):
            log = (r.get("details") or {}).get("diagnosis_name")
            if isinstance(log, dict) and not str(r["image_path"]).startswith("BLANK__"):
                agree.append(bool(log.get("agree")))
    return sum(agree) / len(agree) if agree else None


def blank_answers(results, name):
    folder, shown = Path(results) / name / "paper", []
    files = _run_files(folder, "_controls.jsonl") if folder.is_dir() else {}
    short = {"normal": "N", "tumor": "T", "multiple sclerosis": "MS", "stroke": "S", "other abnormalities": "O"}
    for d in BRAIN:
        rows = _read_jsonl(files[d]) if d in files else []
        if not rows:
            return None
        shown.append(short.get(field(rows[0]["output"].get("parsed_response"), "diagnosis_name"), "--"))
    return " / ".join(shown)


# ---------------------------------------------------------------- LaTeX
def num(x):
    return TBD if x is None else f"{100 * x:.1f}"


def signed(x):
    return TBD if x is None else f"${100 * x:+.1f}$"


def latex(caption, label, spec, header, rows, size="footnotesize"):
    lines = ["% generated by evaluation/stats.py: do not edit", "\\begin{table}[htbp]", "\\centering",
             f"\\caption{{{caption}}}", f"\\label{{{label}}}", f"\\{size}",
             f"\\begin{{tabular}}{{{spec}}}", "\\toprule",
             " & ".join(f"\\textbf{{{h}}}" if h else "" for h in header) + " \\\\", "\\midrule"]
    lines += ["\\midrule" if row is None else " & ".join(str(c) for c in row) + " \\\\" for row in rows]
    return "\n".join(lines + ["\\bottomrule", "\\end{tabular}", "\\end{table}"]) + "\n"


def grouped(make_row, groups=(MODELS_3D, MODELS_SLICES, MODELS_REFERENCE)):
    rows = []
    for group in groups:
        if rows:
            rows.append(None)
        rows += [make_row(key, shown) for key, shown in group.items()]
    return rows


def acquisition_baseline(results):
    path = Path(results) / "site_baseline_paper.md"
    found = re.search(r"\| all \| ([\d.]+)%", path.read_text(encoding="utf-8")) if path.exists() else None
    return float(found.group(1)) / 100 if found else None


def write_tables(results, out, splits):
    results, out = Path(results), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cache = {}

    def run(name):
        if name not in cache:
            cache[name] = load_run(results, name, splits)
        return cache[name]

    def save(name, text):
        (out / f"results_{name}.tex").write_text(text, encoding="utf-8")

    n = {d: len(splits["paper"][d]) for d in splits["paper"]}

    def diagnosis_row(key, shown):
        r, flair = run(key), run(f"{key}-flair")
        recalls = [num(float(correct(r[d], d).mean()) if d in r else None) for d in BRAIN]
        recalls.append(num(float(correct(flair["ms"], "ms").mean()) if "ms" in flair else None))
        if not complete(r):
            return [shown, *recalls, TBD, TBD, TBD, TBD]
        d, (low, high) = diagnosis(r), interval(r)
        return [shown, *recalls, f"{num(d['bal_acc'])} [{num(low)}, {num(high)}]", num(d["accuracy"]),
                num(d["macro_f1"]), num(d["weighted_f1"])]

    rows = grouped(diagnosis_row)
    rows += [None, ["Constant answer", "--", "--", "--", "--", num(1 / 3), "--", "--", "--"],
             ["Acquisition only", "--", "--", "--", "--", num(acquisition_baseline(results)), "--", "--", "--"]]
    save("diagnosis", latex(
        f"Diagnosis (\\%). Recall per dataset (IXI $n={n['ixi']}$, MOLAB $n={n['molab']}$, MS $n={n['ms']}$) and, on "
        "the three T1 conditions together, balanced accuracy with its 95\\% bootstrap interval, accuracy, macro-F1 "
        "and weighted F1. An unparseable answer counts as wrong. Acquisition only: a classifier that sees voxel "
        "spacing, volume shape and intensity percentiles, but no anatomy.",
        "tab:diagnosis", "@{}lrrrrcrrr@{}",
        ["Model", "IXI", "MOLAB", "MS", "MS FLAIR", "Bal.\\ acc.\\ [95\\% CI]", "Acc.", "M-F1", "W-F1"],
        rows, "scriptsize"))

    classes = ("normal", "tumor", "multiple sclerosis", "stroke", "other abnormalities", None)

    def confusion_rows(key, shown):
        r, block = run(key), []
        for i, d in enumerate(BRAIN):
            counts = collections.Counter(field(x.pred, "diagnosis_name") for x in r[d]) if d in r else None
            block.append([shown if i == 0 else "", {"ixi": "IXI", "molab": "MOLAB", "ms": "MS"}[d],
                          *([TBD] * len(classes) if counts is None else [counts[c] for c in classes])])
        return block

    rows = [row for key, shown in MODELS.items() for row in confusion_rows(key, shown)]
    save("confusion", latex("Confusion matrices: number of patients per predicted diagnosis.", "tab:confusion",
                            "@{}llrrrrrr@{}", ["Model", "Truth", "Normal", "Tumor", "MS", "Stroke", "Other", "Null"],
                            rows, "scriptsize"))

    def chest_row(key, shown):
        r = run(key)
        if "ct_rate" not in r:
            return [shown] + [TBD] * 5
        c, rows_ct = chest(r), r["ct_rate"]
        return [shown, num(c["sensitivity"]), num(c["specificity"]), num(c["bal_acc"]),
                num(field_accuracy(rows_ct, "modality", "CT")), num(field_accuracy(rows_ct, "plane", "axial"))]

    save("chest", latex("Chest CT (CT-RATE), normal versus abnormal (\\%). A constant answer has balanced accuracy "
                        "50.0.", "tab:chest", "@{}lrrrrr@{}",
                        ["Model", "Sensitivity", "Specificity", "Bal.\\ acc.", "Modality", "Plane"],
                        grouped(chest_row)))

    def descriptive_row(key, shown):
        r, flair = run(key), run(f"{key}-flair")
        cells = [None, None, None, None, None, None]
        if complete(r):
            brain = [x for d in BRAIN for x in r[d]]
            cells[0] = field_accuracy(brain, "modality", "MRI")
            cells[1] = field_accuracy(brain, "plane", "axial")
        for i, d in enumerate(BRAIN):
            if d in r:
                cells[2 + i] = field_accuracy(r[d], "specialized_sequence", expected_sequence(d))
        if "ms" in flair:
            cells[5] = field_accuracy(flair["ms"], "specialized_sequence", "FLAIR")
        return [shown, *map(num, cells)]

    save("descriptive", latex(
        "Descriptive fields (\\% correct): sanity checks, as their true value is constant within a dataset. HLIP "
        "has no output for them; BrainGemma3D's prompt names FLAIR.",
        "tab:descriptive", "@{}lrrrrrr@{}",
        ["Model", "Modality", "Plane", "Seq.\\ IXI", "Seq.\\ MOLAB", "Seq.\\ MS", "Seq.\\ FLAIR"],
        grouped(descriptive_row, (MODELS_3D, MODELS_SLICES))))

    baselines = {}

    def fine_row(key, shown):
        r, cells = run(key), [shown]
        for name, dataset in (("primary_origin", "molab"), ("ms_type", "ms")):
            f = fine(r, name) if dataset in r else None
            if f:
                baselines[name] = f["baseline"]
            cells += [num(f and f["accuracy"]), num(f and f["macro_f1"])]
        return cells

    rows = grouped(fine_row, (MODELS_3D, MODELS_SLICES))
    rows += [None, ["Majority class", num(baselines.get("primary_origin")), "--", num(baselines.get("ms_type")), "--"]]
    save("fine", latex("Fine-grained fields (\\%): primary origin (MOLAB) and MS subtype (patients with a subtype "
                       "in the schema).", "tab:fine", "@{}lrrrr@{}",
                       ["Model", "Origin acc.", "Origin M-F1", "Subtype acc.", "Subtype M-F1"], rows))

    tests, ps = {}, []
    for key in MODELS_SLICES:
        full, qna = run(key), run(f"{key}-qna")
        if complete(full) and complete(qna):
            diff, p = compare(qna, full)
            tests[key] = (diagnosis(full)["bal_acc"], diagnosis(qna)["bal_acc"], diff, len(ps))
            ps.append(p)
    adjusted = holm(ps)

    def prompting_row(key, shown):
        if key not in tests:
            return [shown] + [TBD] * 5
        full, qna, diff, i = tests[key]
        return [shown, num(full), num(qna), signed(diff), f"{adjusted[i]:.3f}",
                num(position_agreement(results, f"{key}-qna"))]

    save("prompting", latex(
        "Prompting strategies on the same models and patients: balanced accuracy of the diagnosis (\\%), the "
        "difference (Q\\&A minus full prompt) with its Holm-corrected paired permutation $p$, and the share of "
        "Q\\&A answers that survive reversing the options.",
        "tab:prompting", "@{}lrrrrr@{}", ["Model", "Full prompt", "Q\\&A", "Difference", "$p$", "Order-stable"],
        grouped(prompting_row, (MODELS_SLICES,))))

    def prior_row(key, shown):
        base, prior, cells = run(key), run(f"{key}-prior"), [shown]
        for name, dataset in (("primary_origin", "molab"), ("ms_type", "ms")):
            for r in (base, prior):
                f = fine(r, name) if dataset in r else None
                cells.append(num(f and f["accuracy"]))
        return cells

    rows = grouped(prior_row, (MODELS_SLICES,))
    rows += [None, ["Majority class", num(baselines.get("primary_origin")), "--", num(baselines.get("ms_type")), "--"]]
    save("prior", latex("Base versus prior knowledge (\\% correct). The prior prompt states the diagnosis, so only "
                        "the fields it does not state are scored.", "tab:prior", "@{}lrrrr@{}",
                        ["Model", "Origin, base", "Origin, prior", "Subtype, base", "Subtype, prior"], rows))

    def failures_row(key, shown):
        r = run(key)
        if not complete(r):
            return [shown] + [TBD] * 5
        rows_brain = [x for d in BRAIN for x in r[d]]
        kinds = collections.Counter(outcome(x.pred, x.error) for x in rows_brain)
        top = collections.Counter(str(field(x.pred, "diagnosis_name")) for x in rows_brain).most_common(1)[0][1]
        stable = num(position_agreement(results, key)) if key in QNA_NATIVE else "--"
        return [shown, num(kinds["unparseable"] / len(rows_brain)), num(kinds["abstained"] / len(rows_brain)),
                num(top / len(rows_brain)), stable, blank_answers(results, key) or TBD]

    save("failures", latex(
        "Failure modes on the brain datasets (\\% of patients): unparseable output, a parsed answer without "
        "diagnosis, the share of the most frequent diagnosis, stability under reversed options, and the diagnosis "
        "given to a blank volume (IXI / MOLAB / MS; N normal, T tumor, S stroke, O other, -- none).",
        "tab:failures", "@{}lrrrrc@{}",
        ["Model", "Unparseable", "Abstained", "Top answer", "Order-stable", "Blank"], grouped(failures_row)))

    names = list(MODELS)
    pairs = [(a, b) for a, b in itertools.combinations(names, 2) if complete(run(a)) and complete(run(b))]
    compared = [compare(run(b), run(a)) for a, b in pairs]
    by_pair = {pair: (diff, p) for pair, (diff, _), p in zip(pairs, compared, holm([p for _, p in compared]))}

    def pairs_row(j):
        cells = [f"{j + 1}. {MODELS[names[j]]}"]
        for i in range(len(names) - 1):
            if i >= j:
                cells.append("")
                continue
            found = by_pair.get((names[i], names[j]))
            cells.append(TBD if found is None else signed(found[0]) + ("*" if found[1] < 0.05 else ""))
        return cells

    save("pairs", latex(
        f"Pairwise differences in balanced accuracy (points; row minus column). * $p<0.05$, paired permutation "
        f"test, Holm-corrected over all {math.comb(len(names), 2)} pairs.", "tab:pairs",
        "@{}l" + "r" * (len(names) - 1) + "@{}", ["Model", *map(str, range(1, len(names)))],
        [pairs_row(j) for j in range(len(names))], "tiny"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", default=str(MANU / "results"))
    ap.add_argument("--out", default=str(MANU / "paper" / "tables"))
    ap.add_argument("--splits", default=str(MANU / "splits.json"))
    a = ap.parse_args()
    write_tables(a.results, a.out, load_splits(a.splits))
    print(f"tables written to {a.out}")


if __name__ == "__main__":
    main()
