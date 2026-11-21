#!/usr/bin/env python3
"""
Евалуација на сите модели (трудот, дел 6) + дијагностика зошто не работат.

Чита:  reference/*_processed.jsonl         (MedGemma, M3D-Phi3/Llama2, RadFM од март)
       results/radfm_{ixi,molab,ms}.jsonl    (RadFM, ноември)
       results/radfm_*_processed.jsonl       (RadFM, нов run_radfm.py, формат од Outputs)
       results/radfm_probe.jsonl             (ако постои: run_radfm.py probe)
Пишува: results/eval_summary.md (и го печати)

Главна метрика: дијагнозата на IXI+MOLAB+MS како една задача со 3 класи.
Секој модел што секогаш го дава истиот одговор има balanced accuracy = 33.3%,
па тоа е линијата „нема сигнал од сликата“.

Употреба:  python3 -I evaluation/evaluate.py [--outputs reference] [--results results]
           python3 -I evaluation/evaluate.py --selftest
           python3 -I evaluation/evaluate.py --split dev|eval [--results results] [--reference reference]
                                             [--splits splits.json]   -> results/report_<split>.md
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
TRUTH_SEQ = {"ixi": "t1", "molab": "t1c+", "ms": "t1"}  # MOLAB: само 26/373 се именувани T1Gd!
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


def pid(scan, dataset):
    m = re.match(r"\d{6}", scan) if dataset == "molab" else re.search(r"patient(\d{2})", scan)
    return m.group(0) if m else scan


# ---------------------------------------------------------------- вчитување
def describe(path, source):
    """Име на датотека -> (модел, збирка)."""
    n = path.name.lower()
    dataset = ("molab" if re.search(r"metasta|molab", n) else
               "ms" if re.search(r"multiple_sclerosis|_ms", n) else "ixi")
    model = {"medgemma": "medgemma-1.5", "phi3": "m3d-phi3", "llama2": "m3d-llama2",
             "radfm": "radfm (Outputs, mar)" if source == "outputs" else
                       "radfm (new)" if "processed" in n else "radfm (results, nov)"}[n.split("_")[0]]
    return model + (" +prior" if "predznaenje" in n else ""), dataset


def load_rows(path):
    rows = []
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        if "input" in r:  # формат од Outputs
            rows.append((r["input"]["image_path"], r["output"].get("parsed_response"), r["input"]["metadata"]))
        else:
            rows.append((r["scan_name"], r.get("parsed_response"), {}))
    return rows


def load_all(outputs, results):
    runs = collections.defaultdict(dict)  # модел -> збирка -> [(scan, pred или None, meta)]
    files = [(p, "outputs") for p in sorted(outputs.glob("*_processed.jsonl"))]
    files += [(results / f"radfm_{d}.jsonl", "results") for d in ("ixi", "molab", "ms")]
    files += [(p, "results") for p in sorted(results.glob("radfm_*_processed.jsonl"))]
    for path, source in files:
        if path.exists():
            model, dataset = describe(path, source)
            runs[model][dataset] = load_rows(path)
    return {m: {d: per[d] for d in TRUTH_DIAG if d in per} for m, per in runs.items()}  # редослед IXI, MOLAB, MS


def ground_truth(runs, dataset, key):
    """Ознака по пациент од metadata (сите датотеки); печати ако датотеките се разминуваат."""
    seen = collections.defaultdict(set)
    for per_ds in runs.values():
        for scan, _, meta in per_ds.get(dataset, []):
            if meta.get(key):
                seen[pid(scan, dataset)].add(meta[key])
    conflicts = {p: v for p, v in seen.items() if len(v) > 1}
    return {p: sorted(v)[0] for p, v in seen.items()}, conflicts


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


def auroc(pos, neg):
    """P(score на позитивен > score на негативен), еднаквите вредат 0.5."""
    if not pos or not neg:
        return None
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


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


# ---------------------------------------------------------------- извештај
def report(runs, probe_path):
    L = ["# Евалуација на моделите", ""]
    origin_truth, origin_conf = ground_truth(runs, "molab", "primary_origin")
    ms_truth, ms_conf = ground_truth(runs, "ms", "ms_type")

    # 1. Преглед
    L += ["## 1. Преглед: дали дијагнозата зависи од сликата?", "",
          "Pooled = IXI+MOLAB+MS како една задача со 3 класи. Константен одговор = 33.3% balanced acc.", ""]
    rows = []
    for model, per_ds in runs.items():
        all_rows = [r for d in per_ds.values() for r in d]
        parsed = sum(p is not None for _, p, _ in all_rows) / len(all_rows)
        accs = {}
        for d, rs in per_ds.items():
            accs[d] = sum(field(p, "diagnosis_name") == TRUTH_DIAG[d] for _, p, _ in rs) / len(rs)
        if set(per_ds) == set(TRUTH_DIAG):
            t = [TRUTH_DIAG[d] for d, rs in per_ds.items() for _ in rs]
            p = [field(pr, "diagnosis_name") for rs in per_ds.values() for _, pr, _ in rs]
            m = class_metrics(t, p)
            pooled = (pct(m["bal_acc"]), pct(m["macro_f1"]))
        else:
            pooled = ("—", "—")
        rows.append([model, pct(parsed), *pooled, *(pct(accs.get(d)) for d in TRUTH_DIAG)])
    L += table(["модел", "парсирано", "pooled bal.acc", "pooled macro-F1",
                "IXI acc", "MOLAB acc", "MS acc"], rows) + [""]

    # 2. Матрици на конфузија
    L += ["## 2. Матрици на конфузија (дијагноза)", ""]
    L += confusion_lines(runs, "вистина \\ модел", "редови = вистина, колони = предвидено")

    # 3. Колапс
    L += ["## 3. Колапс: колку различни одговори дава моделот?", "",
          "Ќелија = број на различни вредности · удел на најчестата (најчеста вредност).", ""]
    L += collapse_lines(runs, "поле")

    # 4. Описни полиња
    L += ["## 4. Описни полиња по збирка (проверки за пристрасност, не дискриминација)", "",
          "detailed* = вистината е секогаш null, па воздржување се брои како точно (дел 6.3).", ""]
    rows = []
    for model, per_ds in runs.items():
        for d, rs in per_ds.items():
            truth = {"modality": "mri", "plane": "axial", "specialized_sequence": TRUTH_SEQ[d],
                     "diagnosis_detailed": None}
            rows.append([model, d, *(pct(sum(field(p, f) == t for _, p, _ in rs) / len(rs))
                                     for f, t in truth.items())])
    L += table(["модел", "збирка", "modality", "plane", "sequence", "detailed*"], rows) + [""]

    # 5. Потекло (MOLAB) и тип на MS
    L += ["## 5. Тешки подзадачи", "",
          f"Вистина за потекло (MOLAB, {len(origin_truth)} пациенти): "
          f"{dict(collections.Counter(map(norm_origin, origin_truth.values())))}",
          f"Вистина за тип на MS ({len(ms_truth)} пациенти): {dict(collections.Counter(ms_truth.values()))}"
          " (трудот наведува 27/2/1 — проверете)", ""]
    for name, conf in (("потекло", origin_conf), ("MS тип", ms_conf)):
        if conf:
            L += [f"ВНИМАНИЕ: датотеките се разминуваат за {name}: {dict(list(conf.items())[:5])}", ""]

    def subtask(dataset, fname, truth_map, norm_fn, by_patient):
        rows = []
        for model, per_ds in [*runs.items(), ("константа: најчеста класа", None)]:
            if per_ds is not None and dataset not in per_ds:
                continue
            rs = per_ds[dataset] if per_ds else next(r[dataset] for r in runs.values() if dataset in r)
            pairs = [(norm_fn(truth_map[pid(s, dataset)]), field(p, fname)) for s, p, _ in rs
                     if pid(s, dataset) in truth_map]
            t = [a for a, _ in pairs]
            if per_ds is None:  # базна линија: секогаш најчестата вистинска класа
                p = [collections.Counter(t).most_common(1)[0][0]] * len(t)
            else:
                p = [b for _, b in pairs]
            cells = [model, *(pct(v) for v in class_metrics(t, p).values())]
            if by_patient:  # мнозинско гласање по пациент
                votes = collections.defaultdict(list)
                for (s, _, _), pred in zip([r for r in rs if pid(r[0], dataset) in truth_map], p):
                    votes[pid(s, dataset)].append(pred)
                pt = [norm_fn(truth_map[k]) for k in votes]
                pp = [collections.Counter(v).most_common(1)[0][0] for v in votes.values()]
                cells += [pct(v) for k, v in class_metrics(pt, pp).items() if k != "bal_acc"]
            rows.append(cells)
        return rows

    head = ["модел", "acc", "bal.acc", "macro-F1"]
    L += ["**Потекло на метастазата (MOLAB)**: скен / пациент", ""]
    L += table(head + ["пациент acc", "пациент macro-F1"],
               subtask("molab", "primary_origin", origin_truth, norm_origin, True)) + [""]
    L += ["**Тип на MS**", ""] + table(head, subtask("ms", "ms_type", ms_truth, norm_ms, False)) + [""]

    if probe_path.exists():
        L += probe_report(probe_path)
    return L


# ---------------------------------------------------------------- probe (run_radfm.py probe)
PROBE_GEN = ("P0", "P1", "P2", "yes_tumor", "yes_normal", "yes_ms", "plane")


def probe_report(path):
    recs = [json.loads(line) for line in path.open(encoding="utf-8")]
    by = {(r["scan_name"], r["image"]): r for r in recs}
    scans = sorted({r["scan_name"] for r in recs})
    images = [i for i in ("real", "zeros", "noise", "sagittal") if any(r["image"] == i for r in recs)]
    argmax = lambda r: max(r["logp"], key=r["logp"].get)
    L = ["## 6. RadFM probe: слика, прашање или модел?", "",
         f"{len(scans)} скенови × {len(images)} верзии на сликата ({len(recs)} записи).", ""]

    # а) чувствителност на сликата
    L += ["### а) Колку често одговорот се менува кога сликата е заменета", "",
          "Ако празна/шум слика дава ист одговор како вистинската, моделот не ја гледа сликата.", ""]
    rows = []
    for k in [*PROBE_GEN, "argmax logp"]:
        cells = [k]
        for img in images[1:]:
            pairs = [(by[(s, "real")], by[(s, img)]) for s in scans if (s, "real") in by and (s, img) in by]
            get = argmax if k == "argmax logp" else (lambda r, k=k: r["answers"][k])
            cells.append(pct(sum(get(a) != get(b) for a, b in pairs) / len(pairs)) if pairs else "—")
        rows.append(cells)
    L += table(["прашање", *(f"real≠{i}" for i in images[1:])], rows) + [""]

    # б) зависност од прашањето
    L += ["### б) Одговори по прашање (парсирани)", "",
          "P0 = оригинално, P1 = исти опции во друг редослед, P2 = отворено прашање. "
          "Ако P0 и P1 се разликуваат, редоследот на опциите го одредува одговорот.", ""]
    rows = []
    for img in images:
        for d in TRUTH_DIAG:
            rs = [r for r in recs if r["image"] == img and r["dataset"] == d]
            if not rs:
                continue
            top = lambda k: ", ".join(f"{v} {n}" for v, n in collections.Counter(
                str(r["parsed"][k]) for r in rs).most_common(2))
            yes = lambda k: pct(sum(r["parsed"][k] is True for r in rs) / len(rs))
            p01 = pct(sum(r["parsed"]["P0"] == r["parsed"]["P1"] for r in rs) / len(rs))
            rows.append([img, d, len(rs), top("P0"), top("P1"), top("P2"), p01,
                         yes("yes_tumor"), yes("yes_normal"), yes("yes_ms"), top("plane")])
    L += table(["слика", "збирка", "n", "P0", "P1", "P2", "P0=P1", "да тумор", "да нормален",
                "да MS", "рамнина"], rows) + [""]

    # в) веројатност на одговорите
    L += ["### в) Веројатност на фиксни одговори (без генерирање и без парсер)", "",
          "AUROC 0.5 = нема сигнал. tumor: MOLAB наспроти IXI со logp(tumor)−logp(normal); "
          "MS: MS наспроти IXI со logp(MS)−logp(normal).", ""]
    rows = []
    for img in images:
        rs = [r for r in recs if r["image"] == img]
        sc = lambda d, c: [r["logp"][c] - r["logp"]["normal"] for r in rs if r["dataset"] == d]
        t = [TRUTH_DIAG[r["dataset"]] for r in rs]
        spread = statistics.pstdev([r["logp"]["tumor"] for r in rs]) if len(rs) > 1 else 0.0
        rows.append([img, f"{auroc(sc('molab', 'tumor'), sc('ixi', 'tumor')):.2f}" if sc("molab", "tumor") else "—",
                     f"{auroc(sc('ms', 'multiple sclerosis'), sc('ixi', 'multiple sclerosis')):.2f}"
                     if sc("ms", "multiple sclerosis") else "—",
                     pct(class_metrics(t, [argmax(r) for r in rs])["bal_acc"]), f"{spread:.3f}"])
    L += table(["слика", "AUROC tumor", "AUROC MS", "argmax bal.acc", "std logp(tumor)"], rows) + [""]
    gaps = []
    for c in ("normal", "tumor", "multiple sclerosis", "stroke"):
        g = [by[(s, "real")]["logp"][c] - by[(s, "zeros")]["logp"][c]
             for s in scans if (s, "real") in by and (s, "zeros") in by]
        gaps.append(f"{statistics.mean(g):+.3f}" if g else "—")
    L += ["Просечна разлика logp(вистинска) − logp(празна слика): "
          + ", ".join(f"{c} {g}" for c, g in zip(("normal", "tumor", "MS", "stroke"), gaps)), ""]
    if all(r["image"] != "real" or len(set(r["logp"].values())) == 1 for r in recs):
        L += ["ВНИМАНИЕ: сите logp се исти — оценувањето е скршено, не моделот.", ""]

    L += ["### Како да се чита", "",
          "| резултат | значење |", "|---|---|",
          "| real≠zeros ≈ 0% | сликата не се користи: грешка во влезот или слеп модел |",
          "| P0≠P1, а real≈zeros | одговорот го одредува прашањето (јазичен приор) |",
          "| AUROC > ~0.7, а генерирањето е колапсирано | моделот има сигнал, го губат прашањето/парсерот → "
          "оценувај со веројатност |",
          "| рамнината се менува со sagittal | ориентацијата во load_volume не одговара на тренингот |",
          "| сликата влијае, прашањето не, AUROC ≈ 0.5 | вистинско ограничување на моделот (негативен резултат) |",
          ""]
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
        model, dataset = describe(path, "outputs")
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
    assert auroc([2, 3], [0, 1]) == 1.0 and auroc([0], [1]) == 0.0 and auroc([1], [1]) == 0.5
    assert norm_diag("Brain tumor") == "tumor" and norm_diag("Normal") == "normal" and norm_diag("Brain") is None
    assert norm_diag("multiple_sclerosis") == "multiple sclerosis"
    assert norm_ms("SMRR") == "SMRR" and norm_ms("secondary progressive") == "SMSP" and norm_ms("null") is None
    assert norm_origin("overy") == "ovary" and norm_origin("lung-NSCLC") == "lung-nsclc"
    assert pid("010005N__010005_x_img.nii", "molab") == "010005" and pid("patient07_T1W.nii.gz", "ms") == "patient07" == pid("patient07_T1.nii.gz", "ms")
    assert describe(Path("medgemma_15_predznaenje_brain_metastases_processed.jsonl"), "outputs") == \
        ("medgemma-1.5 +prior", "molab")
    assert describe(Path("radfm_ms.jsonl"), "results") == ("radfm (results, nov)", "ms")
    assert describe(Path("radfm_healthy_processed.jsonl"), "results") == ("radfm (new)", "ixi")
    assert describe(Path("radfm_brain_metastases_processed.jsonl"), "results") == ("radfm (new)", "molab")
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default=str(MANU / "reference"))
    ap.add_argument("--results", default=str(MANU / "results"))
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--split", choices=("dev", "eval"), help="per-split report of the new model runs")
    ap.add_argument("--reference", help="colleagues' files (default: --outputs)")
    ap.add_argument("--splits", default=str(MANU / "splits.json"))
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    results = Path(a.results)
    if a.split:
        runs = load_split(results, Path(a.reference or a.outputs), a.split, load_splits(a.splits))
        if not runs:
            sys.exit(f"No runs for split {a.split} in {a.results} or {a.reference or a.outputs}")
        text = "\n".join(split_report(runs, results, a.split)) + "\n"
        results.mkdir(parents=True, exist_ok=True)
        (results / f"report_{a.split}.md").write_text(text, encoding="utf-8")
        print(text)
        return
    runs = load_all(Path(a.outputs), results)
    if not runs:
        sys.exit(f"Нема излези во {a.outputs} или {a.results}")
    text = "\n".join(report(runs, results / "radfm_probe.jsonl")) + "\n"
    (results / "eval_summary.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
