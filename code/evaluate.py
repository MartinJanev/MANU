#!/usr/bin/env python3
"""
Евалуација на сите модели (трудот, дел 6) + дијагностика зошто не работат.

Чита:  ../Outputs/*_processed.jsonl         (MedGemma, M3D-Phi3/Llama2, RadFM од март)
       results/radfm_{ixi,molab,ms}.jsonl    (RadFM, ноември)
       results/radfm_*_processed.jsonl       (RadFM, нов run_radfm.py, формат од Outputs)
       results/radfm_probe.jsonl             (ако постои: run_radfm.py probe)
Пишува: results/eval_summary.md (и го печати)

Главна метрика: дијагнозата на IXI+MOLAB+MS како една задача со 3 класи.
Секој модел што секогаш го дава истиот одговор има balanced accuracy = 33.3%,
па тоа е линијата „нема сигнал од сликата“.

Употреба:  python3 -I code/evaluate.py [--outputs ../Outputs] [--results results]
           python3 -I code/evaluate.py --selftest
"""
import argparse
import collections
import json
import re
import statistics
import sys
from pathlib import Path

MANU = Path(__file__).resolve().parent.parent
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
    for model, per_ds in runs.items():
        counts = {d: collections.Counter(str(field(p, "diagnosis_name")) for _, p, _ in rs)
                  for d, rs in per_ds.items()}
        cols = sorted({c for cnt in counts.values() for c in cnt})
        L += [f"**{model}** (редови = вистина, колони = предвидено)", ""]
        L += table(["вистина \\ модел", *cols],
                   [[f"{d} ({TRUTH_DIAG[d]})", *(counts[d][c] for c in cols)] for d in counts]) + [""]

    # 3. Колапс
    L += ["## 3. Колапс: колку различни одговори дава моделот?", "",
          "Ќелија = број на различни вредности · удел на најчестата (најчеста вредност).", ""]
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
        L += [f"**{model}**", ""] + table(["поле", "IXI", "MOLAB", "MS"], rows) + [""]

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
    ap.add_argument("--outputs", default=str(MANU.parent / "Outputs"))
    ap.add_argument("--results", default=str(MANU / "results"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    results = Path(a.results)
    runs = load_all(Path(a.outputs), results)
    if not runs:
        sys.exit(f"Нема излези во {a.outputs} или {a.results}")
    text = "\n".join(report(runs, results / "radfm_probe.jsonl")) + "\n"
    (results / "eval_summary.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
