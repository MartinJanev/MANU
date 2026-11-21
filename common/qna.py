"""Per-field multiple-choice questions for models that cannot follow a long JSON prompt.

Question and option texts are the M3D-family column of the "Шаблони по поле" table in the model notes. Each field
is one four-option question; every question is asked twice (options in order and reversed) to expose position bias.
Standard library only; the model is reached through the `AskFn` callable, so tests need no weights.
"""
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from common.parse import parse_choice

ORIGINAL = (0, 1, 2, 3)
REVERSED = (3, 2, 1, 0)

# question text -> (raw answer, probabilities of the four letters in displayed order, or None)
AskFn = Callable[[str], Tuple[str, Optional[List[float]]]]


@dataclass(frozen=True)
class Question:
    field: str
    text: str
    options: Tuple[str, ...]
    values: tuple  # values[i] is the schema value for options[i]; None = "Other / Cannot be determined"


_TUMOR = "What type of tumor is this?"

BANK: Dict[str, object] = {
    "modality": Question("modality", "What is the modality of this image?",
                         ("MRI", "CT", "X-ray", "Ultrasound"), ("MRI", "CT", None, None)),
    "specialized_sequence": Question("specialized_sequence", "Which MRI sequence is shown in this image?",
                                     ("T1-weighted", "T1-weighted with contrast", "T2-weighted", "FLAIR"),
                                     ("T1", "T1C+", "T2", "FLAIR")),
    "plane": Question("plane", "Which plane is displayed in the image?",
                      ("Axial", "Sagittal", "Coronal", "Oblique"), ("axial", "sagittal", "coronal", None)),
    "diagnosis_name": Question("diagnosis_name", "What is the main finding in this brain image?",
                               ("Normal", "Tumor", "Stroke", "Multiple sclerosis"),
                               ("normal", "tumor", "stroke", "multiple sclerosis")),
    # Tumor: R1 -> R2 -> R3, each deeper level only on "Other". R1 "Metastasis" is registered as `carcinoma`.
    "diagnosis_detailed": [
        Question("diagnosis_detailed", _TUMOR, ("Glioma", "Meningioma", "Metastasis", "Other"),
                 ("glioma", "meningioma", "carcinoma", None)),
        Question("diagnosis_detailed", _TUMOR, ("Pituitary tumor", "Schwannoma", "Medulloblastoma", "Other"),
                 ("pituitary_tumor", "schwannoma", "medulloblastoma", None)),
        Question("diagnosis_detailed", _TUMOR, ("Germinoma", "Neurocytoma", "Papilloma", "Granuloma"),
                 ("germinoma", "neurocytoma", "papilloma", "granuloma")),
    ],
    "diagnosis_detailed_stroke": Question("diagnosis_detailed", "What type of stroke is this?",
                                          ("Ischemic", "Hemorrhagic", "Both", "Cannot be determined"),
                                          ("ischemic", "hemorrhagic", None, None)),
    "severity_score": Question("severity_score", "How severe is the abnormality?",
                               ("None", "Mild", "Moderate", "Severe"), (0.0, 0.25, 0.5, 0.75)),
    # Metastasis origin: R1 -> R2 -> R3 -> R4 on "Other"; "lung" leads to the lung-type question instead.
    "primary_origin": [
        Question("primary_origin",
                 "If this is a brain metastasis, where is the primary cancer most likely located?",
                 ("Lung", "Breast", "Melanoma", "Other"), ("lung", "breast", "melanoma", None)),
        Question("primary_origin",
                 "If this is a brain metastasis, where is the primary cancer most likely located?",
                 ("Kidney", "Ovary", "Uterus", "Other"), ("kidney", "ovary", "uterus", None)),
        Question("primary_origin",
                 "If this is a brain metastasis, where is the primary cancer most likely located?",
                 ("Colon", "Bladder", "Pharynx", "Other"), ("colon", "bladder", "pharynx", None)),
        Question("primary_origin",
                 "If this is a brain metastasis, where is the primary cancer most likely located?",
                 ("Bone sarcoma", "Unknown primary", "Other", "Cannot be determined"),
                 ("bone-sarcoma", "unknown", "unknown", "unknown")),
    ],
    "lung_type": Question("primary_origin", "Is the primary lung cancer small cell or non-small cell?",
                          ("Non-small cell lung cancer", "Small cell lung cancer", "Unknown", "Not lung"),
                          ("lung-NSCLC", "lung-SCLC", "unknown", "unknown")),
    "ms_type": Question("ms_type", "Which type of multiple sclerosis is this?",
                        ("Relapsing-remitting", "Secondary progressive", "Primary progressive",
                         "Cannot be determined"), ("SMRR", "SMSP", "SMPP", None)),
    "ms_lesion_location": Question("ms_lesion_location", "Where are the lesions mainly located?",
                                   ("Periventricular", "Juxtacortical", "Infratentorial", "Deep white matter"),
                                   ("periventricular", "juxtacortical", "infratentorial", "deep white matter")),
    "ms_lesion_burden": Question("ms_lesion_burden", "How extensive is the lesion burden?",
                                 ("None", "Low", "Moderate", "High"), ("none", "low", "moderate", "high")),
}


def format_mcq(q: Question, style: str, order: Tuple[int, ...]) -> str:
    """The question with its options shown in `order`; style "A" gives "A. x B. y", style "a" gives "(a) x (b) y"."""
    shown = [q.options[i] for i in order]
    if style == "A":
        return f"{q.text} Choices: " + " ".join(f"{'ABCD'[j]}. {o}" for j, o in enumerate(shown))
    return f"{q.text} " + " ".join(f"({'abcd'[j]}) {o}" for j, o in enumerate(shown))


def _ask_once(q: Question, ask_fn: AskFn, style: str, order: Tuple[int, ...]) -> dict:
    """One ask; `index`, `probs` are in the original option order (the answer is mapped back through `order`)."""
    raw, probs = ask_fn(format_mcq(q, style, order))
    shown = [q.options[i] for i in order]
    picked = parse_choice(raw, shown, style)
    index = None if picked is None else order[picked]
    # With options no answer can contain, parse_choice succeeds only through a leading letter.
    method = "none" if picked is None else ("letter" if parse_choice(raw, ["\0"] * 4, style) is not None
                                            else "option_text")
    mapped = None
    if probs is not None:
        mapped = [0.0] * 4
        for j, p in enumerate(probs):
            mapped[order[j]] = p
    return {"order": list(order), "raw": raw, "index": index, "method": method, "probs": mapped}


def ask(q: Question, ask_fn: AskFn, style: str) -> dict:
    """Ask in original and reversed option order. value/confidence/method come from the original-order ask."""
    first = _ask_once(q, ask_fn, style, ORIGINAL)
    second = _ask_once(q, ask_fn, style, REVERSED)
    index = first["index"]
    confidence = None
    if index is not None and first["probs"] is not None:
        confidence = first["probs"][index]
    return {
        "value": None if index is None else q.values[index],
        "confidence": confidence,
        "agree": index is not None and index == second["index"],
        "answers": [first, second],
        "method": first["method"],
    }


def _walk(levels: List[Question], ask_fn: AskFn, style: str) -> dict:
    """Ask levels in turn, descending only while the model chose the level's "Other" option."""
    results = []
    for q in levels:
        result = ask(q, ask_fn, style)
        results.append(result)
        if result["value"] is not None or result["method"] == "none":
            break
    final = dict(results[-1])
    final["levels"] = results
    return final


def interrogate(dataset: str, ask_fn: AskFn, style: str) -> Tuple[dict, dict]:
    """(raw values, log per field) from the per-field questions that apply to `dataset`."""
    values: dict = {}
    log: dict = {}

    def put(field, result):
        values[field] = result["value"]
        log[field] = result

    put("modality", ask(BANK["modality"], ask_fn, style))
    values["specialized_sequence"] = None
    if values["modality"] != "CT":
        put("specialized_sequence", ask(BANK["specialized_sequence"], ask_fn, style))
    put("plane", ask(BANK["plane"], ask_fn, style))

    diagnosis = ask(BANK["diagnosis_name"], ask_fn, style)
    put("diagnosis_name", diagnosis)
    values["diagnosis_confidence"] = diagnosis["confidence"]

    values["diagnosis_detailed"] = None
    if values["diagnosis_name"] == "tumor":
        put("diagnosis_detailed", _walk(BANK["diagnosis_detailed"], ask_fn, style))
    elif values["diagnosis_name"] == "stroke":
        put("diagnosis_detailed", ask(BANK["diagnosis_detailed_stroke"], ask_fn, style))

    severity = ask(BANK["severity_score"], ask_fn, style)
    put("severity_score", severity)
    values["severity_confidence"] = severity["confidence"]

    if dataset == "molab":
        origin = _walk(BANK["primary_origin"], ask_fn, style)
        if origin["value"] == "lung":
            origin["levels"].append(ask(BANK["lung_type"], ask_fn, style))
            origin = dict(origin["levels"][-1], levels=origin["levels"])
        put("primary_origin", origin)
        confidences = [r["confidence"] for r in origin["levels"]]
        values["origin_confidence"] = None if None in confidences else math.prod(confidences, start=1.0)
        values["primary_origin_subtype"] = None
        values["subtype_confidence"] = None
    elif dataset == "ms":
        for field in ("ms_type", "ms_lesion_location", "ms_lesion_burden"):
            put(field, ask(BANK[field], ask_fn, style))
        values["ms_active_lesions"] = None  # open_ms_data has no contrast, so the question is not asked
        probs = diagnosis["answers"][0]["probs"]
        values["ms_confidence"] = None if probs is None else probs[3]  # the "Multiple sclerosis" option
    values["icd10_code"] = None
    return values, log
