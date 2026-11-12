"""The one output record format, the full-JSON prompts and value normalization shared by every model.

Standard library only.
"""
import datetime
import math
import os
import re

from common.datasets import DATASETS

PROMPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")

# Keys of output.parsed_response, copied from the colleague files (order included).
PARSED_KEYS = {
    "ixi": ["diagnosis_confidence", "diagnosis_detailed", "diagnosis_name", "icd10_code", "modality",
            "plane", "severity_confidence", "severity_score", "specialized_sequence"],
    "molab": ["modality", "specialized_sequence", "plane", "diagnosis_name", "diagnosis_detailed",
              "severity_score", "diagnosis_confidence", "severity_confidence", "primary_origin",
              "primary_origin_subtype", "origin_confidence", "subtype_confidence", "scan_name"],
    "ms": ["modality", "specialized_sequence", "plane", "diagnosis_name", "diagnosis_detailed",
           "severity_score", "diagnosis_confidence", "severity_confidence", "ms_type",
           "ms_lesion_location", "ms_lesion_burden", "ms_active_lesions", "ms_confidence"],
}
PARSED_KEYS["ct_rate"] = PARSED_KEYS["ixi"]

# Allowed values of each categorical field, as listed in the full-JSON prompts.
VOCAB = {
    "modality": ("MRI", "CT"),
    "specialized_sequence": ("FLAIR", "T1", "T2", "T1C+"),
    "plane": ("axial", "sagittal", "coronal"),
    "diagnosis_name": ("tumor", "stroke", "multiple sclerosis", "normal", "other abnormalities"),
    "diagnosis_detailed": ("glioma", "meningioma", "pituitary_tumor", "carcinoma", "germinoma", "granuloma",
                           "medulloblastoma", "neurocytoma", "papilloma", "schwannoma", "tuberculoma",
                           "ischemic", "hemorrhagic"),
    "primary_origin": ("breast", "lung-NSCLC", "lung-SCLC", "melanoma", "kidney", "colon", "bladder",
                       "unknown", "ovary", "pharynx", "bone-sarcoma", "uterus"),
    "primary_origin_subtype": ("luminal_a", "luminal_b", "triple_negative", "her2", "adenocarcinoma",
                               "epidermoid", "large_cell_carcinoma", "large_cell_endocrine"),
    "ms_type": ("SMRR", "SMSP", "SMPP"),
    "ms_lesion_location": ("periventricular", "juxtacortical", "infratentorial", "deep white matter"),
    "ms_lesion_burden": ("none", "low", "moderate", "high"),
}

SYNONYMS = {
    "blader": "bladder",
    "overy": "ovary",
    "t1 with contrast": "T1C+",
    "t1-weighted with contrast": "T1C+",
    "t1gd": "T1C+",
    "t1 post-contrast": "T1C+",
    "relapsing-remitting": "SMRR",
    "secondary progressive": "SMSP",
    "primary progressive": "SMPP",
}

NUMERIC_KEYS = ("severity_score", "diagnosis_confidence", "severity_confidence", "origin_confidence",
                "subtype_confidence", "ms_confidence")
ICD10 = re.compile(r"^[A-Z]\d{2}(\.\d+)?$")


def full_prompt(dataset):
    with open(os.path.join(PROMPT_DIR, f"full_json_{dataset}.txt"), encoding="utf-8") as f:
        return f.read()


def slice_preface(n):
    return f"The images are {n} axial slices of the same 3D scan, ordered from inferior to superior.\n"


def _categorical(key, value):
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    if text in SYNONYMS:
        text = SYNONYMS[text].lower()
    for allowed in VOCAB[key]:
        if allowed.lower() == text:
            return allowed
    return None


def _number(value):
    if isinstance(value, bool):
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(x):
        return None
    return min(1.0, max(0.0, x))


def _boolean(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    return None


def _icd10(value):
    if not isinstance(value, str):
        return None
    code = value.strip().upper()
    return code if ICD10.match(code) else None


def normalize_values(dataset, raw):
    """Every key of PARSED_KEYS[dataset] except scan_name, each a valid value or None. Idempotent."""
    values = {}
    for key in PARSED_KEYS[dataset]:
        if key == "scan_name":
            continue
        value = raw.get(key)
        if key in VOCAB:
            values[key] = _categorical(key, value)
        elif key in NUMERIC_KEYS:
            values[key] = _number(value)
        elif key == "ms_active_lesions":
            values[key] = _boolean(value)
        elif key == "icd10_code":
            values[key] = _icd10(value)
        else:
            values[key] = None
    return values


def build_record(model, dataset, mode, row_index, image_path, metadata, values, error):
    values = dict(values)
    if dataset == "molab":  # only MOLAB records keep scan_name
        values["scan_name"] = image_path
    return {
        "timestamp": datetime.datetime.now().isoformat(),
        "experiment_id": f"{model}_{DATASETS[dataset]['experiment']}",
        "api_response": {"id": "", "model": model, "prompt_tokens": 0,
                         "completion_tokens": 0, "total_tokens": 0},
        "input": {"image_path": image_path, "model_requested": model,
                  "system_prompt_type": f"{mode}_v1", "metadata": metadata},
        "output": {"error": error,
                   "parsed_response": None if error else {k: values.get(k) for k in PARSED_KEYS[dataset]},
                   "status": "error" if error else "success"},
        "processing": {"parameters": {"seed": 0, "temperature": 0}, "row_index": row_index},
    }
