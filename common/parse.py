"""Parsing of model answers: JSON objects, multiple-choice letters, and keyword extraction from reports.

Standard library only. Values returned here are raw; common.schema.normalize_values validates them.
"""
import collections
import json
import re

from common.schema import PARSED_KEYS

_THINK = re.compile(r"<think>.*?</think>", re.S)
_ANSWER = re.compile(r"<answer>(.*?)(?:</answer>|$)", re.S)
_FENCE = re.compile(r"```[a-zA-Z]*")
_TRAILING_COMMA = re.compile(r",\s*([}\]])")


def _first_balanced_object(text):
    """The first {...} with balanced braces (braces inside strings ignored), or None."""
    start = text.find("{")
    if start == -1:
        return None
    depth, in_string, escaped = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif c == "\\":
                escaped = True
            elif c == '"':
                in_string = False
        elif c == '"':
            in_string = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return text[start:]  # unbalanced (truncated answer): let json.loads reject it


def _following_answers(rest):
    """The further complete answer objects after the first one (schema echoes and broken ones left out)."""
    found = []
    while True:
        candidate = _first_balanced_object(rest)
        if candidate is None:
            return found
        rest = rest[rest.find(candidate) + len(candidate):]
        try:
            obj = json.loads(_TRAILING_COMMA.sub(r"\1", candidate))
        except ValueError:
            continue
        name = obj.get("diagnosis_name")
        if "diagnosis_name" in obj and not (isinstance(name, str) and "|" in name):
            found.append(obj)


def _majority(objects):
    """Per field the value of a strict majority of the objects, else None (abstain)."""
    merged = {}
    for key in objects[0]:
        votes = collections.Counter(json.dumps(o.get(key), sort_keys=True) for o in objects)
        value, n = votes.most_common(1)[0]
        merged[key] = json.loads(value) if n > len(objects) / 2 else None
    return merged


def parse_json_answer(text):
    """(object or None, problems) with problems a subset of {"no_json", "invalid_json", "echo:<field>"}."""
    text = _THINK.sub("", text)
    match = _ANSWER.search(text)
    if match:
        text = match.group(1)
    text = _FENCE.sub("", text).replace("```", "")
    candidate = _first_balanced_object(text)
    if candidate is None:
        return None, ["no_json"]
    try:
        obj = json.loads(_TRAILING_COMMA.sub(r"\1", candidate))
    except ValueError:
        return None, ["invalid_json"]
    problems = []
    several = [obj, *_following_answers(text[text.find(candidate) + len(candidate):])]
    if len(several) > 1 and "diagnosis_name" in obj:  # QoQ-Med writes one object per slice
        obj = _majority(several)
        problems.append("multiple_objects")
    for key, value in obj.items():
        if isinstance(value, str) and ("<" in value or "|" in value):
            obj[key] = None
            problems.append(f"echo:{key}")
    return obj, problems


def parse_choice(text, options, style):
    """Index of the chosen option, or None (echo of every option, no answer, or an ambiguous answer)."""
    low = text.lower()
    if all(o.lower() in low for o in options):
        return None
    hits = [i for i, o in enumerate(options) if o.lower() in low]
    index = None
    if style == "A":
        m = re.match(r"\s*\(?([A-D])(?:([.):])|\s*$|\s)", text)
        if m:
            letter = ord(m.group(1)) - ord("A")
            punctuated = m.group(2) is not None or not text[m.end(1):].strip()
            if punctuated or not hits or hits == [letter]:
                index = letter
            elif len(hits) == 1:
                return None  # "A coronal slice": the article A, answer is another option
    else:
        m = re.match(r"\s*(?:\(([a-d])\)|([a-d])\.?\s*$)", text)
        if m:
            index = ord((m.group(1) or m.group(2))) - ord("a")
    if index is not None and index < len(options):
        return index
    return hits[0] if len(hits) == 1 else None


# ---------------------------------------------------------------------------
# Report -> schema values: one fixed keyword extractor shared by every report model.
# ---------------------------------------------------------------------------
NEGATION = re.compile(r"\b(no|without|negative for|absence of|not)\b")
TUMOR = re.compile(r"metasta|tumou?r|neoplas|\bmass\b|glioma|meningioma|glioblastoma|lymphoma")
MS = re.compile(r"multiple sclerosis|demyelinat")
STROKE = re.compile(r"infarct|isch(a)?emi|stroke|h(a)?emorrhag")
NORMAL = re.compile(r"no (acute |significant )?(intracranial )?abnormalit|unremarkable|within normal limits"
                    r"|normal (brain|study|mri|head (ct|mri))|parenchyma (appears|is) normal")
OTHER = re.compile(r"lesion|hyperintens|enhanc|abnormal")
DETAILED_TUMOR = [("glioma", "glioma"), ("meningioma", "meningioma"), ("pituitary", "pituitary_tumor"),
                  ("schwannoma", "schwannoma"), ("medulloblastoma", "medulloblastoma"),
                  ("germinoma", "germinoma"), ("neurocytoma", "neurocytoma"), ("papilloma", "papilloma"),
                  ("granuloma", "granuloma"), ("tuberculoma", "tuberculoma"),
                  (r"carcinoma|metasta", "carcinoma")]
ORIGINS = [(r"breast", "breast"), (r"melanoma", "melanoma"), (r"kidney|renal", "kidney"),
           (r"colon|colorectal", "colon"), (r"bladder", "bladder"), (r"ovar", "ovary"),
           (r"uter|endometri", "uterus"), (r"pharyn", "pharynx"), (r"sarcoma", "bone-sarcoma")]
MS_TYPES = [("relapsing", "SMRR"), ("secondary progressive", "SMSP"), ("primary progressive", "SMPP")]
MS_LOCATIONS = ["periventricular", "juxtacortical", "infratentorial", "deep white matter"]


def _first(rules, text):
    for pattern, value in rules:
        if re.search(pattern, text):
            return value
    return None


def extract_from_report(report, dataset):
    """Raw schema values from a free-text report; confidences and severity stay None."""
    sentences = [s.strip().lower() for s in re.split(r"[.;\n]", report) if s.strip()]
    findings = " . ".join(s for s in sentences if not NEGATION.search(s))  # negated sentences carry no finding
    everything = " . ".join(sentences)
    values = {k: None for k in PARSED_KEYS[dataset] if k != "scan_name"}

    if TUMOR.search(findings):
        name = "tumor"
    elif MS.search(findings):
        name = "multiple sclerosis"
    elif STROKE.search(findings):
        name = "stroke"
    elif NORMAL.search(everything):
        name = "normal"
    elif OTHER.search(findings):
        name = "other abnormalities"
    else:
        name = None
    values["diagnosis_name"] = name
    if name == "tumor":
        values["diagnosis_detailed"] = _first(DETAILED_TUMOR, findings)
    elif name == "stroke":
        values["diagnosis_detailed"] = "hemorrhagic" if re.search(r"h(a)?emorrhag", findings) else "ischemic"

    if re.search(r"\bMRI?\b", report) or "magnetic resonance" in everything:
        values["modality"] = "MRI"
    elif re.search(r"\bCT\b", report) or "computed tomography" in everything:
        values["modality"] = "CT"
    values["specialized_sequence"] = _first(
        [(r"\bflair\b", "FLAIR"), (r"post-contrast|gadolinium|\bt1c\b", "T1C+"), (r"\bt2\b", "T2"),
         (r"\bt1\b", "T1")], everything)
    values["plane"] = _first([(r"\baxial\b", "axial"), (r"\bsagittal\b", "sagittal"),
                              (r"\bcoronal\b", "coronal")], everything)

    if dataset == "molab":
        if "lung" in findings:
            values["primary_origin"] = "lung-SCLC" if "small cell" in findings else "lung-NSCLC"
        else:
            values["primary_origin"] = _first(ORIGINS, findings)
    if dataset == "ms":
        values["ms_type"] = _first(MS_TYPES, findings)
        values["ms_lesion_location"] = next((loc for loc in MS_LOCATIONS if loc in findings), None)
    return values
