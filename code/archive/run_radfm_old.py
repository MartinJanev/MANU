"""
RadFM inference на GPGPU кластерот: Q&A по поле, како во трудот (дел 5.1).

За секој 3D волумен RadFM добива посебно прашање на природен јазик за секое поле
од шемата. Суровите одговори се зачувуваат (answers), а потоа се парсираат со
клучни зборови во полињата на шемата (parsed_response).

Влез:  /data/...                  сурови NIfTI (без претходна конверзија)
       /radfm                     RadFM/Quick_demo: Model/, Language_files/, pytorch_model.bin
       /hf_cache                  Med-KEBERT (RadFM го вчитува при старт)
Излез: /results/radfm_<збирка>.jsonl   еден ред по скен; при ново пуштање продолжува каде застанал

Употреба:  python3 /code/run_radfm.py ixi|molab|ms
"""
import gc
import glob
import json
import os
import re
import sys
import time

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

RADFM_DIR = "/radfm"
DATA_DIR = "/data"
OUT_DIR = "/results"

# Збирки: шаблон за датотеките во /data
DATASETS = {
    "ixi": "ixi/T1/*.nii.gz",                                       # 581 здрави
    "molab": "molab/nifti/*_img.nii",                               # 373 метастази
    "ms": "open_ms_data/cross_sectional/raw/patient*/T1W.nii.gz",   # 30 MS (raw T1)
}

# Влез во големината на која RadFM е трениран: 512x512 во рамнина, длабочина најмногу 64
# (radiopaedia.py ги зумира 3D волумените над 64 пресеци на 64). Позициските вградувања
# се научени табели 16x16x128, индексирани со чекор 128 // (D/4): при D=160 се користат
# редови што при тренинг (D <= 64) никогаш не се виделе. Трудот: 256x256x160.
TARGET_H, TARGET_W, TARGET_D = 512, 512, 64
MAX_NEW_TOKENS = 64
EOS_ID, PAD_ID = 2, 0  # LLaMA </s> = 2 (config.json во Language_files погрешно наведува 1)

# ---------------------------------------------------------------- прашања
Q = {
    "modality": "What modality is used to take this image?",
    "specialized_sequence": "Which MRI sequence is used in this image: T1-weighted, "
                            "T1-weighted with contrast, T2-weighted, or FLAIR?",
    "plane": "In which plane is this image scanned: axial, sagittal, or coronal?",
    "diagnosis_name": "What illness can you diagnose from this image, such as tumor, "
                      "stroke, multiple sclerosis, or normal?",
    "diagnosis_detailed": "What type of lesion is seen in this image, for example glioma, "
                          "meningioma, pituitary tumor, carcinoma, schwannoma, "
                          "ischemic stroke, or hemorrhagic stroke?",
    "severity_score": "How severe is the abnormality in this image, on a scale "
                      "from 0 (none) to 1 (very severe)?",
    "diagnosis_confidence": "How confident are you that this image shows {label}? "
                            "Answer with a number between 0 and 1.",
    "severity_confidence": "How confident are you in your assessment of the severity? "
                           "Answer with a number between 0 and 1.",
}
Q_MOLAB = {
    "primary_origin": "If this is a metastatic brain tumor, where is the primary cancer most "
                      "likely located: lung, breast, skin, kidney, colon, bladder, ovary, "
                      "uterus, pharynx, or bone?",
    "lung_type": "Is the primary lung cancer small cell lung cancer or non-small cell lung cancer?",
    "origin_confidence": "How confident are you that the primary cancer is in the {organ}? "
                         "Answer with a number between 0 and 1.",
}
Q_MS = {
    "ms_type": "Which type of multiple sclerosis is this: relapsing-remitting, "
               "secondary progressive, or primary progressive?",
    "lesion_location": "Where are the lesions located in the brain?",
    "lesion_burden": "How many lesions are visible: none, few, several, or many?",
    "active_lesions": "Are there any active lesions in this image?",
    "ms_confidence": "How confident are you that this image shows multiple sclerosis? "
                     "Answer with a number between 0 and 1.",
}
LABEL_PHRASE = {
    "tumor": "a tumor", "stroke": "a stroke", "multiple sclerosis": "multiple sclerosis",
    "normal": "a normal brain", "other abnormalities": "an abnormality",
}
ORGAN_PHRASE = {
    "lung-NSCLC": "lung", "lung-SCLC": "lung", "breast": "breast", "melanoma": "skin",
    "kidney": "kidney", "colon": "colon", "bladder": "bladder", "ovary": "ovary",
    "uterus": "uterus", "pharynx": "pharynx", "bone-sarcoma": "bone",
}

# ---------------------------------------------------------------- парсирање
NEGATION = re.compile(r"\b(no|not|without|negative for|absence of|free of)\b[\w\s,-]{0,20}$")


def found(pattern, text):
    """True ако шаблонот се појавува во текстот и не е негиран (пр. 'no tumor')."""
    text = (text or "").lower()
    for m in re.finditer(pattern, text):
        if not NEGATION.search(text[max(0, m.start() - 30):m.start()]):
            return True
    return False


def first_match(pairs, text):
    for value, pattern in pairs:
        if found(pattern, text):
            return value
    return None


def parse_modality(t):
    return first_match([("MRI", r"\bmri?\b|magnetic resonance|\bt1|\bt2|flair"),
                        ("CT", r"\bct\b|computed tomograph")], t)


def parse_sequence(t):
    return first_match([("FLAIR", r"flair"),
                        ("T1C+", r"(with|post)[- ]?contrast|contrast[- ]enhanced|gadolinium|\bgd\b|\bt1c"),
                        ("T2", r"\bt2"),
                        ("T1", r"\bt1")], t)


def parse_plane(t):
    return first_match([("axial", r"axial|transverse"),
                        ("sagittal", r"sagittal"),
                        ("coronal", r"coronal")], t)


def parse_diagnosis(t):
    if not (t or "").strip():
        return None
    label = first_match([
        ("multiple sclerosis", r"multiple sclerosis|\bms\b|demyelinat"),
        ("stroke", r"stroke|infarct|isch(a)?emi|h(a)?emorrhag|bleed"),
        ("tumor", r"tumou?r|neoplas|\bmass\b|metasta|glioma|meningioma|carcinoma|lymphoma|"
                  r"cancer|malignan|adenoma|schwannoma|blastoma"),
        ("normal", r"\bnormal\b|unremarkable|healthy|no (significant )?abnormalit"),
    ], t)
    return label or "other abnormalities"


def parse_detailed(t, diagnosis):
    if diagnosis not in ("tumor", "stroke"):
        return None
    return first_match([
        ("pituitary_tumor", r"pituitar|adenoma"),
        ("medulloblastoma", r"medulloblastoma"),
        ("glioma", r"glioma|glioblastoma|astrocytoma|oligodendroglioma|ependymoma|\bgbm\b"),
        ("meningioma", r"meningioma"),
        ("germinoma", r"germinoma"),
        ("neurocytoma", r"neurocytoma"),
        ("papilloma", r"papilloma"),
        ("schwannoma", r"schwannoma|acoustic neuroma|vestibular"),
        ("tuberculoma", r"tuberculoma"),
        ("granuloma", r"granuloma"),
        ("carcinoma", r"carcinoma"),
        ("hemorrhagic", r"h(a)?emorrhag|bleed|h(a)?ematoma"),
        ("ischemic", r"isch(a)?emi|infarct"),
    ], t)


SCORE_WORDS = [(r"very severe|extreme", 1.0), (r"severe|high|large|extensive", 0.75),
               (r"moderate|medium", 0.5), (r"mild|slight|low|small|minimal", 0.25),
               (r"\bnone\b|\bno\b|\bnormal\b", 0.0)]


# RadFM често ја повторува скалата од прашањето наместо оцена:
# "0 means no confidence and 1 means ...", "If you are completely confident, choose 1. 0 indicates ..."
SCALE_ECHO = re.compile(r"(?<![\d.])[01](?:\.0+)?\s*(?:means|indicat|would|-\s)"
                        r"|\b(?:if(?: it was)?|between|choose|answer)\s+[01]\b")


def parse_score(t):
    """Број во [0, 1] (или процент / "N out of M"); инаку зборови (mild, moderate, severe ...).
    None ако одговорот само ја опишува скалата."""
    t = (t or "").lower().strip()
    m = re.match(r"\d+(?:\.\d+)?", t)  # почнува со оцена: "0.5 (moderate) - ..."
    if m and not SCALE_ECHO.match(t) and 0 <= float(m.group()) <= 1:
        return round(float(m.group()), 4)
    if SCALE_ECHO.search(t):
        return None
    t = re.sub(r"grade\s*\d+", "", t)  # "grade 1" е степен, не оцена во [0, 1]
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:/|out of)\s*(\d+)", t)  # пр. "8 out of 10"
    if m and float(m.group(2)) > 0:
        v = float(m.group(1)) / float(m.group(2))
        return round(v, 4) if 0 <= v <= 1 else None
    m = re.search(r"(\d+(?:\.\d+)?)\s*(%)?", t)
    if m:
        v = float(m.group(1))
        if m.group(2):  # без % број > 1 е на непозната скала ("worth 3 points") -> зборови
            v /= 100
        if 0 <= v <= 1:
            return round(v, 4)
    for pattern, value in SCORE_WORDS:
        if re.search(pattern, t):
            return value
    return None


def parse_origin(t):
    return first_match([
        ("lung-NSCLC", r"non[- ]?small[- ]?cell|nsclc"),
        ("lung-SCLC", r"small[- ]?cell|\bsclc\b"),
        ("lung", r"lung|pulmonar|bronch"),
        ("breast", r"breast|mammary"),
        ("melanoma", r"melanoma|skin"),
        ("kidney", r"kidney|renal"),
        ("colon", r"colon|colorectal|rectal|rectum|bowel"),
        ("bladder", r"bladder|urothelial"),
        ("ovary", r"ovar"),
        ("uterus", r"uter|endometri|cervi"),
        ("pharynx", r"pharyn"),
        ("bone-sarcoma", r"bone|sarcoma"),
        ("unknown", r"unknown|unclear|uncertain|not sure"),
    ], t)


def parse_lung_type(t):
    return first_match([("lung-NSCLC", r"non[- ]?small|nsclc|adenocarcinoma|squamous|large[- ]cell"),
                        ("lung-SCLC", r"small[- ]?cell|\bsclc\b")], t)


def parse_ms_type(t):
    return first_match([("SMRR", r"relaps|remit|\brrms\b"),
                        ("SMSP", r"secondary|\bspms\b"),
                        ("SMPP", r"primary|\bppms\b")], t)


def parse_burden(t):
    t = (t or "").lower()
    for value, pattern in [("none", r"\bnone\b|\bno\b"), ("many", r"many|numerous|multiple|extensive"),
                           ("several", r"several|moderate|some"), ("few", r"few|single|one|small")]:
        if re.search(pattern, t):
            return value
    return None


def parse_yes_no(t):
    t = (t or "").lower()
    if re.search(r"\bno\b|\bnone\b|absent|\bnot\b", t):
        return False
    if re.search(r"\byes\b|present|active|enhanc", t):
        return True
    return None


# ---------------------------------------------------------------- модел
def get_tokenizer(tokenizer_path, max_img_size=100, image_num=32):
    """Истo како get_tokenizer во RadFM/Quick_demo/test.py."""
    from transformers import LlamaTokenizer
    image_padding_tokens = []
    text_tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
    special_token = {"additional_special_tokens": ["<image>", "</image>"]}
    for i in range(max_img_size):
        image_padding_token = ""
        for j in range(image_num):
            image_token = "<image" + str(i * image_num + j) + ">"
            image_padding_token = image_padding_token + image_token
            special_token["additional_special_tokens"].append(image_token)
        image_padding_tokens.append(image_padding_token)
        text_tokenizer.add_special_tokens(special_token)
        text_tokenizer.pad_token_id = 0
        text_tokenizer.bos_token_id = 1
        text_tokenizer.eos_token_id = 2
    return text_tokenizer, image_padding_tokens


def load_model():
    sys.path.insert(0, RADFM_DIR)
    from transformers import LlamaConfig, LlamaForCausalLM
    import Model.RadFM.multimodality_model as mm

    class LlamaFromConfig:
        """Language_files има само config.json (без тежини), па LlamaForCausalLM.from_pretrained
        би паднал. LLaMA се гради од конфигурацијата; сите тежини доаѓаат од pytorch_model.bin."""
        @staticmethod
        def from_pretrained(path, **kwargs):
            return LlamaForCausalLM(LlamaConfig.from_pretrained(path))

    mm.LlamaForCausalLM = LlamaFromConfig

    lang_path = os.path.join(RADFM_DIR, "Language_files")
    tokenizer, image_padding_tokens = get_tokenizer(lang_path)

    t0 = time.time()
    model = mm.MultiLLaMAForCausalLM(lang_model_path=lang_path)
    model = model.to("cuda")
    print(f"Модел изграден за {time.time() - t0:.0f}s, се вчитуваат тежините ...", flush=True)

    ckpt = torch.load(os.path.join(RADFM_DIR, "pytorch_model.bin"), map_location="cpu")
    result = model.load_state_dict(ckpt, strict=False)
    del ckpt
    gc.collect()
    missing = [k for k in result.missing_keys if not k.startswith("embedding_layer.bert_model.")]
    if missing:
        raise RuntimeError(f"Во pytorch_model.bin недостасуваат {len(missing)} тежини, пр. {missing[:5]}")
    if result.unexpected_keys:
        print(f"Предупредување: {len(result.unexpected_keys)} непознати клучеви, "
              f"пр. {result.unexpected_keys[:5]}", flush=True)
    model.eval()
    print(f"Тежините вчитани ({time.time() - t0:.0f}s вкупно)", flush=True)
    return model, tokenizer, image_padding_tokens[0]


def load_volume(path):
    """NIfTI -> тензор [1, 1, 3, H, W, D] (B, S, C, H, W, D) за RadFM."""
    img = nib.as_closest_canonical(nib.load(path))       # секоја збирка во RAS (IXI е сагитален)
    vol = np.asarray(img.dataobj, dtype=np.float32)
    if vol.ndim == 4:
        vol = vol[..., 0]
    vol = np.nan_to_num(vol)
    # аксијални пресеци како во радиолошки приказ: редови напред->назад,
    # колони десно->лево од пациентот; длабочина долу->горе
    vol = vol[::-1, ::-1, :].transpose(1, 0, 2)
    lo, hi = np.percentile(vol, (0.5, 99.5))
    vol = np.clip((vol - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    t = torch.from_numpy(np.ascontiguousarray(vol))[None, None]           # 1, 1, H, W, D
    t = F.interpolate(t, size=(TARGET_H, TARGET_W, TARGET_D), mode="trilinear", align_corners=False)
    return t.repeat(1, 3, 1, 1, 1).unsqueeze(0)                            # 1, 1, 3, H, W, D


@torch.no_grad()
def ask(model, tokenizer, image_tokens, vision_x, question):
    """Исто како model.generate во RadFM, но со greedy, точен </s> и пократок одговор."""
    text = "<image>" + image_tokens + "</image>" + question
    lang_x = tokenizer(text, max_length=2048, truncation=True, return_tensors="pt")["input_ids"]
    model.embedding_layer.flag = "Text"
    embeds, _ = model.embedding_layer(lang_x.to(vision_x.device), vision_x)
    out = model.lang_model.generate(inputs_embeds=embeds, max_new_tokens=MAX_NEW_TOKENS,
                                    do_sample=False, num_beams=1,
                                    eos_token_id=EOS_ID, pad_token_id=PAD_ID)
    return tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()


def interrogate(dataset, asker):
    a = {}
    for field in ("modality", "specialized_sequence", "plane", "diagnosis_name",
                  "diagnosis_detailed", "severity_score"):
        a[field] = asker(Q[field])
    diagnosis = parse_diagnosis(a["diagnosis_name"])
    a["diagnosis_confidence"] = asker(Q["diagnosis_confidence"].format(
        label=LABEL_PHRASE.get(diagnosis, "an abnormality")))
    a["severity_confidence"] = asker(Q["severity_confidence"])

    modality = parse_modality(a["modality"])
    p = {
        "modality": modality,
        "specialized_sequence": parse_sequence(a["specialized_sequence"]) if modality != "CT" else None,
        "plane": parse_plane(a["plane"]),
        "diagnosis_name": diagnosis,
        "diagnosis_detailed": parse_detailed(a["diagnosis_detailed"], diagnosis),
        "severity_score": parse_score(a["severity_score"]),
        "diagnosis_confidence": parse_score(a["diagnosis_confidence"]),
        "severity_confidence": parse_score(a["severity_confidence"]),
    }

    if dataset == "molab":
        a["primary_origin"] = asker(Q_MOLAB["primary_origin"])
        origin = parse_origin(a["primary_origin"])
        if origin == "lung":
            a["lung_type"] = asker(Q_MOLAB["lung_type"])
            origin = parse_lung_type(a["lung_type"])
        if origin in ORGAN_PHRASE:
            a["origin_confidence"] = asker(Q_MOLAB["origin_confidence"].format(organ=ORGAN_PHRASE[origin]))
        p.update({
            "primary_origin": origin,
            "primary_origin_subtype": None,
            "origin_confidence": parse_score(a.get("origin_confidence")),
            "subtype_confidence": None,
        })

    if dataset == "ms":
        for field in Q_MS:
            a[field] = asker(Q_MS[field])
        p.update({
            "ms_type": parse_ms_type(a["ms_type"]),
            "lesion_location": a["lesion_location"] or None,
            "lesion_burden": parse_burden(a["lesion_burden"]),
            "active_lesions": parse_yes_no(a["active_lesions"]),
            "ms_confidence": parse_score(a["ms_confidence"]),
        })
    return a, p


def scan_names(paths):
    """Име на датотеката; ако се повторува (open_ms_data: T1W.nii.gz), пред него оди папката."""
    base = [os.path.basename(p) for p in paths]
    if len(set(base)) == len(base):
        return base
    return [os.path.basename(os.path.dirname(p)) + "_" + b for p, b in zip(paths, base)]


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in DATASETS:
        sys.exit("Употреба: python3 run_radfm.py " + "|".join(DATASETS))
    dataset = sys.argv[1]

    paths = sorted(glob.glob(os.path.join(DATA_DIR, DATASETS[dataset])))
    names = scan_names(paths)
    out_path = os.path.join(OUT_DIR, f"radfm_{dataset}.jsonl")

    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["scan_name"])
                except (ValueError, KeyError):
                    pass
    todo = [(p, n) for p, n in zip(paths, names) if n not in done]
    print(f"{dataset}: најдени {len(paths)} скенови, веќе обработени {len(done)}, "
          f"остануваат {len(todo)}", flush=True)
    if not todo:
        return

    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    model, tokenizer, image_tokens = load_model()

    with open(out_path, "a", encoding="utf-8") as f:
        for i, (path, scan) in enumerate(todo, 1):
            t0 = time.time()
            try:
                vision_x = load_volume(path).to("cuda")
                answers, parsed = interrogate(
                    dataset, lambda q: ask(model, tokenizer, image_tokens, vision_x, q))
            except Exception as e:  # еден лош скен не го прекинува целиот job
                print(f"[{i}/{len(todo)}] {scan}  ГРЕШКА: {e!r}", flush=True)
                torch.cuda.empty_cache()
                continue
            record = {"scan_name": scan, "dataset": dataset, "answers": answers,
                      "parsed_response": parsed, "seconds": round(time.time() - t0, 1)}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i}/{len(todo)}] {scan}  {record['seconds']}s  "
                  f"{parsed['diagnosis_name']}", flush=True)
    print("Готово", flush=True)


if __name__ == "__main__":
    main()
