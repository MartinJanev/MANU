"""
RadFM inference: one structured diagnostic record per brain scan.

What happens for every scan
  1. The NIfTI volume is turned into the kind of tensor RadFM was trained on
     (axial slices, RADFM_HW x RADFM_HW x 64, default 256, values 0..1, 3 identical channels).
  2. RadFM is asked one short question per schema field. Instead of letting it
     write free text and guessing what it meant, we hand it the allowed answers
     and measure how likely it finds each one (`choose`). Each likelihood is
     divided by the one RadFM gives the same answer for an all-black volume, so an
     answer it would give without looking at the image does not win by default.
     The most likely answer becomes the field value; its probability becomes the confidence.
  3. The values are written in exactly the same JSONL format as the colleagues'
     Outputs/radfm_*_processed.jsonl, so every model is scored by the same
     pipeline. Everything behind each value (questions, all option
     probabilities, a free-text report) goes to a *_details.jsonl side file.

Container paths (mounted by job_radfm.sh)
  /data       raw NIfTI datasets
  /radfm      RadFM/Quick_demo: Model/, Language_files/, pytorch_model.bin
  /reference  colleague files radfm_*_processed.jsonl (ground-truth metadata)
  /results    output

Usage
  python3 run_radfm.py ixi | molab | ms    run one dataset (resumes if interrupted)
  python3 run_radfm.py ms 3                pilot: only the first 3 scans
  python3 run_radfm.py preview             PNGs of exactly what the model sees
"""
import datetime
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

# =============================================================================
# 1. Settings
# =============================================================================
DATA_DIR, RADFM_DIR, REFERENCE_DIR, OUT_DIR = "/data", "/radfm", "/reference", "/results"

DATASETS = {
    "ixi": {
        "scans": "ixi/T1/*.nii.gz",
        "output": "radfm_healthy_processed.jsonl",
        "experiment_id": "radfm_healthy",
    },
    "molab": {
        # All 373 figshare NIfTIs (75 patients). Sequence names vary per scanner
        # (AX_T1_FL3D, Ax_FSPGR_BRAVO, eTRA-3D-T1Gd, ...), so "*T1Gd*" only caught 26 of them.
        "scans": "molab/nifti/*_img.nii*",
        "output": "radfm_brain_metastases_processed.jsonl",
        "experiment_id": "radfm_brain_metastases",
    },
    "ms": {
        # One T1W.nii.gz per patient folder; the FLAIR.nii.gz next to it is fed as a second image.
        "scans": "open_ms_data/cross_sectional/raw/patient*/T1W.nii.gz",
        "output": "radfm_ms_processed.jsonl",
        "experiment_id": "radfm_multiple_sclerosis",
    },
}

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

# Input size. Depth: the paper trains 3D at most 64 slices deep, divisible by 4.
# In-plane: the paper says 256 x 256, the released data code resizes to 512 x 512;
# 256 is the default, RADFM_HW=512 is the second run to compare.
TARGET_H = TARGET_W = int(os.environ.get("RADFM_HW", 256))
TARGET_D = 64

# =============================================================================
# 2. Questions
# Each field: (question, {schema value: answer phrase RadFM is scored on}).
# Wording follows RadFM's training templates where one exists.
# =============================================================================
QUESTIONS = {
    "modality": ("What modality is used to take this image?",  # exact training template
                 {"MRI": "MRI", "CT": "CT"}),
    "specialized_sequence": ("Which MRI sequence is used in this image?",
                             {"T1": "T1-weighted", "T1C+": "T1-weighted with contrast",
                              "T2": "T2-weighted", "FLAIR": "FLAIR"}),
    "plane": ("In which plane is this image scanned?",
              {"axial": "Axial", "sagittal": "Sagittal", "coronal": "Coronal"}),
    "diagnosis_detailed": ("What type of lesion is seen in this image?",
                           {"glioma": "Glioma", "meningioma": "Meningioma",
                            "pituitary_tumor": "Pituitary tumor", "carcinoma": "Carcinoma",
                            "germinoma": "Germinoma", "granuloma": "Granuloma",
                            "medulloblastoma": "Medulloblastoma", "neurocytoma": "Neurocytoma",
                            "papilloma": "Papilloma", "schwannoma": "Schwannoma",
                            "tuberculoma": "Tuberculoma", "ischemic": "Ischemic stroke",
                            "hemorrhagic": "Hemorrhagic stroke"}),
    "severity_score": ("How severe is the abnormality in this image?",
                       {0.0: "None", 0.25: "Mild", 0.5: "Moderate", 0.75: "Severe", 1.0: "Very severe"}),
    "primary_origin": ("Where is the primary cancer of this brain metastasis located?",
                       {"lung-NSCLC": "Non-small cell lung cancer", "lung-SCLC": "Small cell lung cancer",
                        "breast": "Breast cancer", "melanoma": "Melanoma", "kidney": "Kidney cancer",
                        "colon": "Colon cancer", "bladder": "Bladder cancer", "ovary": "Ovarian cancer",
                        "uterus": "Uterine cancer", "pharynx": "Pharyngeal cancer",
                        "bone-sarcoma": "Bone sarcoma"}),
    "ms_type": ("Which type of multiple sclerosis is this?",
                {"SMRR": "Relapsing-remitting", "SMSP": "Secondary progressive",
                 "SMPP": "Primary progressive"}),
    "ms_lesion_location": ("Where are the lesions located?",
                           {"periventricular": "Periventricular", "juxtacortical": "Juxtacortical",
                            "infratentorial": "Infratentorial", "deep white matter": "Deep white matter"}),
    "ms_lesion_burden": ("How many lesions are visible?",
                         {"none": "None", "low": "Few", "moderate": "Several", "high": "Many"}),
    # ms_active_lesions is not asked: active lesions are seen only with gadolinium, and
    # open_ms_data has no contrast scans, so the value is always null (see interrogate).
}

# Diagnosis: RadFM learned diseases as yes/no questions, "Is the <disease> visible in the image?".
DISEASE_QUESTION = "Is the {} visible in the image?"
DISEASES = {"tumor": "brain tumor", "stroke": "stroke", "multiple sclerosis": "multiple sclerosis"}

# Free-text report, only for reading in the details file (RadFM caption training template).
REPORT_QUESTION = "Can you provide a caption consists of finding and impression for this medical image?"


# =============================================================================
# 3. Finding scans and their ground truth
# =============================================================================
def find_scans(dataset):
    """[(image_path as the colleagues spelled it, [volume files fed to RadFM]), ...]"""
    pattern = os.path.join(DATA_DIR, DATASETS[dataset]["scans"])
    scans = []
    for path in sorted(glob.glob(pattern, recursive=True)):
        if dataset == "ixi":
            scans.append((os.path.basename(path), [path]))
        elif dataset == "molab":
            scans.append((molab_name(os.path.basename(path)), [path]))
        else:  # ms: patient folder -> "patient01_T1.nii.gz", fed T1 + FLAIR
            folder = os.path.dirname(path)
            flair = [] if os.environ.get("RADFM_MS_T1_ONLY") else [os.path.join(folder, "FLAIR.nii.gz")]
            scans.append((os.path.basename(folder) + "_T1.nii.gz", [path] + flair))
    return scans


def molab_name(name):
    """figshare '10005_19010618_eTRA-3D-T1Gd_img.nii' -> '010005N__010005_19010618_eTRA-3D-T1Gd_img.nii'."""
    if re.match(r"\d+N__", name):  # already in the colleague spelling
        return name
    patient, rest = name.split("_", 1)
    patient = patient.zfill(6)
    return f"{patient}N__{patient}_{rest}"


def load_reference(dataset):
    """image_path -> metadata, from the colleague output file for this dataset."""
    with open(os.path.join(REFERENCE_DIR, DATASETS[dataset]["output"]), encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return {r["input"]["image_path"]: r["input"]["metadata"] for r in rows}


def metadata_for(image_path, reference):
    """Ground-truth metadata for one scan. MOLAB labels are per patient (first 6 digits),
    so a scan the colleagues did not have takes the metadata of the same patient."""
    if image_path in reference:
        return reference[image_path]
    for name, meta in reference.items():
        if re.match(r"\d{6}N__", image_path) and name[:6] == image_path[:6]:
            return meta
    print(f"WARNING: no reference metadata for {image_path}; labels set to null", flush=True)
    meta = dict(next(iter(reference.values())))  # constant fields (dataset, class, modality ...)
    for key in ("primary_origin", "ms_type"):
        if key in meta:
            meta[key] = None
    return meta


# =============================================================================
# 4. Reading a scan into the tensor RadFM expects
# =============================================================================
def load_volume(path):
    """One NIfTI file -> tensor [3, H, W, 64], laid out like RadFM's training images."""
    # 1. Read and rotate into a standard orientation (RAS), whatever the scanner stored.
    img = nib.as_closest_canonical(nib.load(path))
    vol = np.nan_to_num(np.asarray(img.dataobj, dtype=np.float32))
    if vol.ndim == 4:  # some files carry a 4th axis of length 1
        vol = vol[..., 0]
    # 2. Axial slices as a radiologist views them (training images were Radiopaedia
    #    screenshots): rows front->back, columns patient right->left, depth feet->head.
    vol = vol[::-1, ::-1, :].transpose(1, 0, 2)
    # 3. Brightness: clip the 0.5% darkest and brightest voxels, scale to 0..1
    #    (stands in for the display window of the JPEG training images).
    lo, hi = np.percentile(vol, (0.5, 99.5))
    vol = np.clip((vol - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    # 4. Resize to TARGET_H x TARGET_W in-plane and 64 slices.
    t = torch.from_numpy(np.ascontiguousarray(vol))[None, None]
    t = F.interpolate(t, size=(TARGET_H, TARGET_W, TARGET_D), mode="trilinear", align_corners=False)
    # 5. Grey -> 3 identical colour channels.
    return t[0].repeat(3, 1, 1, 1)


def load_scan(files):
    """All volumes of one scan -> vision_x [1 batch, N images, 3, H, W, D]."""
    return torch.stack([load_volume(f) for f in files]).unsqueeze(0)


# =============================================================================
# 5. Loading RadFM (set once, used by every question)
# =============================================================================
MODEL = TOKENIZER = IMAGE_TOKENS = None


def load_model():
    """Build RadFM, load its weights and the tokenizer with the image placeholder tokens."""
    global MODEL, TOKENIZER, IMAGE_TOKENS
    sys.path.insert(0, RADFM_DIR)
    from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer
    import Model.RadFM.multimodality_model as mm

    # Language_files holds only config.json (no weights), so from_pretrained would fail.
    # Build LLaMA from the config; all weights come from pytorch_model.bin below.
    class LlamaFromConfig:
        @staticmethod
        def from_pretrained(path, **kwargs):
            return LlamaForCausalLM(LlamaConfig.from_pretrained(path))

    mm.LlamaForCausalLM = LlamaFromConfig
    lang_path = os.path.join(RADFM_DIR, "Language_files")

    # Tokenizer exactly as in RadFM/Quick_demo/test.py (kept verbatim: the token ids must
    # match the checkpoint). Image i is written as 32 special tokens <image{32i}>..<image{32i+31}>.
    TOKENIZER = LlamaTokenizer.from_pretrained(lang_path)
    IMAGE_TOKENS = []
    special = {"additional_special_tokens": ["<image>", "</image>"]}
    for i in range(100):
        tokens = [f"<image{i * 32 + j}>" for j in range(32)]
        IMAGE_TOKENS.append("".join(tokens))
        special["additional_special_tokens"] += tokens
        TOKENIZER.add_special_tokens(special)
    TOKENIZER.pad_token_id, TOKENIZER.bos_token_id, TOKENIZER.eos_token_id = 0, 1, 2

    t0 = time.time()
    MODEL = mm.MultiLLaMAForCausalLM(lang_model_path=lang_path).to("cuda")
    ckpt = torch.load(os.path.join(RADFM_DIR, "pytorch_model.bin"), map_location="cpu")
    result = MODEL.load_state_dict(ckpt, strict=False)
    del ckpt
    gc.collect()
    # The text encoder (Med-KEBERT) loads separately from /hf_cache; anything else missing is fatal.
    missing = [k for k in result.missing_keys if not k.startswith("embedding_layer.bert_model.")]
    if missing:
        raise RuntimeError(f"pytorch_model.bin is missing {len(missing)} weights, e.g. {missing[:5]}")
    MODEL.eval()
    print(f"RadFM loaded in {time.time() - t0:.0f}s", flush=True)


# =============================================================================
# 6. Asking RadFM
# =============================================================================
def prompt_ids(vision_x, question):
    """Token ids of '<image>...</image>' (once per image) followed by the question."""
    images = "".join("<image>" + IMAGE_TOKENS[i] + "</image>" for i in range(vision_x.shape[1]))
    return TOKENIZER(images + question, return_tensors="pt")["input_ids"]


@torch.no_grad()
def answer_logprob(vision_x, question, answer):
    """Average log-probability per token that RadFM gives `answer` right after `question`."""
    q = prompt_ids(vision_x, question)
    a = TOKENIZER(answer, add_special_tokens=False, return_tensors="pt")["input_ids"]
    MODEL.embedding_layer.flag = "Text"
    embeds, _ = MODEL.embedding_layer(torch.cat([q, a], dim=1).to(vision_x.device), vision_x)
    logits = MODEL.lang_model(inputs_embeds=embeds).logits[0, q.shape[1] - 1:-1].float()
    return logits.log_softmax(-1).gather(1, a[0].to(logits.device)[:, None]).mean().item()


# (question, number of images) -> option log-probabilities for an all-black volume.
# The probe run showed RadFM answers "multiple sclerosis" for black and noise images as
# well as for healthy IXI scans: much of each answer is a prior that ignores the image.
# ponytail: one blank prior per question (contextual calibration); per-dataset priors if
# the blank volume turns out not to be representative.
BLANK = {}


def choose(vision_x, question, options):
    """options = {schema value: answer phrase} -> (calibrated, raw), each {schema value: probability}.
    raw = softmax of the log-probabilities; calibrated = the same after subtracting the
    log-probabilities RadFM gives each answer for a black volume of the same shape."""
    def logps(x):
        return torch.tensor([answer_logprob(x, question, phrase) for phrase in options.values()])

    key = (question, vision_x.shape[1])
    if key not in BLANK:
        BLANK[key] = logps(torch.zeros_like(vision_x))
    real = logps(vision_x)
    return dict(zip(options, (real - BLANK[key]).softmax(0).tolist())), dict(zip(options, real.softmax(0).tolist()))


@torch.no_grad()
def write_report(vision_x):
    """Free-text findings/impression (greedy decoding); saved for reading, not scored."""
    MODEL.embedding_layer.flag = "Text"
    embeds, _ = MODEL.embedding_layer(prompt_ids(vision_x, REPORT_QUESTION).to(vision_x.device), vision_x)
    out = MODEL.lang_model.generate(inputs_embeds=embeds, max_new_tokens=128, do_sample=False,
                                    num_beams=1, eos_token_id=2, pad_token_id=0)
    return TOKENIZER.batch_decode(out, skip_special_tokens=True)[0].strip()


def interrogate(dataset, vision_x):
    """Ask every field. Returns (field values, {question: {"calibrated": probs, "raw": probs}})."""
    asked = {}

    def ask(question, options):  # calibrated probabilities; both versions go to the details file
        calibrated, raw = choose(vision_x, question, options)
        asked[question] = {"calibrated": calibrated, "raw": raw}
        return calibrated

    def pick(field):  # most likely option and its probability
        probs = ask(*QUESTIONS[field])
        best = max(probs, key=probs.get)
        return best, round(probs[best], 4)

    def p_yes(question):
        return ask(question, {"yes": "yes", "no": "no"})["yes"]

    v = {"modality": pick("modality")[0],
         "specialized_sequence": pick("specialized_sequence")[0],
         "plane": pick("plane")[0]}

    # Diagnosis: the disease RadFM is most sure is visible, if it says "yes" (> 0.5);
    # otherwise normal, with confidence = how sure it is that even that disease is absent.
    yes = {label: p_yes(DISEASE_QUESTION.format(name)) for label, name in DISEASES.items()}
    best = max(yes, key=yes.get)
    v["diagnosis_name"], conf = (best, yes[best]) if yes[best] > 0.5 else ("normal", 1 - yes[best])
    v["diagnosis_confidence"] = round(conf, 4)
    v["diagnosis_detailed"] = pick("diagnosis_detailed")[0] if v["diagnosis_name"] in ("tumor", "stroke") else None
    v["severity_score"], v["severity_confidence"] = pick("severity_score")

    # Dataset-specific fields are asked for every scan (not only when the diagnosis matches),
    # so each record carries the model's answer.
    if dataset == "molab":
        v["primary_origin"], v["origin_confidence"] = pick("primary_origin")
    if dataset == "ms":
        for field in ("ms_type", "ms_lesion_location", "ms_lesion_burden"):
            v[field] = pick(field)[0]
        v["ms_active_lesions"] = None  # no contrast in open_ms_data (see QUESTIONS)
        v["ms_confidence"] = round(yes["multiple sclerosis"], 4)
    return v, asked


# =============================================================================
# 7. Output record, in the colleagues' format
# =============================================================================
def build_record(dataset, row_index, image_path, metadata, values, error=None):
    values = {**values, "scan_name": image_path}  # only MOLAB records keep scan_name
    return {
        "timestamp": datetime.datetime.now().isoformat(),
        "experiment_id": DATASETS[dataset]["experiment_id"],
        "api_response": {"id": "", "model": "radfm", "prompt_tokens": 0,
                         "completion_tokens": 0, "total_tokens": 0},
        "input": {"image_path": image_path, "model_requested": "radfm",
                  "system_prompt_type": "qna_per_field_v1", "metadata": metadata},
        "output": {"error": error,
                   "parsed_response": None if error else {k: values.get(k) for k in PARSED_KEYS[dataset]},
                   "status": "error" if error else "success"},
        "processing": {"parameters": {"seed": 0, "temperature": 0}, "row_index": row_index},
    }


# =============================================================================
# 8. Running a dataset
# =============================================================================
def run(dataset, limit=None):
    out_path = os.path.join(OUT_DIR, DATASETS[dataset]["output"])
    details_path = out_path.replace("_processed.jsonl", "_details.jsonl")
    scans = find_scans(dataset)
    if not scans:
        sys.exit(f"No files match {DATASETS[dataset]['scans']} under {DATA_DIR}")
    reference = load_reference(dataset)

    done = set()  # resume: skip scans already in the output file
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            done = {json.loads(line)["input"]["image_path"] for line in f if line.strip()}
    todo = [(name, files) for name, files in scans if name not in done][:limit]
    print(f"{dataset}: {len(scans)} scans, {len(done)} already done, {len(todo)} to go", flush=True)
    if not todo:
        return

    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    load_model()
    with open(out_path, "a", encoding="utf-8") as out, open(details_path, "a", encoding="utf-8") as details:
        for row_index, (image_path, files) in enumerate(todo, len(done)):  # 0..N-1, as in the colleague files
            t0 = time.time()
            try:
                vision_x = load_scan(files).to("cuda")
                values, asked = interrogate(dataset, vision_x)
                report, error = write_report(vision_x), None
            except Exception as e:  # one bad scan is recorded as an error, the job goes on
                values, asked, report, error = {}, {}, None, repr(e)
                torch.cuda.empty_cache()
            seconds = round(time.time() - t0, 1)
            record = build_record(dataset, row_index, image_path, metadata_for(image_path, reference),
                                  values, error)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            details.write(json.dumps({"image_path": image_path, "files": files, "seconds": seconds,
                                      "answers": asked, "report": report, "error": error},
                                     ensure_ascii=False) + "\n")
            out.flush()
            details.flush()
            print(f"[{row_index + 1}/{len(scans)}] {image_path} {seconds}s "
                  f"{error or values['diagnosis_name']}", flush=True)
    print("Done", flush=True)


def preview(per_dataset=3):
    """results/previews/<scan>.png: 16 of the 64 slices RadFM receives, one 4x4 grid per image."""
    from PIL import Image
    folder = os.path.join(OUT_DIR, "previews")
    os.makedirs(folder, exist_ok=True)
    for dataset in DATASETS:
        for image_path, files in find_scans(dataset)[:per_dataset]:
            grids = []
            for f in files:
                vol = load_volume(f)[0].numpy()[::2, ::2]  # channel 0, half resolution
                slices = [vol[:, :, d] for d in np.linspace(0, TARGET_D - 1, 16).round().astype(int)]
                grids.append(np.vstack([np.hstack(slices[r * 4:r * 4 + 4]) for r in range(4)]))
            png = os.path.join(folder, image_path.split(".nii")[0] + ".png")
            Image.fromarray((np.hstack(grids) * 255).astype(np.uint8)).save(png)
            print(png, flush=True)


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) in (2, 3) else None
    if arg == "preview":
        preview()
    elif arg in DATASETS:
        run(arg, int(sys.argv[2]) if len(sys.argv) == 3 else None)
    else:
        sys.exit("Usage: python3 run_radfm.py " + "|".join(DATASETS) + "|preview [limit]")
