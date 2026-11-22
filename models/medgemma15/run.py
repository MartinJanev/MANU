"""MedGemma 1.5 4B-it: the official CT/MRI volume recipe, the colleagues' full-JSON prompt, JSON answer.

Volume -> slices as in notebooks/high_dimensional_ct_hugging_face.ipynb of github.com/Google-Health/medgemma and
section "CT and MRI" of the MedGemma 1.5 technical report (arXiv 2604.05081v2):
- at most 85 axial slices, sampled with the notebook's formula when the volume has more (all slices otherwise);
- MRI: one min-max over the whole volume to 0-255, the grey value in R, G and B, no windowing;
- CT (the CT-RATE control): three windows in R, G, B, (-1024, 1024), (-135, 215), (0, 80) HU (the notebook's code);
- each slice JPEG-encoded as a data URI (the notebook's _encode); the processor resizes it to 896x896.
One user message (no system role): instruction, then (image, "SLICE n") per slice, then the JSON schema.
Greedy decoding, max_new_tokens 1024. On real scans the model often writes a FINDINGS/IMPRESSION report instead of
the JSON (dev run, job 2759): such an answer is read with the shared report extractor (details.parsed_from "report").
An answer with no usable value either way raises PredictionError: an error record, raw text in details.

Usage: sbatch -J medgemma15 jobs/run_model.sh medgemma15 --split dev|eval|full [--controls] [--limit N]
       (jobs/run_model.sh runs `python3 /manu/models/medgemma15/run.py --split ...` inside the container)
"""
import base64
import io

import numpy as np
from PIL import Image

from common import runner
from common.parse import extract_from_report, parse_json_answer
from common.schema import full_prompt, normalize_values, slice_preface
from evaluation.evaluate import norm_diag

MAX_SLICE = 85
CT_WINDOWS = [(-1024, 1024), (-135, 215), (0, 80)]  # the notebook's window_clips (R, G, B)
MAX_NEW_TOKENS = 1024
QNA_MAX_NEW_TOKENS = 128  # one question
SCHEMA_START = "Output ONLY this JSON schema below"  # first line of the prompt that goes after the slices


def sample_slices(n_slices):
    """Slice indices: the notebook's uniform sampling when there are more than MAX_SLICE, else every slice."""
    indices = list(range(n_slices))
    if n_slices > MAX_SLICE:
        indices = [indices[int(round(i / (MAX_SLICE) * (n_slices - 1)))] for i in range(1, MAX_SLICE + 1)]
    return indices


def _norm(vol, lo, hi):
    """The notebook's norm(): clip to [lo, hi] and scale to 0-255 (float)."""
    vol = np.clip(vol, lo, hi).astype(np.float32)
    vol -= lo
    vol /= (hi - lo)
    vol *= 255.0
    return vol


def to_rgb_slices(vol, dataset, indices):
    """uint8 (H, W, 3) slices. CT: the notebook's three windows; MRI: min-max over the whole volume, R = G = B.

    A constant volume (the blank control) gives zeros for MRI, as there is no range to scale."""
    vol = np.asarray(vol, dtype=np.float32)
    if dataset == "ct_rate":
        rgb = np.stack([_norm(vol[indices], lo, hi) for lo, hi in CT_WINDOWS], axis=-1)
    else:
        lo, hi = float(vol.min()), float(vol.max())
        grey = _norm(vol, lo, hi)[indices] if hi > lo else np.zeros_like(vol[indices])
        rgb = np.repeat(grey[..., None], 3, axis=-1)
    return np.round(rgb, 0).astype(np.uint8)


def encode(data):
    """The notebook's _encode(): one slice as a JPEG data URI."""
    fmt = "jpeg"
    with io.BytesIO() as img_bytes:
        with Image.fromarray(data) as img:
            img.save(img_bytes, format=fmt)
        img_bytes.seek(0)
        encoded_string = base64.b64encode(img_bytes.getbuffer()).decode("utf-8")
    return f"data:image/{fmt};base64,{encoded_string}"


def split_prompt(dataset, prior=False):
    """(instruction, schema text): the prompt's role and task lines plus the slice preface go before the slices,
    the schema (from "Output ONLY this JSON schema below") after them, as the notebook's instruction / query_text."""
    prompt = full_prompt(dataset, prior)
    cut = prompt.index(SCHEMA_START)
    return prompt[:cut] + slice_preface().rstrip("\n"), "\n\n" + prompt[cut:]


def build_messages(slice_uris, instruction, query_text):
    """The notebook's content list: instruction, (image, "SLICE n") for every slice, query; one user turn."""
    content = [{"type": "text", "text": instruction}]
    for slice_number, uri in enumerate(slice_uris, 1):
        content.append({"type": "image", "image": uri})
        content.append({"type": "text", "text": f"SLICE {slice_number}"})
    content.append({"type": "text", "text": query_text})
    return [{"role": "user", "content": content}]


def answer_values(raw, dataset, details):
    """Raw schema values of one answer: its JSON, else the answer read as a report. Fills details (problems,
    parsed_from, diagnosis_name_raw); raises PredictionError when nothing usable is left. Also used by
    tools/reparse.py."""
    obj, problems = parse_json_answer(raw)
    details["problems"] = problems
    details["parsed_from"] = "json" if obj is not None else "report"
    if obj is None:
        obj = extract_from_report(raw, dataset)
    if "diagnosis_name" in obj:  # "Normal" -> normal, "Brain tumor" -> tumor (evaluation's normaliser)
        details["diagnosis_name_raw"] = obj["diagnosis_name"]
        obj["diagnosis_name"] = norm_diag(obj["diagnosis_name"])
    if not any(v is not None for v in normalize_values(dataset, obj).values()):
        if details["parsed_from"] == "report":
            raise runner.PredictionError(
                f"no usable JSON in the answer ({', '.join(problems)}) and no finding in it as a report", details)
        raise runner.PredictionError("JSON without any valid schema value", details)
    return obj


class MedGemma15Adapter:
    name = "medgemma-1.5-4b"
    model_id = "google/medgemma-1.5-4b-it"
    mode = "full_json"

    def load(self):
        import torch
        import transformers
        from huggingface_hub import snapshot_download

        # The local snapshot, not the repo id: given a repo id, transformers 4.57.3 asks the Hub whether the
        # tokenizer is a Mistral one (tokenization_utils_base._patch_mistral_regex -> model_info), which raises
        # OfflineModeIsEnabled in a job (HF_HUB_OFFLINE=1). 4.57.6 skips that call offline; the pin stays 4.57.3.
        local_dir = snapshot_download(self.model_id)
        # The notebook's loading code (processor without the model kwargs it also passes there)
        self.processor = transformers.AutoProcessor.from_pretrained(local_dir, use_fast=True)
        self.model = transformers.AutoModelForImageTextToText.from_pretrained(
            local_dir, dtype=torch.bfloat16, device_map="auto", offload_buffers=True)
        self.model.eval()

    def _generate(self, messages, max_new_tokens=MAX_NEW_TOKENS):
        import torch

        # The notebook's tokenization and greedy generation; decoding of the new tokens as on the model card
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, continue_final_message=False,
            return_tensors="pt", tokenize=True, return_dict=True)
        with torch.inference_mode():
            inputs = inputs.to(self.model.device, dtype=torch.bfloat16)
            generated = self.model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
        input_len = inputs["input_ids"].shape[-1]
        return self.processor.decode(generated[0][input_len:], skip_special_tokens=True)

    def predict(self, scan, volumes):
        vol = volumes[0]
        indices = sample_slices(len(vol))
        slices = to_rgb_slices(vol, scan.dataset, indices)
        instruction, query_text = split_prompt(scan.dataset, scan.prior)
        messages = build_messages([encode(s) for s in slices], instruction, query_text)
        raw = self._generate(messages)
        details = {"prompt": {"instruction": instruction, "query": query_text}, "raw": raw, "slices": indices}
        return answer_values(raw, scan.dataset, details), details

    def asker(self, scan, volumes):
        """One question per call, over the same slices (runner --task qna): the question where the schema was."""
        vol = volumes[0]
        uris = [encode(s) for s in to_rgb_slices(vol, scan.dataset, sample_slices(len(vol)))]
        instruction = slice_preface().rstrip("\n")

        def ask(question):
            messages = build_messages(uris, instruction, "\n\n" + question)
            return self._generate(messages, QNA_MAX_NEW_TOKENS), None

        return ask


if __name__ == "__main__":
    runner.main(MedGemma15Adapter())
