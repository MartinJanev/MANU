"""Shared adapter of the two M3D-LaMed backbones (Phi-3-4B, Llama-2-7B): one four-option question per schema field.

The calls are the Quick start of github.com/BAAI-DCAI/M3D: a (1, 32, 256, 256) volume in [0, 1], 256 <im_patch>
tokens directly before the question, model.generate(images, input_ids). The empty border is cropped before the
resize, as the authors' Data/process scripts do. Deviations: greedy decoding (the Quick start samples), 16 new
tokens, and no confidences (the March runs filled in a constant 0.6).
"""
import numpy as np

from common.qna import interrogate
from common.volume import resize, to01

INPUT_SHAPE = (32, 256, 256)
IMAGE_TOKENS = "<im_patch>" * 256  # proj_out_num of the Quick start
MAX_NEW_TOKENS = 16


def model_volume(vol01):
    """(D, H, W) in [0, 1] -> (1, 32, 256, 256), cropped to the voxels above 0."""
    filled = vol01 > 0
    if filled.any():
        box = []
        for axis in range(3):
            hits = np.flatnonzero(filled.any(axis=tuple(a for a in range(3) if a != axis)))
            box.append(slice(hits[0], hits[-1] + 1))
        vol01 = vol01[tuple(box)]
    return np.ascontiguousarray(resize(vol01, INPUT_SHAPE)[None], dtype=np.float32)


class M3DLaMedAdapter:
    mode = "qna_per_field"

    def load(self):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoTokenizer

        path = snapshot_download(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True).eval()
        self.model = self.model.to("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(
            path, model_max_length=512, padding_side="right", use_fast=False, trust_remote_code=True)

    def _generate(self, volume, prompt, max_new_tokens):
        import torch

        ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to("cuda")
        image = torch.from_numpy(volume)[None].to(dtype=torch.bfloat16, device="cuda")
        with torch.inference_mode():
            out = self.model.generate(image, ids, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1)
        return self.tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()  # only the new tokens

    def predict(self, scan, volumes):
        volume = model_volume(to01(volumes[0], scan.dataset))

        def ask_fn(question):
            return self._generate(volume, IMAGE_TOKENS + question, MAX_NEW_TOKENS), None

        values, log = interrogate(scan.dataset, ask_fn, "A")
        return values, dict(log, input_shape=list(volume.shape))
