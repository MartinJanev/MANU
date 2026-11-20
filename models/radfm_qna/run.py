"""RadFM (MedLLaMA-13B + 3D ViT perceiver, trained on RadMD): the shared per-field questions, like the other models.

The older models/radfm/run.py ran RadFM outside the shared pipeline (its own questions, likelihood choice
calibrated on a black volume, its own run loop). This adapter puts RadFM through common.runner instead: the same
splits, canonical volume, question bank (common.qna, the M3D "Choices: A. ... D. ..." format med-2e3 uses), output
records, resume and blank controls as every other model, so its row in the dev/eval report is comparable.

Per scan
  1. The canonical axial stack (common.volume.load_canonical) -> to01 -> 64 x 256 x 256 -> RadFM's layout
     [1 batch, 1 image, 3 channels, H 256, W 256, D 64] (grey repeated to 3 channels), as load_volume in
     models/radfm/run.py builds it.
  2. Each question of common.qna.interrogate is asked as "<image>...</image>" + question (the image first, as in
     RadFM/Quick_demo/test.py), greedy, at most 16 new tokens. RadFM was trained to answer VQA in free text, not
     with letters, so the four option probabilities are the likelihoods of the four displayed option texts after
     the question (mean log-probability per token, softmax over the four). They only give the confidence fields;
     the value is what common.parse.parse_choice reads in the generated answer, as for every model.
  3. One free-text report (RadFM's caption template) per scan goes to the details file, for reading only.

Model code, tokenizer files and pytorch_model.bin come from the RadFM clone already on the cluster
(third_party/RadFM/Quick_demo, the folder jobs/job_radfm.sh mounts as /radfm); Med-KEBERT, which RadFM's
embedding layer loads at construction, from hf_cache. Image: containers/radfm.sif (transformers 4.28.1).

Usage: sbatch -J radfm_qna jobs/run_model.sh radfm_qna --split dev|eval|full [--controls] [--limit N]
       (jobs/run_model.sh runs `python3 /manu/models/radfm_qna/run.py --split ...` inside the container)
"""
import re

import numpy as np

from common import runner
from common.qna import interrogate
from common.volume import resize, to01

RADFM_DIR = "/third_party/RadFM/Quick_demo"  # Model/, Language_files/, pytorch_model.bin
INPUT_SHAPE = (64, 256, 256)  # (D, H, W): the paper trains 3D at most 64 slices deep; 256 in-plane (RADFM_HW default)
IMAGE_TOKENS_PER_IMAGE = 32  # <image0>..<image31> stand for the 32 perceiver outputs of image 0
PAD_ID, BOS_ID, EOS_ID = 0, 1, 2  # as set in Quick_demo/test.py (Language_files/config.json says eos 1)
MAX_NEW_TOKENS = 16  # MCQ answers (global constraint)
REPORT_MAX_NEW_TOKENS = 512  # reports (global constraint)
# RadFM's caption training template, the same report question models/radfm/run.py asks
REPORT_QUESTION = "Can you provide a caption consists of finding and impression for this medical image?"

_CHOICES = re.compile(r" Choices: A\. (.*?) B\. (.*?) C\. (.*?) D\. (.*)$", re.S)


def model_volume(volume, dataset):
    """Canonical (D, H, W) array -> (H, W, D) float32 in [0, 1], RadFM's axis order (depth last).

    load_canonical's H (front->back) and W (patient right->left) are the rows and columns load_volume in
    models/radfm/run.py gives RadFM, so only depth moves to the end.
    """
    vol = resize(to01(volume, dataset), INPUT_SHAPE)
    return np.ascontiguousarray(vol.transpose(1, 2, 0), dtype=np.float32)


def image_prefix(n_images=1):
    """'<image><image0>...<image31></image>' once per image, as Quick_demo/test.py writes it."""
    parts = []
    for i in range(n_images):
        tokens = "".join(f"<image{i * IMAGE_TOKENS_PER_IMAGE + j}>" for j in range(IMAGE_TOKENS_PER_IMAGE))
        parts.append("<image>" + tokens + "</image>")
    return "".join(parts)


def shown_options(question):
    """The four option texts of a common.qna question in style "A", in displayed order (inverse of format_mcq)."""
    m = _CHOICES.search(question)
    if m is None:
        raise ValueError(f"not a four-option 'Choices: A. ...' question: {question!r}")
    return list(m.groups())


def option_probabilities(logps):
    """Softmax over the options' mean log-probabilities; None if none is finite."""
    x = np.asarray(logps, dtype=np.float64)
    if not np.isfinite(x).any():
        return None
    e = np.exp(x - x[np.isfinite(x)].max())
    e[~np.isfinite(x)] = 0.0
    return [float(p) for p in e / e.sum()]


def add_image_tokens(tokenizer, n_images=100):
    """The special tokens of Quick_demo/test.py, kept verbatim (incl. adding them inside the loop): the token ids
    must match the checkpoint's embedding rows."""
    special = {"additional_special_tokens": ["<image>", "</image>"]}
    for i in range(n_images):
        special["additional_special_tokens"] += [f"<image{i * IMAGE_TOKENS_PER_IMAGE + j}>"
                                                 for j in range(IMAGE_TOKENS_PER_IMAGE)]
        tokenizer.add_special_tokens(special)
    tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id = PAD_ID, BOS_ID, EOS_ID
    return tokenizer


class RadFMQnAAdapter:
    name = "radfm-qna"  # not "radfm": results/radfm/full holds the older run, which evaluate.py still reports
    model_id = "chaoyi-wu/RadFM"
    mode = "qna_per_field"

    def load(self):
        import gc
        import os
        import sys
        import time

        import torch
        from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer

        torch.manual_seed(0)
        sys.path.insert(0, RADFM_DIR)
        import Model.RadFM.multimodality_model as mm

        # Language_files holds only config.json and the tokenizer (no LLaMA weights), so from_pretrained would
        # fail: build LLaMA from the config; every weight comes from pytorch_model.bin below.
        class LlamaFromConfig:
            @staticmethod
            def from_pretrained(path, **kwargs):
                return LlamaForCausalLM(LlamaConfig.from_pretrained(path))

        mm.LlamaForCausalLM = LlamaFromConfig
        lang_path = os.path.join(RADFM_DIR, "Language_files")
        self.tokenizer = add_image_tokens(LlamaTokenizer.from_pretrained(lang_path))

        t0 = time.time()
        model = mm.MultiLLaMAForCausalLM(lang_model_path=lang_path).to("cuda")  # float32, as in the full run
        ckpt = torch.load(os.path.join(RADFM_DIR, "pytorch_model.bin"), map_location="cpu")
        result = model.load_state_dict(ckpt, strict=False)
        del ckpt
        gc.collect()
        # The text encoder (Med-KEBERT) is loaded separately from hf_cache; anything else missing is fatal.
        missing = [k for k in result.missing_keys if not k.startswith("embedding_layer.bert_model.")]
        if missing:
            raise RuntimeError(f"pytorch_model.bin is missing {len(missing)} weights, e.g. {missing[:5]}")
        model.eval()
        self.model = model
        print(f"{self.name}: RadFM loaded in {time.time() - t0:.0f}s", flush=True)

    def _generate(self, volume, question, options, max_new_tokens):
        """(greedy answer, probabilities of `options` or None). The only GPU code; tests replace it."""
        import torch

        tok, model = self.tokenizer, self.model
        vision_x = torch.from_numpy(volume)[None].repeat(3, 1, 1, 1)[None, None].to("cuda")  # [1, 1, 3, H, W, D]
        q = tok(image_prefix(1) + question, return_tensors="pt")["input_ids"]
        with torch.no_grad():
            model.embedding_layer.flag = "Text"
            embeds, _ = model.embedding_layer(q.to("cuda"), vision_x)
            out = model.lang_model.generate(inputs_embeds=embeds, max_new_tokens=max_new_tokens, do_sample=False,
                                            num_beams=1, eos_token_id=EOS_ID, pad_token_id=PAD_ID)
            answer = tok.batch_decode(out, skip_special_tokens=True)[0].strip()
            if options is None:
                return answer, None
            logps = []
            for option in options:  # mean log-probability per token of the option text right after the question
                a = tok(option, add_special_tokens=False, return_tensors="pt")["input_ids"]
                embeds, _ = model.embedding_layer(torch.cat([q, a], dim=1).to("cuda"), vision_x)
                logits = model.lang_model(inputs_embeds=embeds).logits[0, q.shape[1] - 1:-1].float()
                logps.append(logits.log_softmax(-1).gather(1, a[0].to(logits.device)[:, None]).mean().item())
        return answer, option_probabilities(logps)

    def predict(self, scan, volumes):
        volume = model_volume(volumes[0], scan.dataset)

        def ask_fn(question):
            return self._generate(volume, question, shown_options(question), MAX_NEW_TOKENS)

        values, log = interrogate(scan.dataset, ask_fn, "A")
        report, _ = self._generate(volume, REPORT_QUESTION, None, REPORT_MAX_NEW_TOKENS)
        return values, dict(log, report=report, input_shape=list(volume.shape))


if __name__ == "__main__":
    runner.main(RadFMQnAAdapter())
