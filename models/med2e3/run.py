"""Med-2E3 (M3D-CLIP 3D encoder + SigLIP slice encoder + Phi-3-mini, trained on M3D-Data): one four-option
question per schema field through the shared question bank.

The volume goes in as the (1, 32, 256, 256) array in [0, 1] that M3D's .npy files hold; Med-2E3's own eval
transform (src/data/preprocess/vision3d_preprocess.py) then crops the foreground and resizes it to 32x256x256
again. Each question is "<image3d>\\n" + M3D's close-ended format "{Q} Choices: A. ... D. ...", in the "phi"
conversation template; decoding is greedy (16 new tokens), and the letter probabilities come from the scores of
the first generated token. Model code comes from the Med-2E3 git clone (third_party/Med-2E3), weights from the
Hugging Face snapshot.

Usage: sbatch -J med2e3 jobs/run_model.sh med2e3 --split dev|eval|full [--controls] [--limit N]
       (jobs/run_model.sh runs `python3 /manu/models/med2e3/run.py --split ...` inside the container)
"""
import importlib.util
import json
import os
import sys

import numpy as np

from common import runner
from common.qna import interrogate
from common.volume import resize, to01

MED2E3_REPO = "/third_party/Med-2E3"  # its modules import each other as `src.*`, so the repo root goes on sys.path
INPUT_SHAPE = (32, 256, 256)  # M3D's npys_256 arrays are (1, 32, 256, 256) in [0, 1]
IMAGE3D_TOKEN = "<image3d>"  # DEFAULT_IMAGE3D_TOKEN of src/utils/constants.py
IMAGE3D_PREFIX = IMAGE3D_TOKEN + "\n"  # as in the README's fine-tuning and evaluation examples
CONV_VERSION = "phi"  # --conv_version of scripts/eval/*.sh
LLM_DTYPE = "bfloat16"  # --llm_dtype of scripts/eval/*.sh
QUESTION_TOKENIZER = "GoodBaiBai88/M3D-CLIP"  # tokenizer of the TG-IS question text (MultiModalDataset)
MAX_NEW_TOKENS = 16
LETTERS = "ABCD"


def letter_token_ids(tokenizer):
    """Token ids of the answer letters A-D as the first token generated after "ASSISTANT:".

    The phi template trains answers as "ASSISTANT: C. ..." and the generation prompt ends with "ASSISTANT:", so the
    first answer token is the letter with its leading space (SentencePiece "▁C" for Phi-3's tokenizer).
    """
    prefix = tokenizer("ASSISTANT:", add_special_tokens=False).input_ids
    ids = []
    for letter in LETTERS:
        full = tokenizer("ASSISTANT: " + letter, add_special_tokens=False).input_ids
        if full[:len(prefix)] != prefix or len(full) != len(prefix) + 1:
            raise ValueError(f"answer letter {letter} is not one token after 'ASSISTANT:': {full}")
        ids.append(full[-1])
    return ids


def letter_probabilities(scores, ids):
    """Softmax of the first generated position's scores, at the four letter ids, renormalised over the four.

    None if no letter has a finite score.
    """
    letters = np.asarray(scores, dtype=np.float64)[list(ids)]
    if not np.isfinite(letters.max()):
        return None
    e = np.exp(letters - letters.max())
    return [float(p) for p in e / e.sum()]


def load_text_mlp(connector_state, text_mlp):
    """Load the TG-IS question projection from the vision3d connector's saved state dict, if it is there.

    The repo's Connector.load_model restores only the `_connector.*` weights, although training saves every
    connector parameter, so SpatialPoolingScore15Connector.text_mlp keeps its random initialisation. Keys are
    selected the way the repo's get_w selects `_connector` (prefix up to "text_mlp." dropped) and loaded strictly.
    Returns "checkpoint", or "random_seed0" if the state dict has no text_mlp keys (the init after seed 0 stays).
    """
    keyword = "text_mlp."
    weights = {k.split(keyword, 1)[1]: v for k, v in connector_state.items() if keyword in k}
    if not weights:
        return "random_seed0"
    text_mlp.load_state_dict(weights, strict=True)
    return "checkpoint"


def model_volume(volume, vision3d_preprocess):
    """The (1, 32, 256, 256) array through Med-2E3's eval transform (CropForeground, then Resize to 32x256x256).

    A volume with no voxel > 0 (the runner's blank control) has no foreground to crop to, and CropForeground would
    return an empty array that Resize rejects; it is already 32x256x256, so it goes in unchanged.
    """
    if not np.any(volume > 0):
        return volume
    return vision3d_preprocess(volume)


def preload_formatters(repo):
    """Import Med-2E3's src/data/template/formatter.py under its package name, with hashable formatter classes.

    The conversation templates (src/data/template/*.py) use EmptyFormatter/StringFormatter instances as dataclass
    field defaults, which Python >= 3.11 rejects as mutable ("use default_factory"); the repo targets Python 3.10,
    the image has 3.12. Registered before the template package is imported, and with the identity hash, the module
    gives the templates their 3.10 behaviour (one shared default instance per template class).
    """
    name = "src.data.template.formatter"
    spec = importlib.util.spec_from_file_location(name, os.path.join(repo, "src", "data", "template", "formatter.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    for cls in (module.EmptyFormatter, module.StringFormatter):
        cls.__hash__ = object.__hash__
    return module


def local_checkpoint(snapshot, out_dir, attn_implementation=None):
    """The checkpoint layout load_model() expects, with an args/config.json that resolves on this machine.

    load_model (src/inference.py) builds MedMLLMConfig from <checkpoint>/args/config.json, and MedMLLMConfig reads
    the base configs (Phi-3, SigLIP, M3D-CLIP) with the cache_dir_hf stored there: the authors' cache folder,
    where an offline job finds nothing. cache_dir_hf=None means the default cache (HF_HOME). attn_implementation,
    if given, replaces llm_attn_implementation. Every other top-level entry is a symlink to the snapshot.
    """
    with open(os.path.join(snapshot, "args", "config.json"), encoding="utf-8") as f:
        config = json.load(f)
    config["cache_dir_hf"] = None
    if attn_implementation is not None:
        config["llm_attn_implementation"] = attn_implementation
    os.makedirs(os.path.join(out_dir, "args"))
    for entry in os.listdir(snapshot):
        if entry != "args":
            os.symlink(os.path.join(snapshot, entry), os.path.join(out_dir, entry))
    with open(os.path.join(out_dir, "args", "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return out_dir


class Med2E3Adapter:
    name = "med-2e3"
    model_id = "shiym2000/Med-2E3-M3D"
    mode = "qna_per_field"
    text_mlp_source = None  # set by load(): "checkpoint" or "random_seed0"

    def load(self):
        import tempfile
        from types import SimpleNamespace

        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer
        from transformers.utils import is_flash_attn_2_available

        sys.path.insert(0, MED2E3_REPO)
        preload_formatters(MED2E3_REPO)  # before anything imports src.data.template (Python 3.12, see there)
        from src.data.dataset import DataCollatorForMultiModalDataset
        from src.data.preprocess.vision3d_preprocess import Vision3DPreprocess
        from src.data.text_preprocess import TextPreprocess
        from src.inference import load_model

        # Seed 0 (global constraints): if the checkpoint has no text_mlp weights (below), its random initialisation
        # is the same at every load.
        torch.manual_seed(0)
        # The training scripts set llm_attn_implementation=flash_attention_2; without flash-attn use SDPA, which
        # Phi3ForCausalLM supports in transformers 4.45.2.
        attn_implementation = None if is_flash_attn_2_available() else "sdpa"
        checkpoint = local_checkpoint(snapshot_download(self.model_id), tempfile.mkdtemp(prefix="med2e3_ckpt_"),
                                      attn_implementation)
        model = load_model(SimpleNamespace(resume_from_checkpoint=checkpoint, cache_dir_hf=None, llm_dtype=LLM_DTYPE))
        # The TG-IS question projection, which load_model leaves at its random initialisation (see load_text_mlp)
        connector_path = os.path.join(checkpoint, "vision3d_connector", "pytorch_model.bin")
        connector_state = torch.load(connector_path, map_location="cpu") if os.path.exists(connector_path) else {}
        self.text_mlp_source = load_text_mlp(connector_state, model.vision3d_connector.text_mlp)
        print(f"{self.name}: TG-IS text_mlp from {self.text_mlp_source}", flush=True)
        # Stage 3 of inference(): the whole model on the GPU in the eval dtype
        model.to(device="cuda", dtype=torch.bfloat16)
        model.eval()
        self.model = model

        # The parts of MultiModalDataset (mode "eval") and its collator, for one in-memory item per question.
        # MultiModalDataset itself reads a JSON file and loads the question tokenizer from the authors' cache path.
        self.text_preprocess = TextPreprocess(model.tokenizer, CONV_VERSION)
        self.vision3d_preprocess = Vision3DPreprocess(model.vision3d_model.vision3d_processor, None, "eval")
        self.question_tokenizer = AutoTokenizer.from_pretrained(
            QUESTION_TOKENIZER,
            model_max_length=512,
            padding_side="right",
            use_fast=False,
        )
        self.collator = DataCollatorForMultiModalDataset(tokenizer=model.tokenizer, mode="eval")
        self.letter_ids = letter_token_ids(model.tokenizer)

    def _generate(self, volume, prompt):
        import torch

        # MultiModalDataset.__getitem__ (mode "eval") for one conversation turn and one volume
        data_dict = self.text_preprocess(messages=[{"from": "human", "value": prompt}], mode="eval")
        data_dict["vision2d"] = None
        data_dict["vision3d"] = [torch.as_tensor(model_volume(volume, self.vision3d_preprocess))]
        data_dict["vision3d_224"] = None
        # TG-IS scores the slices against the question; the published code does not show what the eval JSONs put
        # in "question", so it is the question text without the image token (best guess).
        question = prompt.replace(IMAGE3D_TOKEN, "").strip()
        text_tensor = self.question_tokenizer(question, max_length=512, truncation=True, padding="max_length",
                                              return_tensors="pt")
        data_dict["questions"] = text_tensor["input_ids"][0]
        data_dict["questions_mask"] = text_tensor["attention_mask"][0]
        batch = self.collator([data_dict])

        # Stage 4 of inference()
        device, model_dtype = "cuda", torch.bfloat16
        for k, v in batch.items():
            if v is not None:
                batch[k] = v.to(device)
                if k == "vision2d" or k == "vision3d" or k == "vision3d_224":
                    batch[k] = v.to(device=device, dtype=model_dtype)
        output, _ = self.model.generate(  # (LLM output, TG-IS slice scores)
            **batch,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            num_beams=1,
            pad_token_id=self.model.tokenizer.pad_token_id,
            eos_token_id=self.model.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )
        answer = self.model.tokenizer.batch_decode(output.sequences, skip_special_tokens=True)[0]
        first = output.scores[0][0].float().cpu().numpy()  # scores of the first generated token
        return answer, letter_probabilities(first, self.letter_ids)

    def predict(self, scan, volumes):
        volume = resize(to01(volumes[0], scan.dataset), INPUT_SHAPE)[None]  # (1, 32, 256, 256), as M3D's .npy

        def ask_fn(question):
            return self._generate(volume, IMAGE3D_PREFIX + question)

        values, log = interrogate(scan.dataset, ask_fn, "A", scan.patient)
        return values, dict(log, text_mlp_source=self.text_mlp_source)  # field logs + where text_mlp came from


if __name__ == "__main__":
    runner.main(Med2E3Adapter())
