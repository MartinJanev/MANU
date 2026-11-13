"""MedM-VL-CT-3B-en (M3D-CLIP 3D encoder + spatial pooling + Llama-3.2-3B-Instruct, MedM-VL's `lvlm` code, trained
on M3D-Data): one four-option question per schema field through the shared question bank.

The volume goes in as a (32, 256, 256) array in [-1000, 1000] (pseudo-HU for MRI, the clipped HU for CT-RATE),
in place of the `.npy` that the repo's MultiModalDataset loads; the M3D-CLIP processor (lvlm/model/encoder/m3dclip.py)
then min-max scales it to [0, 1], crops the foreground and resizes it to 32x256x256 again. Each question is
"<image3d>\\n" + "{Q} (a) ... (d) ...", in the "llama3" conversation template; decoding is greedy (16 new tokens),
and the letter probabilities come from the scores at the letter of an answer "(c) ...". Model code comes from the
MedM-VL git clone (third_party/MedM-VL, at the commit whose config format the checkpoint has), weights from the
Hugging Face snapshots.

Usage: sbatch -J medmvl3b jobs/run_model.sh medmvl3b --split dev|eval|full [--controls] [--limit N]
       (jobs/run_model.sh runs `python3 /manu/models/medmvl3b/run.py --split ...` inside the container)
"""
import importlib.util
import json
import os
import sys

import numpy as np

from common import runner
from common.qna import interrogate
from common.volume import resize, to01, to_pseudo_hu

MEDMVL_REPO = "/third_party/MedM-VL"  # its modules import each other as `lvlm.*`, so the repo root goes on sys.path
INPUT_SHAPE = (32, 256, 256)  # the processor's Resize target and M3D-CLIP's img_size
IMAGE3D_TOKEN = "<image3d>"  # DEFAULT_IMAGE3D_TOKEN of lvlm/utils/constants.py
IMAGE3D_PREFIX = IMAGE3D_TOKEN + "\n"  # as in the repo's inference examples
CONV_VERSION = "llama3"  # --conv_version of the CT-3B-en eval scripts (docs/param_interpretation.md: llama3)
MODEL_DTYPE = "bfloat16"  # --model_dtype of the CT-3B-en eval scripts
DEVICE = "cuda"
MAX_NEW_TOKENS = 16
LETTERS = "abcd"


def model_input(volume, dataset):
    """The canonical volume as the (32, 256, 256) float32 array in [-1000, 1000] that goes in place of the `.npy`.

    MRI: percentile scaling, resize, pseudo-HU. CT-RATE: to01 is the [-1000, 1000] HU window and to_pseudo_hu its
    inverse, so this is the real HU clipped to the window (and resized), not a rescaling.
    """
    return to_pseudo_hu(resize(to01(volume, dataset), INPUT_SHAPE)).astype(np.float32)


def model_volume(volume, processor):
    """The array through the M3D-CLIP processor in mode "eval" (min-max to [0, 1], CropForeground, Resize).

    A constant volume (the runner's blank control) min-max scales to zeros, where CropForeground finds no voxel > 0
    and crops to an empty array that Resize rejects; it is already 32x256x256, so it becomes that zero volume with
    the processor's channel axis, without the crop.
    """
    if volume.max() == volume.min():
        return np.zeros((1,) + volume.shape, dtype=np.float32)
    return processor(volume, mode="eval")


def letter_token_ids(tokenizer):
    """Token ids of " (" and of the letters a-d in an answer "ASSISTANT: (c) ...".

    The llama3 template's generation prompt ends with "ASSISTANT:", and Llama-3's tokenizer splits " (c)" into
    " (", "c", ")", so the letter is the second answer token.
    """
    prefix = tokenizer("ASSISTANT:", add_special_tokens=False).input_ids
    opens, ids = set(), []
    for letter in LETTERS:
        full = tokenizer("ASSISTANT: (" + letter + ")", add_special_tokens=False).input_ids
        if full[:len(prefix)] != prefix or len(full) != len(prefix) + 3:
            raise ValueError(f"answer '({letter})' is not ' (', letter, ')' after 'ASSISTANT:': {full}")
        opens.add(full[len(prefix)])
        ids.append(full[len(prefix) + 1])
    if len(opens) != 1:
        raise ValueError(f"the opening parenthesis differs between letters: {opens}")
    return opens.pop(), ids


def letter_probabilities(scores, ids):
    """Softmax of one position's scores at the four letter ids, renormalised over the four.

    None if no letter has a finite score.
    """
    letters = np.asarray(scores, dtype=np.float64)[list(ids)]
    if not np.isfinite(letters.max()):
        return None
    e = np.exp(letters - letters.max())
    return [float(p) for p in e / e.sum()]


def answer_letter_probabilities(tokens, scores, open_id, letter_ids):
    """Letter probabilities at the answer's letter position, from the generated tokens and the per-step scores.

    The four answers "(a)" ... "(d)" share their first token " (", so renormalised over the four their
    probabilities are those of the letter after " (": the scores of the second generated token. None if the answer
    does not start with " (" and another token (then the second step's scores are not conditioned on " (").
    """
    if len(tokens) < 2 or len(scores) < 2 or tokens[0] != open_id:
        return None
    return letter_probabilities(scores[1], letter_ids)


def preload_formatters(repo):
    """Import MedM-VL's lvlm/dataset/template/formatter.py under its package name, with hashable formatter classes.

    The conversation templates (lvlm/dataset/template/*.py) use EmptyFormatter/StringFormatter instances as dataclass
    field defaults, which Python >= 3.11 rejects as mutable ("use default_factory"); the repo targets Python 3.10,
    the image has 3.12. Registered before the template package is imported, and with the identity hash, the module
    gives the templates their 3.10 behaviour (one shared default instance per template class).
    """
    name = "lvlm.dataset.template.formatter"
    path = os.path.join(repo, "lvlm", "dataset", "template", "formatter.py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    for cls in (module.EmptyFormatter, module.StringFormatter):
        cls.__hash__ = object.__hash__
    return module


def local_checkpoint(snapshot, out_dir, attn_implementation=None):
    """The checkpoint folder with a config.json that resolves on this machine.

    LVLMConfig.from_pretrained and the model's __init__ load the base models by name (Llama-3.2-3B-Instruct,
    M3D-CLIP) with the cache_dir_hf stored in config.json: the authors' cache folder, where an offline job finds
    nothing. cache_dir_hf=None means the default cache (HF_HOME). attn_implementation, if given, replaces
    llm_attn_implementation. Every other entry is a symlink to the snapshot.
    """
    with open(os.path.join(snapshot, "config.json"), encoding="utf-8") as f:
        config = json.load(f)
    config["cache_dir_hf"] = None
    if attn_implementation is not None:
        config["llm_attn_implementation"] = attn_implementation
    os.makedirs(out_dir, exist_ok=True)
    for entry in os.listdir(snapshot):
        if entry != "config.json":
            os.symlink(os.path.join(snapshot, entry), os.path.join(out_dir, entry))
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return out_dir


class MedMVLAdapter:
    name = "medm-vl-ct-3b"
    model_id = "shiym2000/MedM-VL-CT-3B-en"
    mode = "qna_per_field"

    def load(self):
        import tempfile

        import torch
        from huggingface_hub import snapshot_download
        from transformers.utils import is_flash_attn_2_available

        sys.path.insert(0, MEDMVL_REPO)
        preload_formatters(MEDMVL_REPO)  # before anything imports lvlm.dataset.template (Python 3.12, see there)
        from lvlm.dataset.dataset import DataCollatorForMultiModalDataset
        from lvlm.dataset.template import TEMPlATE_FACTORY
        from lvlm.model.modeling_lvlm import LVLMForConditionalGeneration

        torch.manual_seed(0)  # seed 0 (global constraints); greedy decoding draws no random numbers
        # The checkpoint stores llm_attn_implementation=flash_attention_2; without flash-attn use SDPA, which
        # LlamaForCausalLM supports.
        attn_implementation = None if is_flash_attn_2_available() else "sdpa"
        checkpoint = local_checkpoint(snapshot_download(self.model_id), tempfile.mkdtemp(prefix="medmvl_ckpt_"),
                                      attn_implementation)
        # Stage 1 of lvlm/inference.py: from_pretrained (the base models by name, then the whole checkpoint's state
        # dict over them), then the eval dtype on the GPU. The loading info shows that every module was restored.
        model, info = LVLMForConditionalGeneration.from_pretrained(checkpoint, output_loading_info=True)
        print(f"{self.name}: checkpoint keys missing {info['missing_keys']}, unexpected {info['unexpected_keys']}, "
              f"mismatched {info['mismatched_keys']}", flush=True)
        if info["missing_keys"]:  # from_pretrained gives such weights a random initialisation
            raise RuntimeError(f"{self.name}: weights missing from the checkpoint: {info['missing_keys']}")
        # Llama-3.2-3B ties lm_head to embed_tokens, and the checkpoint stores only embed_tokens. When the stored and
        # the loaded dtype are equal, transformers 4.43 assigns the loaded tensors instead of copying them, which
        # unties lm_head and leaves it at Llama-3.2's own embeddings; the outer model's tie_weights() does not reach
        # the LLM. Tying again puts lm_head back on the checkpoint's embeddings (a no-op when it is still tied).
        tied = model.llm.get_output_embeddings().weight is model.llm.get_input_embeddings().weight
        model.llm.tie_weights()
        model.to(dtype=getattr(torch, MODEL_DTYPE), device=DEVICE)
        model.eval()
        self.model = model
        print(f"{self.name}: lm_head tied after from_pretrained {tied}, now "
              f"{model.llm.get_output_embeddings().weight is model.llm.get_input_embeddings().weight}; "
              f"LLM rope_scaling {model.llm.config.rope_scaling}, attention {model.llm.config._attn_implementation}",
              flush=True)

        # The parts of MultiModalDataset (mode "eval") and its collator, for one in-memory item per question.
        # MultiModalDataset itself reads its items from a JSON list and the volumes from .npy files.
        self.template = TEMPlATE_FACTORY[CONV_VERSION]()
        self.processor = model.encoder_image3d.processor
        self.collator = DataCollatorForMultiModalDataset(tokenizer=model.tokenizer, mode="eval")
        self.open_id, self.letter_ids = letter_token_ids(model.tokenizer)

    def _generate(self, volume, prompt):
        import torch

        tokenizer = self.model.tokenizer
        # MultiModalDataset.__getitem__ (mode "eval") for one conversation turn and one volume
        data_dict = self.template.encode(messages=[{"from": "human", "value": prompt}], tokenizer=tokenizer,
                                         mode="eval")
        data_dict["image"] = None
        data_dict["image3d"] = [torch.as_tensor(model_volume(volume, self.processor))]
        batch = self.collator([data_dict])

        # Stage 4 of lvlm/inference.py
        device, model_dtype = DEVICE, getattr(torch, MODEL_DTYPE)
        for k, v in batch.items():
            if v is not None:
                batch[k] = v.to(device)
                if k == "image" or k == "image3d":
                    batch[k] = v.to(dtype=model_dtype, device=device)
        with torch.no_grad():
            output = self.model.generate(
                **batch,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
        answer = tokenizer.batch_decode(output.sequences, skip_special_tokens=True)[0]
        tokens = output.sequences[0].tolist()  # only the new tokens: the LLM got inputs_embeds, no input_ids
        scores = [s[0].float().cpu().numpy() for s in output.scores[:2]]
        return answer, answer_letter_probabilities(tokens, scores, self.open_id, self.letter_ids)

    def predict(self, scan, volumes):
        volume = model_input(volumes[0], scan.dataset)

        def ask_fn(question):
            return self._generate(volume, IMAGE3D_PREFIX + question)

        return interrogate(scan.dataset, ask_fn, "a")  # (values, per-field log)


if __name__ == "__main__":
    runner.main(MedMVLAdapter())
