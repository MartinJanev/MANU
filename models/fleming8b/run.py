"""Fleming-VL-8B (InternVL3-8B based, trust_remote_code): the card's 3D path, the full-JSON prompt, JSON answer.

The model card's 3D example reads a (32, 256, 256) array, takes 11 evenly spaced slices along axis 0, min-max
scales each to uint8 grey, repeats it into RGB and makes one 448x448 tile of it; the question is the card's
reasoning prompt, one "<image>\\n" per slice, then our question. The model thinks in <think>...</think> and
answers in <answer>...</answer>; parse_json_answer reads the answer tag. A <think> that never closes (generation
stopped inside the reasoning) gives no values.

Usage: sbatch -J fleming8b jobs/run_model.sh fleming8b --split dev|eval|full [--controls] [--limit N]
       (jobs/run_model.sh runs `python3 /manu/models/fleming8b/run.py --split ...` inside the container)
"""
import numpy as np
from PIL import Image

from common import runner
from common.parse import parse_json_answer
from common.schema import full_prompt
from common.volume import resize, to01

INPUT_SHAPE = (32, 256, 256)  # the only shape the card's convert_npy_to_images accepts
N_SLICES = 11
IMAGE_SIZE = 448
MAX_NEW_TOKENS = 4096  # it reasons in <think> before answering; 2048 cut 8/32 eval answers (the card's 3D example: 1024)
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# REASONING_PROMPT of the card's Quick Start (the default prompt of every inference function there)
REASONING_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, "
    "and the Assistant solves it. The assistant first thinks about the "
    "reasoning process in the mind and then provides the user a concise "
    "final answer in a short word or phrase. The reasoning process and "
    "answer are enclosed within <think> </think> and <answer> </answer> "
    "tags, respectively, i.e., <think> reasoning process here </think>"
    "<answer> answer here </answer>"
)
SLICES = [int(i) for i in np.linspace(0, INPUT_SHAPE[0] - 1, N_SLICES, dtype=int)]  # 0, 3, 6, ..., 27, 31


def slice_images(volume):
    """convert_npy_to_images of the card for an in-memory (32, 256, 256) array, up to its torchvision transform.

    11 RGB PIL images of 448x448. With max_num=1, the card's dynamic_preprocess always picks the single 1x1 tile:
    a resize to 448x448 with Pillow's default filter (bicubic), no thumbnail.
    """
    if volume.shape != INPUT_SHAPE:
        raise ValueError(f"expected shape {INPUT_SHAPE}, got {volume.shape}")
    images = []
    for idx in SLICES:
        slice_img = volume[idx]
        lo, hi = np.min(slice_img), np.max(slice_img)
        if hi - lo == 0:  # normalize_image of the card: per-slice min-max to uint8 (truncated, not rounded)
            grey = np.zeros_like(slice_img, dtype=np.uint8)
        else:
            grey = ((slice_img - lo) / (hi - lo) * 255).astype(np.uint8)
        rgb = Image.fromarray(np.stack([grey, grey, grey], axis=-1))
        images.append(rgb.resize((IMAGE_SIZE, IMAGE_SIZE)))
    return images


class FlemingAdapter:
    name = "fleming-vl-8b"
    model_id = "IQuestLab/Fleming-VL-8B"
    mode = "full_json"

    def load(self):
        import torch
        from transformers import AutoModel, AutoTokenizer

        # load_model() of the card, with use_flash_attn=False as in its 3D example
        self.model = AutoModel.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=False,
            trust_remote_code=True,
        ).eval().cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True, use_fast=False)

    def _generate(self, volume, text, max_new_tokens):
        import torch
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode

        # build_transform(448) of the card
        transform = T.Compose([
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        pixel_values = torch.stack([transform(img) for img in slice_images(volume)])
        pixel_values = pixel_values.to(torch.bfloat16).cuda()
        generation_config = dict(max_new_tokens=max_new_tokens, do_sample=False, num_beams=1)
        response, _ = self.model.chat(
            self.tokenizer,
            pixel_values,
            text,
            generation_config,
            num_patches_list=[1] * N_SLICES,  # one tile per slice
            history=None,
            return_history=True,
        )
        return response

    def predict(self, scan, volumes):
        volume = resize(to01(volumes[0], scan.dataset), INPUT_SHAPE)
        text = f"{REASONING_PROMPT}\n" + "<image>\n" * N_SLICES + full_prompt(scan.dataset)
        raw = self._generate(volume, text, MAX_NEW_TOKENS)
        if raw.rfind(THINK_OPEN) > raw.rfind(THINK_CLOSE):
            obj, problems = None, ["truncated_think"]  # stopped inside the reasoning: any JSON there is not an answer
        else:
            obj, problems = parse_json_answer(raw)
        return obj or {}, {"prompt": text, "raw": raw, "problems": problems, "slices": SLICES}


if __name__ == "__main__":
    runner.main(FlemingAdapter())
