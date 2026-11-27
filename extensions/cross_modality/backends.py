"""Model-specific input and inference paths for the development extension."""
import json
import sys
from unittest.mock import patch

import numpy as np

from common import runner
from common.volume import normalize01, to_rgb_images
from extensions.cross_modality import native


def gpu_runtime():
    import torch

    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one allocated CUDA GPU to an extension run")
    torch.manual_seed(0)
    return torch


def one_volume(scan, volumes):
    if len(scan.files) != 1 or len(volumes) != 1:
        raise ValueError("This extension expects one series per scan")
    return volumes[0]


def finish_loading(model, info):
    if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError(f"Incomplete model restoration: {info}")
    if any(p.is_meta for p in model.parameters()):
        raise RuntimeError("Unrestored meta parameters")
    return model.to("cuda").eval()


def generate(model, tokenizer, batch, budget, *, prefix, cast_all=False, tower=None, **kwargs):
    import torch

    input_len = batch["input_ids"].shape[1]
    if input_len + budget > 32768:
        raise ValueError("Input plus output exceeds the extension context budget")
    device_batch = {
        k: v.to(device="cuda", dtype=torch.bfloat16 if v.is_floating_point()
                and (cast_all or k == "pixel_values") else v.dtype)
        if isinstance(v, torch.Tensor) else v for k, v in batch.items()
    }
    hits = []
    hook = tower.register_forward_pre_hook(lambda *_: hits.append(True)) if tower is not None else None
    try:
        with torch.inference_mode():
            answer = model.generate(**device_batch, do_sample=False, num_beams=1,
                                    max_new_tokens=budget, use_cache=True, **kwargs)
    finally:
        if hook is not None:
            hook.remove()
    has_prefix = torch.equal(answer[:, :input_len], device_batch["input_ids"])
    if prefix == "required" and not has_prefix:
        raise ValueError("Unexpected generation prefix")
    if prefix != "continuation" and has_prefix:
        answer = answer[:, input_len:]
    if (tower is not None and not hits) or answer.shape[1] > budget:
        raise ValueError("Native encoder was not called or generation exceeded its budget")
    return tokenizer.batch_decode(answer, skip_special_tokens=True)[0], {
        "token_ids": answer.cpu().tolist(), "hit_token_cap": answer.shape[1] == budget,
    }


def slice_frames(volume, *, ct, blank):
    if volume.ndim != 3 or volume.shape[0] < 32 or not np.isfinite(volume).all():
        raise ValueError("32 finite axial slices required")
    if blank:
        pixels = np.zeros_like(volume)
    elif ct:
        if volume.min() > -500 or volume.max() < 300:
            raise ValueError("Unexpected corrected chest CT intensity range")
        pixels = np.clip((volume + 160) / 400, 0, 1)
    else:
        pixels = normalize01(volume)
    indices = np.linspace(0, len(volume) - 1, 32).round().astype(int).tolist()
    return to_rgb_images(pixels, indices, 224), indices


def slice_batch(kind, processor, frames, prompt):
    if kind == "sgmri":
        content = [{"type": "image", "image": frame} for frame in frames]
        batch = processor.apply_chat_template(
            [{"role": "user", "content": content + [{"type": "text", "text": prompt}]}],
            tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
        )
        batch.pop("token_type_ids", None)
        grid = batch.get("image_grid_thw")
        if grid is None or len(grid) != len(frames):
            raise ValueError("SGMRI frame count mismatch")
    else:
        # NumPy HWC avoids the pinned Hulu processor's PIL width/height swap.
        batch = processor(images=[[np.asarray(f) for f in frames]],
                          conversation=[{"role": "user", "content": [
                              {"type": "video", "num_frames": len(frames)},
                              {"type": "text", "text": prompt}]}],
                          add_system_prompt=False, add_generation_prompt=True, return_tensors="pt")
        grid = batch.get("grid_sizes")
        if grid is None or tuple(grid.shape) != (1, 3) or int(grid[0, 0]) != len(frames) or batch.get("modals") != ["video"]:
            raise ValueError("Hulu video grouping mismatch")
        dims = grid[0].tolist()
        if min(dims) < 1 or int(np.prod(dims)) ** 2 * 16 * 4 > 16 * 1024**3:
            raise ValueError("Hulu dense-attention estimate exceeds the 16 GiB safety bound")
    if batch["input_ids"].shape != batch["attention_mask"].shape or not batch["pixel_values"].numel():
        raise ValueError("Invalid visual input or attention mask")
    return batch


class Slices:
    def __init__(self, kind, assets):
        self.kind, self.path = kind, str(assets["model"])

    def load(self):
        torch = gpu_runtime()
        from transformers import AutoProcessor

        hulu = self.kind == "hulu"
        settings = {} if hulu else {"min_pixels": 224**2, "max_pixels": 280**2}
        self.processor = AutoProcessor.from_pretrained(self.path, trust_remote_code=hulu,
                                                       local_files_only=True, use_fast=False, **settings)
        if hulu:
            from transformers import AutoModelForCausalLM as cls
            self.processor.image_processor.max_tokens = 3072
            self.processor.image_processor.min_tokens = 16
        else:
            from transformers import Qwen3VLForConditionalGeneration as cls
        model, info = cls.from_pretrained(
            self.path, trust_remote_code=hulu, local_files_only=True, use_safetensors=True,
            torch_dtype=torch.bfloat16, attn_implementation="sdpa", output_loading_info=True,
        )
        self.model = finish_loading(model, info)

    def generate(self, scan, volumes, prompt, budget):
        ct, blank = scan.dataset == "ct_rate", scan.image_path.startswith("BLANK__")
        frames, indices = slice_frames(one_volume(scan, volumes), ct=ct, blank=blank)
        batch = slice_batch(self.kind, self.processor, frames, prompt)
        text, detail = generate(self.model, self.processor.tokenizer, batch, budget,
                                prefix="continuation" if self.kind == "hulu" else "required")
        return text, dict(detail, slices=indices, window_hu=[-160, 240] if ct else None)


class Native:
    def __init__(self, kind, assets):
        self.kind, self.assets = kind, assets

    def load(self):
        gpu_runtime()
        self.processor, model, info, self.tower = native.load(self.kind, self.assets)
        self.model = finish_loading(model, info)

    def generate(self, scan, volumes, prompt, budget):
        one_volume(scan, volumes)
        ct, blank = scan.dataset == "ct_rate", scan.image_path.startswith("BLANK__")
        batch, profile, tokenizer = native.prepare(self.kind, self.processor, scan.files[0], prompt, ct, blank)
        rad = self.kind == "radsight"
        text, detail = generate(self.model, tokenizer, batch, budget, prefix="optional" if rad else "required",
                                cast_all=self.kind != "nv", tower=self.tower, **({"modals": ["volume"]} if rad else {}))
        return text, dict(detail, profile=profile, native_encoder_called=True)


class NeuroVFM:
    name = "ext-neurovfm"
    mode = "native_report"

    def __init__(self, assets, model_id):
        self.assets, self.model_id = assets, model_id

    def load(self):
        gpu_runtime()
        sys.path.insert(0, str(self.assets["source"]))
        from neurovfm.pipelines.generator import load_vlm

        path = self.assets["model"]
        required = ("config.json", "language_model/config.json", "vision_encoder.pt", "vision_connector.pt")
        if not all((path / name).is_file() for name in required):
            raise ValueError("Incomplete NeuroVFM local checkpoint")
        self.generator, self.preprocessor = load_vlm(str(path), device="cuda")
        model = self.generator.model
        for part, info in (
            ("encoder", getattr(model, "encoder_loading_info", None)),
            ("connector", getattr(model, "connector_loading_info", None)),
            ("language model", getattr(model.language_model, "loading_info", None)),
        ):
            if not isinstance(info, dict) or any(info.get(k) for k in (
                    "missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
                raise RuntimeError(f"NeuroVFM requires audited strict restoration of {part}")
        if any(p.is_meta for p in model.parameters()):
            raise RuntimeError("Unrestored NeuroVFM meta parameters")
        settings = self.generator.GENERATE_KWARGS
        if (settings["max_new_tokens"], settings["num_beams"], settings["generation_schema"],
                settings["do_sample"]) != (512, 4, "shortreport", False):
            raise ValueError("Unexpected NeuroVFM native generation settings")

    def predict(self, scan, volumes):
        import torch

        one_volume(scan, volumes)
        ct = scan.dataset == "ct_rate"
        modality, windows = ("ct", 3) if ct else ("mri", 1)
        batch = self.preprocessor.load_study(list(scan.files), modality=modality)
        pixels, boundaries = batch["img"], batch["series_cu_seqlens"].tolist()
        if (pixels.ndim != 2 or pixels.shape[1] != 1024 or not torch.isfinite(pixels).all()
                or len(boundaries) != windows + 1 or boundaries[0] != 0 or boundaries[-1] != len(pixels)
                or any(not 0 < b - a <= 20000 for a, b in zip(boundaries, boundaries[1:]))
                or list(batch["mode"]) != [modality] * windows
                or tuple(batch["coords"].shape) != (len(pixels), 3)
                or batch["study_cu_seqlens"].tolist() != [0, len(pixels)]):
            raise ValueError("Invalid NeuroVFM packed study")
        blank = scan.image_path.startswith("BLANK__")
        if blank:
            # Same packed locations and modality; replace all image token values.
            batch = dict(batch, img=torch.zeros_like(pixels))
        model, answers, hits = self.generator.model, [], []
        original = self.generator.model.generate

        def capture(*args, **kwargs):
            answer = original(*args, **kwargs)
            answers.append(answer.detach().cpu())
            return answer

        hook = model.vision_encoder.register_forward_pre_hook(lambda *_: hits.append(True))
        try:
            with patch.object(model, "generate", capture):
                report = self.generator.generate(batch, clinical_context=None)
        finally:
            hook.remove()
        if len(answers) != 1 or not hits:
            raise RuntimeError("NeuroVFM native generation not observed")
        text = self.generator.tokenizer.decode(answers[0].squeeze(0), skip_special_tokens=True).strip()
        details = {"raw_json": text, "report": report, "token_ids": answers[0].tolist(),
                   "packed_windows": windows, "blank_packed_pixels": blank}
        try:
            obj = json.loads(text)
            if not isinstance(obj, dict) or not isinstance(obj.get("exam_type"), str) or not isinstance(obj.get("findings"), list):
                raise ValueError("Expected exam_type and findings")
        except ValueError as exc:
            raise runner.PredictionError("Invalid NeuroVFM native report", details) from exc
        # Native reports are not coerced into the paper's diagnosis ontology.
        return {}, details
