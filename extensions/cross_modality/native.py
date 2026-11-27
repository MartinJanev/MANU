"""Physical-volume profiles ported from the CT and MRI-transfer pilots."""
import ast
from contextlib import contextmanager
import importlib.util
import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import warnings

import numpy as np


@contextmanager
def in_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def read_volume(path, *, ct, blank):
    import nibabel as nib

    img = nib.load(path)
    voxels = img.get_fdata(dtype=np.float32, caching="unchanged")
    if (voxels.ndim != 3 or not np.isfinite(voxels).all() or voxels.min() == voxels.max()
            or not np.isfinite(img.affine).all() or not np.allclose(img.affine[3], [0, 0, 0, 1])
            or abs(np.linalg.det(img.affine[:3, :3])) < 1e-8
            or any(not 0 < s < 10 for s in img.header.get_zooms())
            or max(np.array(voxels.shape) * img.header.get_zooms()) > 1000):
        raise ValueError("Invalid physical volume geometry or intensity")
    if ct:
        if voxels.min() > -500 or voxels.max() < 300:
            raise ValueError("Unexpected corrected chest CT intensity range")
        bounds = None
    elif not blank:
        positive = voxels[voxels > 0]
        if not positive.size:
            raise ValueError("MRI numeric proxy requires positive source intensities")
        low, high = np.percentile(positive, [0.5, 99.5])
        if low >= high:
            raise ValueError("Degenerate MRI percentile range")
        voxels = (np.clip((voxels - low) / (high - low), 0, 1) * 2000 - 1000).astype(np.float32)
        bounds = [float(low), float(high)]
    else:
        bounds = None
    if blank:
        voxels = np.full_like(voxels, -1000)
    return nib.Nifti1Image(voxels, img.affine), bounds


def physical_volume(source, out, kind):
    import SimpleITK as sitk

    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(4)
    img = sitk.ReadImage(str(source), sitk.sitkFloat32)
    img = sitk.DICOMOrient(img, "RAI" if kind == "photon" else "RAS")
    spacing = [1., 1., 1.] if kind == "photon" else [1., 1., 5.]
    size = [max(1, round(s * z / t)) for s, z, t in zip(img.GetSize(), img.GetSpacing(), spacing)]
    if np.prod(size) > 256 * 1024**2:
        raise ValueError("Physical preparation size bound")
    img = sitk.Resample(img, size, sitk.Transform(), sitk.sitkLinear, img.GetOrigin(),
                        spacing, img.GetDirection(), -1000., sitk.sitkFloat32)
    img = sitk.Clamp(img, lowerBound=-1000., upperBound=1000.)
    if kind == "photon":
        size = np.array(img.GetSize())
        lower = np.maximum((size - 364) // 2, 0)
        upper = np.maximum(size - 364 - lower, 0)
        img = sitk.Crop(img, lower.tolist(), upper.tolist())
        missing = 364 - np.array(img.GetSize())
        lo = missing // 2
        sitk.WriteImage(sitk.ConstantPad(img, lo.tolist(), (missing - lo).tolist(), -1000.), str(out))
        return None

    import torch
    return torch.from_numpy(np.ascontiguousarray((sitk.GetArrayFromImage(img) + 1000.) / 2000.))[None]


def require_tensor(tensor, shape, bounds, blank):
    import torch

    if (tuple(tensor.shape) != shape or not torch.isfinite(tensor).all()
            or tensor.min() < bounds[0] - 1e-4 or tensor.max() > bounds[1] + 1e-4
            or (not blank and tensor.min() == tensor.max())):
        raise ValueError(f"Invalid native tensor, expected {shape} in {bounds}")


def photon_loader(source):
    # Reuse the pinned author's standalone method without importing Swift's CLI.
    import monai.transforms as mtf
    import nibabel as nib
    import SimpleITK as sitk
    import torch

    tree = ast.parse((source / "swift/llm/template/template/qwen.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PhotonTemplate")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "load_volume")
    scope = dict(torch=torch, np=np, nib=nib, sitk=sitk, mtf=mtf)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<reviewed PhotonTemplate.load_volume>", "exec"), scope)
    return scope["load_volume"]


def photon_batch(processor, source, prompt, blank, folder):
    import torch

    prepared = folder / "physical.nii.gz"
    physical_volume(source, prepared, "photon")
    volume = processor["load_volume"](None, str(prepared))
    require_tensor(volume, (140, 1, 280, 280), (0, 1), blank)
    p = processor["processor"]
    batch = p.image_processor(images=None, videos=[volume], return_tensors="pt",
                              do_resize=False, do_rescale=False, do_normalize=False)
    grid = batch["video_grid_thw"]
    if grid.tolist() != [[10, 20, 20]]:
        raise ValueError("Photon native grid mismatch")
    batch["second_per_grid_ts"] = [p.image_processor.temporal_patch_size / 1.]
    marker = "<|video_pad|>"
    text = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
            f"<|vision_start|>{marker}<|vision_end|> {prompt}<|im_end|>\n<|im_start|>assistant\n")
    ids = p.tokenizer(text, return_tensors="pt")["input_ids"]
    if int((ids == 151656).sum()) != 1:
        raise ValueError("Expected one Photon volume marker")
    count = int(grid.prod()) // p.image_processor.merge_size**2
    expanded = p.tokenizer(text.replace(marker, marker * count, 1), return_tensors="pt")["input_ids"]
    if int((expanded == 151656).sum()) != count:
        raise ValueError("Photon token expansion mismatch")
    return dict(batch, input_ids=expanded, attention_mask=torch.ones_like(expanded)), p.tokenizer


def radsight_batch(processor, source, prompt, blank):
    volume = physical_volume(source, None, "radsight")
    messages = [{"role": "user", "content": [{"type": "video", "num_frames": 12},
                                             {"type": "text", "text": prompt}]}]
    batch = processor(images=[volume], text=messages, modal="volume", merge_size=1, return_tensors="pt")
    require_tensor(batch["pixel_values"], (1, 1, 96, 256, 384), (0, 1), blank)
    return batch, processor.tokenizer


def nv_batch(processor, source, prompt, ct, blank):
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        batch = processor(text=text, images3d=[str(source)], anatomy_region="chest" if ct else None,
                          return_tensors="pt")
    for notice in recorded:
        if not (notice.category is FutureWarning
                and notice.filename.endswith("/monai/utils/deprecate_utils.py")
                and str(notice.message).startswith("monai.transforms.spatial.dictionary Orientationd.__init__:labels:")):
            raise RuntimeError(f"NV processor warning needs review: {notice.message}")
    require_tensor(batch["pixel_values"], (1, 1, 192, 192, 192), (-1, 1), blank)
    if batch["image_grid_thw"].tolist() != [[24, 24, 24]]:
        raise ValueError("NV native grid mismatch")
    return batch, processor.tokenizer


def clinfusion_batch(adapter, image, prompt, blank):
    processor = adapter.processor
    volumes, images = adapter._pre_sample_slices_from_volume([image], processor)
    if not volumes or len(images) != 4:
        raise ValueError("ClinFusion hybrid volume and four slices required")
    content = [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]
    text = processor.apply_chat_template([{"role": "user", "content": content}],
                                         tokenize=False, add_generation_prompt=True)
    batch = processor(text=[text], images=[images], padding=True, return_tensors="pt",
                      images_kwargs={"volume": [volumes]})
    require_tensor(batch["pixel_values_pe_3d"], (1, 1, 32, 256, 256), (0, 1), blank)
    return batch, processor.tokenizer


def prepare(kind, processor, path, prompt, ct, blank):
    import nibabel as nib

    image, bounds = read_volume(path, ct=ct, blank=blank)
    profile = {
        "nv": "NV native 192 cube; chest crop for CT, centered for MRI",
        "photon": "RAI 1mm, 364 cube, then author's load_volume",
        "radsight": "RAS 1x1x5mm, full volume; no mask crop",
        "clinfusion": "author object path, native volume plus four slices",
    }[kind]
    with TemporaryDirectory(prefix="manu-volume-") as tmp:
        folder = Path(tmp)
        source = folder / "volume.nii.gz"
        if kind != "clinfusion":
            nib.save(image, source)
        if kind == "nv":
            batch, tokenizer = nv_batch(processor, source, prompt, ct, blank)
        elif kind == "photon":
            batch, tokenizer = photon_batch(processor, source, prompt, blank, folder)
        elif kind == "radsight":
            batch, tokenizer = radsight_batch(processor, source, prompt, blank)
        else:
            batch, tokenizer = clinfusion_batch(processor, image, prompt, blank)
    if batch["input_ids"].shape[1] > 16000:
        raise ValueError("Native input exceeds the 16000-token budget")
    batch = {k: v.as_tensor() if hasattr(v, "as_tensor") else v for k, v in batch.items()}
    return batch, {"preprocessing": profile, "intensity": "HU" if ct else "MRI numeric proxy, not HU",
                   "mri_percentiles": bounds, "blank": blank}, tokenizer


def strict_shards(model, path):
    from safetensors import safe_open

    path = Path(path)
    index = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    state = model.state_dict()
    if set(index) != set(state):
        raise ValueError("ClinFusion final checkpoint/model key sets differ")
    for filename in set(index.values()):
        if Path(filename).name != filename or not filename.endswith(".safetensors"):
            raise ValueError("Invalid ClinFusion shard path")
        keys = {k for k, v in index.items() if v == filename}
        with safe_open(path / filename, framework="pt", device="cpu") as shard:
            if set(shard.keys()) != keys or any(
                tuple(shard.get_slice(k).get_shape()) != tuple(state[k].shape) for k in keys
            ):
                raise ValueError("ClinFusion final shard does not match model")


def load(kind, assets):
    import torch
    from transformers import AutoConfig, AutoProcessor, AutoTokenizer

    path = str(assets["model"])
    if "source" in assets:
        sys.path.insert(0, str(assets["source"]))
    kwargs = dict(local_files_only=True, attn_implementation="sdpa", output_loading_info=True)
    if kind == "nv":
        from transformers import AutoModelForImageTextToText

        processor = AutoProcessor.from_pretrained(path, trust_remote_code=True, local_files_only=True)
        model, info = AutoModelForImageTextToText.from_pretrained(
            path, trust_remote_code=True, use_safetensors=True, dtype=torch.bfloat16, **kwargs)
        return processor, model, info, model.model.vision3d

    if kind == "clinfusion":
        from custom_model.medevalkit_adapter_qwen3_vl import MONAI_AVAILABLE, MedEvalKitAdapter
        from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

        if not MONAI_AVAILABLE or is_deepspeed_zero3_enabled():
            raise RuntimeError("ClinFusion requires MONAI and a single-process, non-ZeRO load")

        # Local paths for the author's relative bootstrap dependencies; no writes in the repo.
        with TemporaryDirectory(prefix="manu-clinfusion-") as tmp:
            cache = Path(tmp) / "cache/models"
            cache.mkdir(parents=True)
            for name, asset in (("Qwen3-VL-8B-Instruct", "qwen"), ("dinov2-large", "dinov2"),
                                ("CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup", "convnext")):
                (cache / name).symlink_to(assets[asset], target_is_directory=True)
            with in_directory(tmp):
                adapter = MedEvalKitAdapter(path, {}, {"max_new_tokens": 512, "temperature": 0.})
        model = adapter.llm
        info = getattr(adapter, "checkpoint_load_info", None)
        if not isinstance(info, dict):
            raise RuntimeError("Use the strict ClinFusion source from prepare_source")
        return adapter, model, info, model.model.visual.pe_3d.visual

    config = AutoConfig.from_pretrained(path, local_files_only=True)
    config._attn_implementation = "sdpa"
    if kind == "photon":
        spec = importlib.util.spec_from_file_location(
            "pinned_photon_s1", assets["source"] / "swift/photon/modeling_photon_s1.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        cls = module.Qwen2_5_VLForConditionalGeneration
        config.vision_config._attn_implementation = "sdpa"
        processor = {"processor": AutoProcessor.from_pretrained(path, local_files_only=True),
                     "load_volume": photon_loader(assets["source"])}
    else:
        from radsight.model.processor import RadSightProcessor
        from radsight.model.radsight_qwen2 import RadSightQwen3ForCausalLM as cls

        siglip = assets["siglip"]
        # The upstream constructor selects its implementation from the path name.
        if "VL3-SigLIP-NaViT" not in siglip.name:
            raise ValueError("Use a RadSight SigLIP directory named VL3-SigLIP-NaViT...")
        config.vision_encoder = str(siglip)
        config.mm_attn_implementation = "sdpa"
        config.use_token_compression = False
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=False)
        processor = RadSightProcessor(image_processor=None, tokenizer=tokenizer)
    model, info = cls.from_pretrained(path, config=config, weights_only=True, torch_dtype=torch.bfloat16, **kwargs)
    if kind == "radsight":
        encoder = model.get_vision_encoder(modal="image")
        if type(encoder).__name__ != "RadSightVisionEncoder" or type(encoder.vision_encoder).__name__ != "RadSightVisionEncoderModel":
            raise ValueError("Wrong RadSight 2D encoder constructor")
    tower = model.visual if kind == "photon" else model.get_vision_encoder(modal="volume")
    return processor, model, info, tower
