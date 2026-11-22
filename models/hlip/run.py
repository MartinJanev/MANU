"""HLIP (CLIP-style 3D brain MRI encoder, ViT-L with scan + study attention): zero-shot classification, no text output.

The volume is scored against one text prompt per class (class names of HLIP's Pub-Brain-5 and MR-RATE zero-shot
evaluations); the softmax probabilities are averaged per diagnosis group, then renormalised. Only diagnosis_name and
diagnosis_confidence are filled: HLIP has no output for any other field. Model code comes from the HLIP git clone
(third_party/hlip), weights from the Hugging Face snapshot.

Usage: sbatch -J hlip jobs/run_model.sh hlip --split dev|eval|full [--controls] [--limit N]
       (jobs/run_model.sh runs `python3 /manu/models/hlip/run.py --split ...` inside the container)
"""
import numpy as np

from common import runner
from common.volume import to01

HLIP_SRC = "/third_party/hlip/src"
MODEL_NAME = "clip_vit_large_scan_study_dualdinotxt1568"  # src/hlip/model_configs/<MODEL_NAME>.json
DEPTH, SIZE, CROP = 48, 256, 224
# Normalize(mean(IMAGENET_DEFAULT_MEAN), mean(IMAGENET_DEFAULT_STD)) of zeroshot_pubbrain5.py, in float32 as torch
MEAN = np.float32(np.mean(np.array((0.485, 0.456, 0.406), dtype=np.float32)))
STD = np.float32(np.mean(np.array((0.229, 0.224, 0.225), dtype=np.float32)))
NORMAL_CLASS = "no significant abnormalities"

# Class names copied verbatim from src/hlip/zeroshot_metadata_pubbrain5.py (first five) and
# src/hlip/zeroshot_metadata_mrrate.py (the rest, in file order), each mapped to our diagnosis group.
# MR-RATE classes that fit none of the four groups are left out (listed in NOTES.md).
CLASS_GROUPS = {
    "no significant abnormalities": "normal",
    "acute stroke": "stroke",
    "glioma": "tumor",
    "meningioma": "tumor",
    "metastasis": "tumor",
    "Cerebral infarction": "stroke",
    "Cerebral hemorrhage": "stroke",
    "Metastatic malignant neoplasm to brain": "tumor",
    "Intracranial meningioma": "tumor",
    "Demyelinating disease of central nervous system": "multiple sclerosis",
    "Lacunar infarct": "stroke",
    "Schwannoma": "tumor",
    "Watershed infarct": "stroke",
    "Glioma": "tumor",
    "Pituitary adenoma": "tumor",
}
GROUPS = ("normal", "tumor", "stroke", "multiple sclerosis")


def prompt(c):
    """TEMPLATES[1] of zeroshot_metadata_pubbrain5.py, its wording of the normal class included."""
    if c == NORMAL_CLASS:
        return "This study shows: no significant abnormalities."
    return f"This study shows: {c} identified."


def _linear_indices(n_in, n_out):
    """Source indices and weights of F.interpolate(mode='bilinear', align_corners=False) along one axis."""
    scale = np.float32(n_in) / np.float32(n_out)
    src = np.maximum(scale * (np.arange(n_out, dtype=np.float32) + np.float32(0.5)) - np.float32(0.5), 0)
    i0 = np.minimum(np.floor(src).astype(np.int64), n_in - 1)
    i1 = np.minimum(i0 + 1, n_in - 1)
    lam = np.clip(src - i0, 0, 1).astype(np.float32)
    return i0, i1, lam


def _nearest_exact_indices(n_in, n_out):
    """Source indices of F.interpolate(mode='nearest-exact') along one axis."""
    scale = np.float32(n_in) / np.float32(n_out)
    src = np.floor((np.arange(n_out, dtype=np.float32) + np.float32(0.5)) * scale).astype(np.int64)
    return np.minimum(src, n_in - 1)


def to_model_axes(vol):
    """Our axial stack -> the axis layout of HLIP's Pub-Brain-5 scans.

    Ours (common.volume.load_canonical): D feet->head, H front->back, W patient right->left.
    HLIP (load_nifti_file in data/pub_brain_5/*/process.py of its v1.0 tag): SimpleITK reorientation to 'RPI',
    then GetArrayFromImage = D head->feet, H front->back, W patient left->right. Same rows, D and W reversed.
    """
    return np.ascontiguousarray(vol[::-1, :, ::-1])


def preprocess(vol01):
    """to01 output (our canonical axes) -> (48, 224, 224) float32, the image steps of zeroshot_pubbrain5.py.

    PubBrain5Dataset.__getitem__ divides the stored uint8 scan by 255 (to01 gives the same [0, 1] range), pads
    H and W to a square with zeros, resizes every slice to 256x256 (bilinear), picks 48 slices (nearest-exact),
    crops 16:240 in-plane and normalizes with the mean ImageNet mean/std. Here in numpy, without torch.
    """
    vol = to_model_axes(np.asarray(vol01, dtype=np.float32))
    d, h, w = vol.shape

    # padding to the longest side
    size = max(h, w)
    pad_h, pad_w = size - h, size - w
    top, left = pad_h // 2, pad_w // 2
    vol = np.pad(vol, ((0, 0), (top, pad_h - top), (left, pad_w - left)))

    # resize to 256 (bilinear, slice by slice), 48 slices (nearest-exact), crop to 224
    i0, i1, lam = _linear_indices(size, SIZE)
    vol = vol[:, i0, :] * (1 - lam)[None, :, None] + vol[:, i1, :] * lam[None, :, None]
    vol = vol[:, :, i0] * (1 - lam)[None, None, :] + vol[:, :, i1] * lam[None, None, :]
    vol = vol[_nearest_exact_indices(d, DEPTH)]
    lo = (SIZE - CROP) // 2
    vol = vol[:, lo:lo + CROP, lo:lo + CROP]

    # normalize
    return np.ascontiguousarray((vol - MEAN) / STD, dtype=np.float32)


class HLIPAdapter:
    name = "hlip"
    model_id = "zch0414/clip-vit_large-scan_study-dualdinotxt1568"
    mode = "clip_or_heads"

    def load(self):
        import json
        import os
        import sys
        from pathlib import Path

        import torch
        from huggingface_hub import snapshot_download

        sys.path.append(HLIP_SRC)
        from hlip import visual_encoder  # noqa: F401  (registers HLIP's encoders with timm)
        from open_clip import build_zero_shot_classifier, create_model_and_transforms, get_tokenizer
        from open_clip.factory import _MODEL_CONFIGS

        # create model (as main() of zeroshot_pubbrain5.py)
        config_dir = os.path.join(HLIP_SRC, "hlip", "model_configs")
        for c in os.listdir(config_dir):
            m, e = os.path.splitext(c)
            if e.lower() == ".json":
                with open(os.path.join(config_dir, c), "r") as f:
                    _MODEL_CONFIGS[m] = json.load(f)
        # pretrained_text=False: BiomedBERT is built from its config only; the checkpoint below holds every weight.
        model, _, _ = create_model_and_transforms(MODEL_NAME, device="cuda", precision="amp", output_dict=True,
                                                  pretrained_text=False)

        # load checkpoint
        local_dir = snapshot_download(repo_id=self.model_id)
        state_dict = torch.load(Path(local_dir) / "pytorch_model.bin", map_location="cpu")
        sd = state_dict["state_dict"] if isinstance(state_dict, dict) and "state_dict" in state_dict else state_dict
        model.load_state_dict(sd)
        model.eval()
        tokenizer = get_tokenizer(MODEL_NAME)

        # build classifier (as run() of zeroshot_pubbrain5.py, with our classes and template)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            self.classifier = build_zero_shot_classifier(
                model,
                tokenizer=tokenizer,
                classnames=list(CLASS_GROUPS),
                templates=(prompt,),
                num_classes_per_batch=None,  # all
                device="cuda",
                use_tqdm=False,
            )
        self.model = model

    def _score(self, volume):
        import torch

        # [batch 1, scans 1, channel 1, D, H, W] in float16, as PubBrain5Dataset returns it
        image = torch.from_numpy(volume)[None, None, None].to(device="cuda", dtype=torch.float16)
        with torch.inference_mode(), torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            model_out = self.model(image=image)
            logit_scale = model_out["logit_scale"]
            image_features = model_out["image_features"][:, 0, :]
            logits_per_image = logit_scale * image_features @ self.classifier
        probs = torch.softmax(logits_per_image.float(), dim=-1)[0].cpu().tolist()
        return dict(zip(CLASS_GROUPS, probs))

    def predict(self, scan, volumes):
        volume = preprocess(to01(volumes[0], scan.dataset))
        probs = self._score(volume)
        # the mean, not the sum: with 8 tumor classes against 1 normal, a sum says tumor when the model has no idea
        means = {}
        for g in GROUPS:
            members = [float(p) for c, p in probs.items() if CLASS_GROUPS[c] == g]
            means[g] = sum(members) / len(members)
        total = sum(means.values())
        groups = {g: m / total for g, m in means.items()}
        name = max(groups, key=groups.get)
        values = {"diagnosis_name": name, "diagnosis_confidence": groups[name]}
        return values, {"class_probabilities": probs, "group_probabilities": groups, "input_shape": list(volume.shape)}


if __name__ == "__main__":
    runner.main(HLIPAdapter())
