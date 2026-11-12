"""BrainGemma3D (MedGemma 1.5 + 3D-inflated MedSigLIP): the native 3D volume, its fixed report prompt, keyword extraction.

The model reads one (1, 1, D, H, W) volume in [0, 1] and writes a free-text report; the shared keyword extractor
maps the report to the schema. Model code and weights both come from the Hugging Face snapshot.

Usage (inside the job): python3 -m models.braingemma3d.run <runner args>
"""
import numpy as np

from common import runner
from common.parse import extract_from_report
from common.volume import resize, to01

INPUT_SHAPE = (32, 128, 128)  # target_size of the model card Quick Start (training used 64 slices)
# CANONICAL_PROMPT of braingemma3d_architecture.py, trailing " \n" included: the prompt it was trained on.
PROMPT = "Generate a radiology report for this brain MRI FLAIR scan. \n"
MAX_NEW_TOKENS = 512


def to_model_axes(vol):
    """Our axial stack -> the axis layout of BrainGemma3D's NIfTI loader.

    Ours (common.volume.load_canonical): D feet->head, H front->back, W patient right->left.
    load_nifti_volume (BraTS branch): np.transpose(RAS array, (2, 1, 0)) = D feet->head, H back->front,
    W patient left->right. Same depth, H and W reversed (an in-plane 180 degree rotation).
    """
    return np.ascontiguousarray(vol[:, ::-1, ::-1])


class BrainGemma3DAdapter:
    name = "braingemma3d"
    model_id = "praiselab-picuslab/BrainGemma3D"
    mode = "findings_then_extract"

    def load(self):
        import os
        import sys

        import torch
        from huggingface_hub import snapshot_download

        # The snapshot holds the architecture module, vision_model/, language_model/ and the projector.
        local_dir = snapshot_download(self.model_id)
        sys.path.append(local_dir)
        from braingemma3d_architecture import BrainGemma3D

        model = BrainGemma3D(
            vision_model_dir=f"{local_dir}/vision_model",
            language_model_dir=f"{local_dir}/language_model",
            depth=2,
            num_vision_tokens=32,
            freeze_vision=True,
            freeze_language=True,
            device_map={"": 0},
        )
        ckpt = torch.load(os.path.join(local_dir, "projector_vis_scale.pt"), map_location="cuda")
        model.vision_projector.load_state_dict(ckpt["vision_projector"])
        if ckpt.get("vis_scale") is not None:
            if isinstance(ckpt["vis_scale"], torch.Tensor):
                model.vis_scale.data = ckpt["vis_scale"].to("cuda")
            else:
                model.vis_scale.data.fill_(ckpt["vis_scale"])
        model.eval()
        self.model = model

    def _generate(self, volume, prompt):
        import torch

        x = torch.from_numpy(volume)[None, None].to("cuda")  # (1, 1, D, H, W), as load_nifti_volume returns
        with torch.no_grad():
            # temperature=None -> generate_report sets do_sample=False (greedy); its repetition_penalty=1.2,
            # no_repeat_ngram_size=3 and min_new_tokens=10 defaults stay.
            return self.model.generate_report(x, prompt=prompt, max_new_tokens=MAX_NEW_TOKENS,
                                              temperature=None, top_p=None)

    def predict(self, scan, volumes):
        volume = to_model_axes(resize(to01(volumes[0], scan.dataset), INPUT_SHAPE))
        report = self._generate(volume, PROMPT)
        return extract_from_report(report, scan.dataset), {"raw_report": report, "input_shape": list(volume.shape)}


if __name__ == "__main__":
    runner.main(BrainGemma3DAdapter())
