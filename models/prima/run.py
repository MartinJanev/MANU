"""Prima (brain MRI study model, MLNeurosurg/prima; MIT): 52 diagnosis heads, no text output.

Each scan becomes a DICOM study (common.dicom: one series per NIfTI, geometry kept) and goes through Prima's own
end-to-end pipeline (end-to-end_inference_pipeline/pipeline.py, unchanged), step by step as its __main__ runs them:
load_mri_study -> run_tokenizer_model -> run_prima_model. The VQ-VAE and the full model are loaded once with the
pipeline's own loaders and handed to every per-scan Pipeline (its cleanup only drops the pipeline's references).
Series: MS = T1 and FLAIR (two series, the patient's folder); IXI and MOLAB = one series (Prima was trained on
studies with at least 2 series).

A diagnosis head returns logit - thresh, thresh being that head's best-accuracy cut-off, so >= 0 is Prima's own
"present". Per head p = sigmoid(logit - thresh) (0.5 = Prima's cut-off); a group's probability is the largest p of
its heads. Diagnosis: the most probable group among GROUP_HEADS, or "normal" when every one of those heads is below
NORMAL_THRESHOLD (diagnosis_confidence = 1 - the largest of them). Every other field is null. All 52 values,
thresholds and p, the referral and priority outputs go to the details file.

Usage: sbatch -J prima jobs/run_model.sh prima --split dev|eval|full [--controls] [--limit N]
       (jobs/run_model.sh runs `python3 /manu/models/prima/run.py --split ...` inside the container)
"""
import importlib.util
import json
import math
import os
import sys
import tempfile

import nibabel as nib
import numpy as np

from common import runner
from common.dicom import native_from_canonical, write_dicom_series

PRIMA_SRC = "/third_party/prima"
# Where Prima's own tools/download_models_and_setup_test.py puts the two Google Drive files
WEIGHTS = os.path.join(PRIMA_SRC, "test", "trained_models")
FULL_MODEL_CKPT = os.path.join(WEIGHTS, "primafullmodel107.pt")
VQVAE_CKPT = os.path.join(WEIGHTS, "tokenizer_model", "vqvae_model_step16799.pth")
N_HEADS = 52
STUDY_DESCRIPTION = "MRI BRAIN WO AND W CONTRAST"  # configs/pipeline_config.yaml of the repo
SERIES = {"ixi": ["T1"], "molab": ["T1 GD"], "ms": ["T1", "FLAIR"], "ct_rate": ["CT"]}
MS_FILES = {"T1": "T1W.nii.gz", "FLAIR": "FLAIR.nii.gz"}

# Diagnosis heads (names of Prima_training_and_evaluation/configs/jsons/prospective_eval.json) -> our groups
GROUP_HEADS = {
    "tumor": ["tumor_adult_brain_metastasis", "tumor_adult_glioma", "tumor_adult_high_grade_glioma",
              "tumor_adult_low_grade_glioma", "tumor_adult_meningioma"],
    "multiple sclerosis": ["inflammatory_multiple_sclerosis"],
}
NORMAL_THRESHOLD = 0.5  # on p: every relevant head below Prima's own cut-off


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def decide(head_values):
    """head name -> logit - thresh  ->  (values for the record, group probabilities)."""
    p = {name: sigmoid(v) for name, v in head_values.items()}
    groups = {g: max(p[h] for h in heads) for g, heads in GROUP_HEADS.items()}
    best = max(groups, key=groups.get)
    if groups[best] < NORMAL_THRESHOLD:
        return {"diagnosis_name": "normal", "diagnosis_confidence": 1.0 - groups[best]}, groups
    return {"diagnosis_name": best, "diagnosis_confidence": groups[best]}, groups


def series_files(scan):
    """(series description, NIfTI path) of the study: MS = T1 and FLAIR of the patient folder, else the scan."""
    if scan.dataset == "ms":
        folder = os.path.dirname(scan.files[0])
        return [(name, os.path.join(folder, MS_FILES[name])) for name in SERIES["ms"]]
    return [(SERIES[scan.dataset][0], scan.files[0])]


def write_study(scan, volumes, study_dir):
    """One DICOM series per NIfTI under study_dir/<n>_<name>/. A blank control (all-zero volume) gives zero series
    of each file's own shape."""
    blank = not np.any(volumes[0])
    written = []
    for n, (name, path) in enumerate(series_files(scan), 1):
        img = nib.load(path)
        if path == scan.files[0]:
            data = native_from_canonical(volumes[0], img)
        else:  # the second MS series (not among the runner's volumes)
            data = np.asarray(img.dataobj, dtype=np.float32)
            data = data[..., 0] if data.ndim == 4 else data
            if blank:
                data = np.zeros_like(data)
        folder = os.path.join(study_dir, f"{n}_{name.replace(' ', '_')}")
        write_dicom_series(data, img.affine, folder, name, n, scan.image_path, STUDY_DESCRIPTION,
                           modality="CT" if scan.dataset == "ct_rate" else "MR")
        written.append({"series": name, "file": path, "shape": list(data.shape)})
    return written


def _import_pipeline():
    """Prima's end-to-end_inference_pipeline/pipeline.py (a folder name with dashes, so loaded by path)."""
    # Prima imports its own top-level `tools` package; MANU has one too (/manu is on PYTHONPATH), so Prima's
    # source goes first and an already imported MANU `tools` is dropped.
    if PRIMA_SRC in sys.path:
        sys.path.remove(PRIMA_SRC)
    sys.path.insert(0, PRIMA_SRC)
    for name in [m for m in sys.modules if m == "tools" or m.startswith("tools.")]:
        if not os.path.abspath(getattr(sys.modules[name], "__file__", "") or "").startswith(PRIMA_SRC):
            del sys.modules[name]
    path = os.path.join(PRIMA_SRC, "end-to-end_inference_pipeline", "pipeline.py")
    spec = importlib.util.spec_from_file_location("prima_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Pipeline


class PrimaAdapter:
    name = "prima"
    model_id = "MLNeurosurg/prima@604f1fc (primafullmodel107.pt, vqvae_model_step16799.pth)"
    mode = "clip_or_heads"

    def _config(self, study_dir, output_dir):
        return {"study_dir": study_dir, "output_dir": output_dir, "study_description": STUDY_DESCRIPTION,
                "tokenizer_model_config": self.tokenizer_config,
                "prima_model_config": {"full_model_ckpt": FULL_MODEL_CKPT}}

    def load(self):
        for path in (FULL_MODEL_CKPT, VQVAE_CKPT):
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Prima weights missing: {path} (run Prima's "
                                        "tools/download_models_and_setup_test.py on the login node, NOTES.md)")
        self.Pipeline = _import_pipeline()
        with open(os.path.join(PRIMA_SRC, "configs", "sample_tokenizer_config.json"), encoding="utf-8") as f:
            self.tokenizer_config = json.load(f)
        self.tokenizer_config["vqvae_config"]["ckpt_path"] = VQVAE_CKPT
        self.workdir = tempfile.mkdtemp(prefix="prima_")
        first = self.Pipeline(self._config(self.workdir, os.path.join(self.workdir, "out")))
        self.tokenizer_model = first.load_tokenizer_model()
        self.prima_model = first.load_full_prima_model()
        self.thresholds = {n: float(h[0].thresh) for n, h in self.prima_model.diagnosisheads.items()}
        heads = list(self.thresholds)
        missing = [h for hs in GROUP_HEADS.values() for h in hs if h not in heads]
        if len(heads) != N_HEADS or missing:
            raise RuntimeError(f"expected {N_HEADS} diagnosis heads incl. {missing}, the checkpoint has {heads}")
        print(f"Prima diagnosis heads ({len(heads)}): {heads}", flush=True)
        print(f"Groups: {GROUP_HEADS}; normal when every group head has p < {NORMAL_THRESHOLD}", flush=True)

    def _infer(self, study_dir, output_dir):
        """The steps of pipeline.py's __main__, with the models loaded once in load()."""
        pipeline = self.Pipeline(self._config(study_dir, output_dir))
        pipeline.tokenizer_model, pipeline.prima_model = self.tokenizer_model, self.prima_model
        mri_study, series_names = pipeline.load_mri_study()
        series_embeddings, series_names_for_embeddings, all_ser_emb_meta = pipeline.run_tokenizer_model(
            mri_study, series_names=series_names)
        if series_names_for_embeddings is not None:
            series_names = series_names_for_embeddings
        if not series_embeddings:
            raise RuntimeError("No series could be tokenized; pipeline cannot continue. Check logs for skipped series.")
        predictions = pipeline.run_prima_model(series_embeddings=series_embeddings, series_names=series_names,
                                               all_ser_emb_meta=all_ser_emb_meta)
        tokens = {name: int(len(e)) for name, e in zip(series_names, series_embeddings)}
        return predictions, tokens

    def predict(self, scan, volumes):
        with tempfile.TemporaryDirectory(dir=self.workdir) as tmp:
            study_dir = os.path.join(tmp, "study")
            written = write_study(scan, volumes, study_dir)
            predictions, tokens = self._infer(study_dir, os.path.join(tmp, "out"))

        def value(x):
            return float(np.asarray(x.detach().float().cpu() if hasattr(x, "detach") else x).reshape(-1)[0])

        head_values = {n: value(v) for n, v in predictions["diagnosis"].items()}
        values, groups = decide(head_values)
        details = {
            "series": written, "series_tokens": tokens, "study_description": STUDY_DESCRIPTION,
            "diagnosis_logit_minus_thresh": head_values, "diagnosis_thresh": self.thresholds,
            "diagnosis_p": {n: sigmoid(v) for n, v in head_values.items()},
            "group_probabilities": groups, "normal_threshold": NORMAL_THRESHOLD,
            "referral": {n: value(v) for n, v in predictions.get("referral", {}).items()},
            "priority": {n: value(v) for n, v in predictions.get("priority", {}).items()},
        }
        return values, details


if __name__ == "__main__":
    runner.main(PrimaAdapter())
