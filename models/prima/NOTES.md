# Prima

## Setup

Sources read (2026-11-20), https://github.com/MLNeurosurg/prima cloned at 604f1fc4dd7059c278d4d34bbbcaab3e404fbcd4
(MIT):
- README: weights `primafullmodel107.pt` (full model; the priority head was corrected on 2026-02-19, so a copy
  downloaded before then must be replaced) and `vqvae_model_step16799.pth` (tokenizer), both Google Drive links;
  Prima's UM-220K studies were filtered to "a minimum of 2 series per study".
- `end-to-end_inference_pipeline/pipeline.py`: `Pipeline(config)`, `load_mri_study` (`tools/DicomUtils.py`: one
  sub-folder per series, `ImageSeriesReader`, file names natsorted, in-plane resampled to 256x256, `DICOMOrient`
  to LPS, series name from SeriesDescription / ProtocolName), `run_tokenizer_model`, `run_prima_model` (fp16
  autocast, `make_no_flashattn()`); its `__main__` is the order of calls used here.
- `tools/models.py` `FullMRIModel.forward`: `diagnosis[name] = head(clip_embed)[:, idx] - head.thresh`;
  `Prima_training_and_evaluation/classification_altogether.py` + `utils.getbestthresh`: heads are trained with
  `BCEWithLogitsLoss`, `thresh` is the best-accuracy cut-off on the logit (grid -5.0 ... 2.0), so value >= 0 is
  Prima's own "present" (`eval_prospective_classification.evalacc` with offset 0).
- `Prima_training_and_evaluation/configs/jsons/prospective_eval.json`: the 52 diagnosis names (the adapter checks
  the checkpoint has 52 heads including the mapped ones, and prints them in the job log):
  cyst_arachnoid_cyst, cyst_colloid_cyst, developmental_dandy_walker_malformation,
  developmental_dysgenesis_corpus_callosum, developmental_heterotopia, infectious_brain_abscess,
  infectious_viral_encephalitis, inflammatory_multiple_sclerosis, inflammatory_neurosarcoid,
  sellar_craniopharyngioma, sellar_pituitary_adenoma, sellar_rathkes_cleft_cyst, spine_syrinx,
  structural_brain_herniation, structural_cephaloceles, structural_cerebral_atrophy, structural_chiari_malformation,
  structural_edema, structural_encephalomalacia, structural_mass_effect, structural_midline_shift,
  surgical_catheter, surgical_craniotomy, surgical_resection_cavity, trauma_brain_contusion,
  trauma_diffuse_axonal_injury, trauma_subdural_hematoma, tumor_adult_brain_metastasis, tumor_adult_glioma,
  tumor_adult_high_grade_glioma, tumor_adult_low_grade_glioma, tumor_adult_lymphoma, tumor_adult_meningioma,
  tumor_adult_pineal_tumor, tumor_adult_schwannoma, tumor_pediatric_brainstem_glioma, tumor_pediatric_ependymoma,
  tumor_pediatric_germ_cell_tumor, tumor_pediatric_medulloblastoma, tumor_pediatric_pilocytic_astrocytoma,
  vascular_hemorrhagic_intracranial_hemorrhage, vascular_hemorrhagic_intraventricular_hemorrhage,
  vascular_hemorrhagic_subarachnoid_hemorrahage, vascular_ischemic_lacunar_stroke,
  vascular_ischemic_large_vessel_stroke, vascular_ischemic_moyamoya, vascular_ischemic_small_vessel_disease,
  vascular_malformation_arteriovenous_malformation, vascular_malformation_cavernoma,
  vascular_malformation_cerebral_aneurysm, ventricular_intracranial_hypotension, ventricular_ventriculomegaly.
- `requirements.txt` of the repo (pins copied for the packages on the inference import path).

Used unchanged: Prima's pipeline code, called step by step as its `__main__`. The VQ-VAE and the full model are
loaded once with the pipeline's own `load_tokenizer_model` / `load_full_prima_model` and set on each per-scan
`Pipeline` (the pipeline frees only its own references after each study), so the weights are not re-read per scan.

Input: `common/dicom.py` writes each NIfTI as a DICOM series (slices along the file's third voxel axis, geometry
from the affine, uint16 + rescale; checked against SimpleITK on straight, sagittal LPS, oblique and left-handed
affines, and read with Prima's `DicomUtils.load_mri_study`). MS: T1W and FLAIR of the patient folder (two series);
IXI and MOLAB: one series; CT-RATE (control): one CT series. Series descriptions T1 / T1 GD / FLAIR / CT; study
description "MRI BRAIN WO AND W CONTRAST" (the repo's `configs/pipeline_config.yaml`).

Mapping (no "normal" head): p = sigmoid(logit - thresh) per head, so p = 0.5 is Prima's cut-off; a group's
probability = the largest p of its heads. tumor = brain_metastasis, glioma, high/low grade glioma, meningioma;
multiple sclerosis = inflammatory_multiple_sclerosis. If every one of these heads has p < 0.5 (NORMAL_THRESHOLD,
logged in the job and in every details row) the diagnosis is normal with confidence 1 - max p; otherwise the most
probable group with its probability. Every other field is null. Details: all 52 values, thresholds and p, referral
and priority outputs, series written and tokens per series.

Pins (`requirements.txt`): Prima's own pins for monai, monai-generative, SimpleITK, pydicom, natsort, transformers
(4.49.0, below the 4.53 torch_compat case), positional-encodings, perceiver-pytorch, einops, pathos, tqdm, pyyaml,
psutil. Left out: torch (the image's 2.6), flash-attn (optional in `model_parts.py`; the pipeline calls
`make_no_flashattn()`), nibabel (common pins 5.3.3, Prima 5.3.2), packages not imported at inference (wandb, lime,
pytorch-lightning, opencv, pandas, scikit-learn, patchify, gast). scipy and numpy from the image.

Pins added: gdown==5.2.0 (for Prima's own weight download script, which passes `fuzzy=True` to `gdown.download`; gdown 6 removed that argument).

Before running (login node): `bash jobs/setup_env.sh prima > setup_prima.log 2>&1` (packages, clone into
`third_party/prima`), then the weights with Prima's own script (README "Quick start", step 2), which writes
`test/trained_models/primafullmodel107.pt` and `test/trained_models/tokenizer_model/vqvae_model_step16799.pth`
inside the clone:
`singularity exec -B ./third_party:/third_party -B ./pyuser/prima:/pyuser containers/ngc2412.sif env
PYTHONUSERBASE=/pyuser python3 /third_party/prima/tools/download_models_and_setup_test.py`.
A copy of the full model downloaded before 2026-02-19 has the old priority head (README): delete it and rerun.

## Dev run

Smoke test (1 IXI, 1 MS): `sbatch -J prima jobs/run_model.sh prima --split dev --datasets ixi,ms --controls`

## Eval run

`sbatch -J prima jobs/run_model.sh prima --split eval --controls`

## Pitfalls

- IXI and MOLAB studies have one series; Prima was trained on studies with at least 2 (state this in the report).
- Prima's tokenizer takes the first axis that is not 256 (after the in-plane 256x256 resample) as the slice axis; a
  series with exactly 256 slices has none and fails inside Prima's code (error record). More than 5000 tokens
  (about 312 slices) and Prima skips the series. IXI (150 sagittal slices) and MS (144-160) are clear of both.
