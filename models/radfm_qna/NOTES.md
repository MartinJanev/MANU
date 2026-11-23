# RadFM through the shared pipeline (radfm-qna)

A last attempt to run RadFM the way the other models run: `common.runner` (splits, resume, blank controls,
`results/radfm-qna/<split>/`), the canonical volume, and the shared per-field questions of `common/qna.py`
in the M3D close-ended format that med-2e3 also uses. It replaces the earlier runner `models/radfm/run.py`
(own questions, likelihood choice calibrated on a black volume, own loop, run through `jobs/job_radfm.sh`),
removed after 45b8448: `git show 45b8448:models/radfm/run.py`. Its one lasting finding is under Pitfalls.

## Setup

Nothing new is downloaded. Everything comes from the full RadFM run already on the cluster:
- Image `containers/radfm.sif` (NGC PyTorch 23.05, Python 3.10, torch 2.0, transformers 4.28.1, built from
  `containers/radfm.Dockerfile`). `requirements.txt` repeats the Dockerfile's pins (a test keeps them equal),
  so `setup_env.sh` installs none of them; it only adds `jobs/requirements_common.txt` (nibabel 5.3.3 and its
  small dependencies) to `pyuser/radfm_qna`.
- Model code, `Language_files/` and `pytorch_model.bin` from `third_party/RadFM/Quick_demo` (mounted at
  `/third_party/RadFM/Quick_demo`).
- Med-KEBERT, which RadFM's embedding layer loads at construction, from `hf_cache` (offline, as in the full run).

Copied from the earlier runner (which follows `RadFM/Quick_demo/test.py`): building LLaMA from
`Language_files/config.json` and loading every weight from `pytorch_model.bin` (`strict=False`, only
`embedding_layer.bert_model.*` may be missing); the tokenizer's image tokens, added inside the loop as in
`test.py`; pad/bos/eos = 0/1/2 (the config's eos 1 is wrong); float32 on one GPU (about 56 GB);
`embedding_layer.flag = "Text"`; the prompt `<image><image0>…<image31></image>` + question, image first;
`lang_model.generate(inputs_embeds=…)`; the answer-likelihood forward pass (`answer_logprob`).

Input: `load_canonical` → `to01` → `resize(·, (64, 256, 256))` → `(H, W, D) = (256, 256, 64)` → grey repeated
to 3 channels → `[1, 1, 3, 256, 256, 64]`. The in-plane layout is the one `load_volume` of the earlier runner
gave RadFM (kept in `common.volume.load_canonical`) (rows front → back, columns patient right → left, depth feet → head); depth 64 is the paper's
maximum for 3D training (the November run used 160). MS gets T1 only, like every other model (the full run
also fed FLAIR); `--ms-flair` gives the FLAIR variant under `radfm-qna-flair`.

Questions: `common.qna.interrogate(dataset, ask_fn, "A")`, i.e. "{question} Choices: A. … B. … C. … D. …",
each asked in original and reversed option order. Greedy, `max_new_tokens=16`. The value is what
`common.parse.parse_choice` finds in the generated text (a leading letter or exactly one option text), the same
as for every model.

## Deviations and best guesses

- Option probabilities: RadFM was trained to answer VQA in free text (PMC-VQA, VQA-RAD and SLAKE as open answers),
  not with letters, so letter probabilities would mean little. `ask_fn` returns instead the softmax over the four
  displayed option texts of their mean log-probability per token right after the question (the
  `answer_logprob` of the earlier runner, without its black-volume calibration). These only fill the
  confidence fields; they never pick the value.
- No calibration: the runner's blank-volume control now shows how much RadFM ignores the image, as for the
  other models, instead of being subtracted.
- Report: one free-text report per scan with RadFM's caption template ("Can you provide a caption consists of
  finding and impression for this medical image?"), greedy, `max_new_tokens=512`, saved as `details.report` only
  (not scored). If it makes the job too slow, lower `REPORT_MAX_NEW_TOKENS`.
- The adapter is named `radfm-qna`, not `radfm`: with `--split full` the runner would otherwise resume into
  `results/radfm/full/`, where the older run's files have the same names, and `evaluate.py` would stop showing
  the older run for dev/eval.

## Before running

- `bash jobs/setup_env.sh radfm_qna` on the login node: no pull (the image exists), no model download; it must end
  without "does not import". If the import check fails on `scipy` or `PIL` (`common/volume.py`), add
  `scipy==1.10.1` / `Pillow==9.5.0` to `requirements.txt` and rerun setup.
- `ls third_party/RadFM/Quick_demo` shows `Model/`, `Language_files/`, `pytorch_model.bin`.
- Then `sbatch -J radfm_qna jobs/run_model.sh radfm_qna --split dev --controls`.
- Expected per scan: about 10-14 asked questions × 2 orders, each one short generation plus four scoring passes,
  and one long report; budget roughly 0.5-1 min per scan plus 3-6 min loading.

## Dev run

- Job 2896 (2026-11-23), the authors' Quick_demo example through this adapter: "yes" for the image and "yes" for a blank image of the same size. Job 2900 (dev): 47 of 140 answers empty, reports "Finding: nan" and, for a brain MRI, "a focal lesion ... in the left 2nd metatarsal bone". Fails the validation gate (PROTOCOL.md); run on the split paper and reported as not validated.


## Eval run

- 30-patient eval split (2026-11-21, 230 records, 100% valid): `diagnosis_name` is "normal" for every scan of
  IXI, MOLAB and MS (IXI 100% only because "normal" is its truth; MOLAB 0%, MS 0%; pooled balanced accuracy
  33.3% = no signal). Blank agreement 79.2%, blank → normal on all three datasets. CT-RATE: modality CT 0%,
  plane axial 0%. Verdict: RadFM gives no usable answer on these brain MRI datasets through this pipeline; the
  adapter stays for other datasets.

## Pitfalls

- Probe run of the earlier runner: RadFM answered "multiple sclerosis" for black and noise volumes as well as for
  healthy IXI scans, so much of each answer is a prior that ignores the image. That is why the earlier runner
  divided every answer's likelihood by the one for a black volume (contextual calibration).
- The full run answered "multiple sclerosis" / "stroke" for almost every scan and "coronal" for every axial
  volume: compare blank agreement and the per-field `agree` (original vs reversed order) in the dev report.
- `generate` with `inputs_embeds` returns only the new tokens in transformers 4.28.1, so the decoded text is the
  answer alone.
