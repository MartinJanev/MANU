# MANU: can 3D medical AI models read a brain MRI?

This project tests whether today's medical multimodal language models (models that take a scan and answer in
text) can tell normal brain MRI from brain with a disease. Each model is shown a scan and asked for a structured
report: imaging modality, sequence, whether it is normal, and the diagnosis. Its answers are then scored against
the known truth.

| dataset | what it contains | correct answer |
|---|---|---|
| [IXI](https://brain-development.org/ixi-dataset/) | healthy volunteers | normal |
| MOLAB | patients with brain metastases | tumor |
| [open_ms_data](https://github.com/muschellij2/open_ms_data) | patients with multiple sclerosis | multiple sclerosis |
| [CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE) | chest CT (a control outside the brain) | from its labels file |

We also give every model a **blank volume** with the same question. If a model gives the same answer to the
blank as to the real scan, it is not looking at the image.

## Results so far

The table below is from the **pilot** (30 patients per dataset). The pilot had flaws that the final run fixes:
the prompt differed per dataset (so it hinted at the diagnosis), images were stretched, and the MS subtype labels
were wrong. **Do not cite these numbers.** The paper's results come from the `paper` split (see
[Splits](#splits)) and the protocol in [`evaluation/PROTOCOL.md`](evaluation/PROTOCOL.md).

Seven additional checkpoints are available in the separate
[cross-modality development extension](extensions/cross_modality/README.md).
They are not part of the frozen paper runs.

How to read it:
- **bal.acc** (balanced accuracy) across IXI, MOLAB and MS. Guessing gives 33.3%.
- **blank agreement**: how often the answer to the blank volume matches the answer to the real scan. A high
  number means the image is ignored.
- Scans are T1-weighted unless marked FLAIR.
- "Not usable here" means the model gave no image-dependent answer on these brain MRI scans through this
  pipeline. The model may still work on other data. Each model's `models/<model>/NOTES.md` has the details.

### Model status

| model | status here | evidence |
|---|---|---|
| hlip | works (IXI/MOLAB) | pooled bal.acc 56.0%, MOLAB 96.7% per patient; MS T1 27.6% is its default answer (blank → MS), FLAIR 6.9% |
| healthgpt-pro-27b | works | 51.9%, IXI 93.3%, MOLAB 60.0%; MS T1 0%, **FLAIR 82.8%** (blank → tumor) |
| qwen3-vl-8b | partial | 48.0%, IXI 100%, MOLAB 33.3%; MS T1 3.4%, FLAIR 27.6% |
| medgemma-1.5-4b | partial | 44.9%, blank agreement 87.5% on T1; MS **FLAIR 96.6%** (blank → normal) |
| lingshu-7b | ignores the image | 47.8% from answering "normal"; blank agreement 85.4% |
| qoq-med-vl-7b | ignores the image | 38.4%: "tumor/glioma" for almost every scan, MS 0% on T1 and FLAIR |
| med-2e3 | not usable here | 23.1%; says CT for every MRI scan, blank → stroke (a setup problem is not ruled out) |
| fleming-vl-8b | not usable here | 13.5%, 69.6% valid answers, mostly null or "normal" |
| braingemma3d | not usable here | 2.3%: "other abnormalities" for almost every scan (a setup problem is not ruled out) |
| radfm-qna | not usable here | "normal" for every scan (33.3%), blank agreement 79.2% |
| m3d-lamed (Phi-3-4B, Llama-2-7B) | not run in the pilot | added for the `paper` split |
| prima | not run | adapter ready; trained on studies with ≥ 2 series, IXI and MOLAB have one |

Two things limit what any score can mean:
- **FLAIR.** MS is the only dataset with FLAIR scans (IXI has none, MOLAB is T1 with contrast). A model that says
  "MS" whenever it sees FLAIR scores the same as one that actually finds the lesions.
- **Dataset recognition.** Acquisition details alone (voxel spacing, volume shape, intensities) identify which
  dataset a scan comes from most of the time, so a model can score partly by recognising the scanner rather than
  the disease. `tools/site_baseline.py` measures this.

## How it works

Every model gets a small **adapter** in `models/<model>/run.py` that loads the model and turns one scan into an
answer. Everything else is shared, in `common/`:

1. load the scan in one fixed orientation with square pixels (`common/volume.py`);
2. ask the same question for every dataset, so the wording gives nothing away (`common/prompts/`, `common/qna.py`);
3. parse the free-text answer into fixed fields (`common/parse.py`, `common/schema.py`);
4. save one record per scan; a crashed or interrupted job resumes where it stopped (`common/runner.py`).

`evaluation/evaluate.py` then turns the saved answers into the per-model tables, confusion matrices and
blank-volume agreement. `evaluation/stats.py` writes the tables for the paper.

```
common/        shared pipeline: runner, datasets, volume loading, prompts, parsing, Q&A bank
models/        one folder per model: run.py (adapter), env.sh, requirements.txt, NOTES.md
jobs/          SLURM job, environment setup, run_all.sh for a whole stage of runs
evaluation/    evaluate.py, stats.py, PROTOCOL.md
tools/         cluster sync, re-scoring, split drawing, site baseline
data_tools/    dataset download and checks (comments in Macedonian)
reference/     earlier results from colleagues, in the same record format (not in git)
containers/    container definitions (the built images live on the cluster only)
tests/         runs without a GPU or model weights
results/       outputs, one folder per model and split (not in git)
```

## Running it

The models run on a SLURM GPU cluster inside Singularity containers. A laptop is enough for the code, the tests
and the scoring. You need [uv](https://docs.astral.sh/uv/); nothing else has to be installed.

**Tests** (no GPU, no weights):

```
uv run -q --no-project --python 3.11 --with pytest,numpy,nibabel,scipy,pillow python -m pytest -q tests
```

**On the cluster login node**, once per model (downloads the weights and installs its packages):

```
bash jobs/setup_env.sh <model>
```

Then a short test on the `dev` patients, and the real run after it:

```
sbatch -J <model> jobs/run_model.sh <model> --split dev --controls
sbatch -J <model> jobs/run_model.sh <model> --split paper --controls
```

`<model>` is the folder name under `models/` (e.g. `healthgptpro27b`). `--controls` adds the blank-volume runs.
`bash jobs/run_all.sh dev|main|second` submits every model at once; `DRY_RUN=1` in front prints the commands
without submitting.

**Scoring**, back on the laptop:

```
uv run -q --no-project --python 3.11 --with numpy,nibabel,scipy,pillow python -I evaluation/evaluate.py --split paper
```

### Reruns and resuming

Submitting the same job again **resumes**: scans that already have an answer are skipped and failed ones are
retried. So a job killed half-way should be resubmitted as it was.

To start over (after changing code, prompts or packages), add `--fresh`. The old results are moved to
`results/archive/<model>/<split>_<start time>/` first, so nothing is lost. Decoding is deterministic, so
rerunning an unchanged setup gives the same answers; there is no point repeating a run without changing something.

### Re-scoring without the GPU

After a fix to the answer parser, re-score the saved answers instead of running the model again (old files are
copied to `results/archive/` first):

```
uv run -q --no-project --python 3.11 --with numpy,nibabel,scipy,pillow python -I tools/reparse.py <model> <split>
```

This works for `healthgpt-pro-27b` and `medgemma-1.5-4b`.

### Moving files to and from the cluster

With the VPN on:

```
bash tools/sync_to_cluster.sh     # code to the cluster; never deletes there (RSYNC_DRY=1 for a dry run)
bash tools/fetch_results.sh       # results back to ./results/, keeping local files that are newer
```

## Splits

`splits.json` is the only place that says which patients are used:

| split | IXI | MOLAB | MS | CT-RATE | purpose |
|---|---|---|---|---|---|
| `dev` | 1 | 1 | 1 | 1 | quick check that a model runs |
| `eval` | 30 | 30 | 29 | 30 | the pilot above; kept only for reference |
| `paper` | 120 | 75 | 30 | 60 | the paper's comparison, one scan per patient |

Results from `eval` and `paper` were produced with different prompts and preprocessing, so they must never be
compared with each other.

## Adding a model

Create `models/<name>/` with `__init__.py`, `run.py`, `requirements.txt` (every package pinned), `env.sh` and
`NOTES.md` (sections Setup / Dev run / Eval run / Pitfalls, with the sources you followed and anything you did
differently). Add `tests/adapters/test_<name>.py`, which fakes the model calls so the test needs no weights.
`tests/test_jobs.py` checks that nothing is missing.
