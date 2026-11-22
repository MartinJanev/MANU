# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Evaluation of 3D medical multimodal LLMs on brain MRI: IXI (truth: normal), MOLAB (brain metastases, truth: tumor),
open_ms_data (truth: multiple sclerosis), plus CT-RATE chest CT as a control. Models run on a SLURM GPU cluster
inside Singularity images; the Mac is for code, tests and scoring. `README.md` holds the run commands and the
**Model status** table (update that table when results change).

## Commands (on the Mac)

There is no project venv and no system Python has the dependencies (numpy, nibabel, scipy, Pillow, pytest); run
everything through uv with throwaway dependencies (written out in full: the shell is zsh, which does not
word-split a command stored in a variable):

```bash
uv run -q --no-project --python 3.11 --with pytest,numpy,nibabel,scipy,pillow python -m pytest -q tests
uv run -q --no-project --python 3.11 --with pytest,numpy,nibabel,scipy,pillow python -m pytest -q \
    tests/adapters/test_hlip.py::test_preprocess_shape                       # one test
uv run -q --no-project --python 3.11 --with numpy,nibabel,scipy,pillow python -I evaluation/evaluate.py --split eval
uv run -q --no-project --python 3.11 --with numpy,nibabel,scipy,pillow python -I tools/reparse.py <model> dev|eval
DRY_RUN=1 bash jobs/run_model.sh <model_dir> --split eval --controls      # print the cluster command only
```

`evaluate.py --split eval` writes `results/report_eval.md`; `--selftest` checks its metrics. `reparse.py` re-scores
saved answers of `healthgpt-pro-27b` or `medgemma-1.5-4b` without the GPU. The suite needs no GPU or weights.

Scripts are run with `python3 -I` and add the repo root to `sys.path` themselves.

Cluster round trip (EduVPN + the user's SSH passphrase; Claude cannot reach the cluster, so give the user the
commands): `tools/sync_to_cluster.sh` → on the login node `bash jobs/setup_env.sh <model_dir>` once, then
`sbatch -J <model_dir> jobs/run_model.sh <model_dir> --split dev|eval --controls [--fresh] [--datasets ms --ms-flair]`
→ `tools/fetch_results.sh`. `results/` is not in git, and `results/archive/` is never fetched.

## Architecture

**Adapter + shared runner.** Each `models/<dir>/run.py` defines a small adapter (`name`, `model_id`, `mode`,
`load()`, `predict(scan, volumes) -> (raw values, details)`) and calls `common.runner.main(adapter)`. The runner
owns everything else: picking scans from `splits.json`, resume (skip recorded scans, retry error rows), error
isolation (`PredictionError(message, details)` keeps the raw output), blank-volume controls (`--controls`),
`run_info.json`, `--fresh` (archive the old run first), and `--ms-flair` (FLAIR for MS, results under
`<name>-flair`). The adapter's `name` (e.g. `healthgpt-pro-27b`) names `results/<name>/<split>/`; the folder name
(`healthgptpro27b`) is what `sbatch`/`setup_env.sh` take.

**One record format.** `common/schema.build_record` writes the colleagues' JSONL format (`PARSED_KEYS` per dataset)
and `normalize_values` maps raw answers onto `VOCAB`. Each dataset gets `<name>_<tag>_{processed,details,controls}.jsonl`
with tags `healthy`, `brain_metastases`, `ms`, `ct_rate`. Metadata and truth come from the colleagues' files in
`reference/` (`common.datasets.metadata_for`); CT-RATE truth comes from its labels CSV.

**Input path.** `common.volume.load_canonical` (one fixed orientation, square pixels, square frame) → `to01` (percentile scaling for MRI, a CT
window for CT-RATE) → whatever the model needs (`pick_slices` + `to_rgb_images` for 2D VLMs, `resize` for 3D).

**Modes** (`runner.MODES`): `full_json` (prompt from `common/prompts/full_json.txt`, parsed by
`common.parse.parse_json_answer`), `qna_per_field` (the `common/qna.py` bank, every question asked in original and
reversed option order, `parse_choice`), `findings_then_extract` (free report → `extract_from_report`),
`clip_or_heads`. `--task qna|prior` runs a full-JSON model in the other prompting mode (its `asker`) or with the
diagnosis stated (`schema.PRIOR`, MOLAB and MS only); results under `<name>-qna` / `<name>-prior`.
`bash jobs/run_all.sh dev|main|second` submits a whole stage of the paper's runs.

**Standard-library-only modules:** `common/datasets.py`, `schema.py`, `parse.py`, `qna.py` must import without
numpy, nibabel or torch (they run on the login node and in light tests).

**Cluster jobs.** `jobs/run_model.sh` sources `models/<dir>/env.sh` (`SIF`, `SIF_SOURCE`, `HF_REPOS`, `GIT_REPOS`,
optional `EXTRA_BINDS`) and runs offline in the image with binds `/manu`, `/data`, `/results`, `/reference`,
`/hf_cache`, `/pyuser` (= `pyuser/<dir>`), `/third_party`. `setup_env.sh` installs `jobs/requirements_common.txt`
plus the model's pinned `requirements.txt` into `pyuser/<dir>`.

**Evaluation.** `evaluation/evaluate.py --split` reads `results/<model>/<split>/` and `reference/` (filtered to the
split's patients) and writes the per-model table, confusion matrices, collapse tables (distinct answers per field),
blank agreement, the site-shortcut baseline (`results/site_baseline.md` from `tools/site_baseline.py`) and the
CT-RATE table.

## Adding a model

`tests/test_jobs.py::test_model_dirs_complete` requires `__init__.py`, `run.py`, `requirements.txt` (every line
pinned with `==` or `@ git+`), `env.sh` and `NOTES.md`. NOTES.md follows the sections Setup / Dev run / Eval run /
Pitfalls and records the sources read (model card, repo commit) and every deviation from them. Each adapter has
`tests/adapters/test_<dir>.py`, which monkeypatches the generate/processor calls, so tests never load weights.

## Things that matter when reading results

- The paper's runs are split `paper` (one scan per patient; `tools/make_split.py --paper`). Split `eval` is the
  30-patient pilot of 21 Nov 2026: per-dataset prompts (they named the diagnosis), stretched images, MSLesSeg's
  `ms_type` labels. Never mix or compare the two, and never rerun `eval` with today's code into the same folder.
- One prompt for every dataset, no slice count and no dataset-specific wording: each diagnosis is its own dataset,
  so anything chosen per dataset is a label.
- Decoding is greedy in every adapter: rerunning the same setup reproduces the outputs. Change the input or prompt
  instead of repeating a run.
- After a parser fix, re-score with `tools/reparse.py` instead of rerunning on the GPU. `fetch_results.sh` uses
  `rsync -u`, so locally re-scored files are not overwritten by older cluster copies.
- Thinking models (healthgpt-pro-27b, qoq-med, fleming): an answer counts as cut off inside `<think>` only when
  generation used all `max_new_tokens`, because `skip_special_tokens` can strip the `</think>` tag.
- Confounds: acquisition features alone identify the dataset 91% of the time (so diagnosis accuracy is partly
  dataset recognition), and MS is the only dataset with FLAIR. Blank-volume agreement shows which models ignore
  the image.
- `data_tools/` and some older comments are in Macedonian. The first RadFM runner was removed
  (`git show 45b8448:models/radfm/run.py`); `models/radfm_qna` replaces it.
