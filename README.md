# MANU: 3D medical MLLM evaluation

## Layout

```
README.md  splits.json  .gitignore
common/        shared pipeline: runner, datasets, volume, prompts, parsing, QnA bank
models/        one folder per model: run.py (adapter), env.sh, requirements.txt, NOTES.md
jobs/          run_model.sh (SLURM), setup_env.sh, requirements_common.txt
containers/    radfm.Dockerfile (+ *.sif on the cluster only)
data_tools/    prezemi_podatoci.sh  prezemi_ct_rate.py  proveri_podatoci.py
evaluation/    evaluate.py  check_schema.py
tools/         sync_to_cluster.sh, fetch_results.sh, make_split.py, reparse.py, site_baseline.py
reference/     colleagues' *_processed.jsonl
results/       <model>/<split>/..., slurm/, archive/  (not in git)
tests/
cluster only (not in git): data/ hf_cache/ pyuser/<model>/ third_party/ containers/*.sif
```

## Model status

Eval split (30 patients per dataset), T1 for every dataset unless marked FLAIR; numbers from
`results/report_eval.md`. Pooled bal.acc 33.3% = no signal. Blank agreement = share of answers the blank-volume
control repeats (high = the image is ignored). Every adapter stays in `models/`: "not usable here" means no answer
that depends on the image on these brain MRI datasets through this pipeline, not that the model cannot work on
other data. Details per model in `models/<model>/NOTES.md`.

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
| prima | not run | adapter ready; trained on studies with ≥ 2 series, IXI and MOLAB have one |

FLAIR caveat: MS is the only dataset given as FLAIR (IXI has none, MOLAB is T1+Gd), so a model that answers MS
whenever it sees FLAIR scores the same as one that reads the lesions.

## Splits

`splits.json` is the only place that defines which patients are used.
`dev` is the small smoke-test set, `eval` the full comparison set.

## Running a model

On the cluster login node, once per model (downloads weights, builds `pyuser/<model>`):

```
bash jobs/setup_env.sh <model>
```

Then submit dev first, eval after:

```
sbatch -J <model> jobs/run_model.sh <model> --split dev --controls
sbatch -J <model> jobs/run_model.sh <model> --split eval --controls
```

Results go to `results/<model>/<split>/`. Score them locally with
`python3 -I evaluation/evaluate.py --split dev`.

A rerun **resumes**: scans that already have a result are skipped (the log says `N already done`, and
`Nothing to run` if the model was never loaded). To run again after changing code, prompts or packages, or to
repeat an experiment, add `--fresh`. It moves the old `results/<model>/<split>/` to
`results/archive/<model>/<split>_<start time>/` and runs every scan:

```
sbatch -J <model> jobs/run_model.sh <model> --split eval --controls --fresh
```

Resubmit a job that was killed half-way *without* `--fresh`, so it continues where it stopped.
`run_info.json` keeps the first run's start time and commit; each resume is listed under `resumed`.

## Re-scoring, a larger split, the site baseline

After a parser fix, re-score saved answers without the GPU (copies the old files to `results/archive/` first;
needs numpy, nibabel, scipy and Pillow, as the tests do):

```
python3 -I tools/reparse.py healthgpt-pro-27b|medgemma-1.5-4b dev|eval
```

A larger eval split (on the cluster login node, standard library only). It keeps the current eval patients, adds
random ones up to `--per-dataset` (dev excluded) and writes `results/splits_proposed.json`. Fetch it, copy it over
`splits.json` on the Mac, commit and sync; otherwise the next sync puts the old `splits.json` back on the cluster:

```
python3 tools/make_split.py --data data --per-dataset 30
```

Site-shortcut baseline: how well acquisition features alone (voxel spacing, shape, intensity percentiles) tell IXI,
MOLAB and MS apart. Near 100% means a model can score by recognising the dataset. CPU only, after the new split:

```
SINGULARITYENV_PYTHONUSERBASE=/pyuser singularity exec -B .:/manu -B ./data:/data -B ./results:/results \
    -B ./pyuser/hlip:/pyuser containers/ngc2412.sif python3 /manu/tools/site_baseline.py --data /data --results /results
```

## Cluster sync

EduVPN on. From the Mac (never deletes on the cluster; `RSYNC_DRY=1` for a dry run,
`CLUSTER` / `REMOTE` override the target):

```
bash tools/sync_to_cluster.sh
```

Fetch results back to `./results/` (without `archive/`; a local file newer than the cluster's, e.g. one
re-scored by `tools/reparse.py`, is kept):

```
bash tools/fetch_results.sh
```
