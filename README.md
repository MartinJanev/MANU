# MANU: 3D medical MLLM evaluation

## Layout

```
README.md  splits.json  .gitignore
common/        shared pipeline (grows in later tasks)
models/        radfm/run.py, one folder per model
jobs/          job_radfm.sh (SLURM)
containers/    radfm.Dockerfile (+ *.sif on the cluster only)
data_tools/    prezemi_podatoci.sh  proveri_podatoci.py
evaluation/    evaluate.py  check_schema.py
tools/         layout.py, migrate_cluster.sh, sync_to_cluster.sh, fetch_results.sh,
               make_split.py, reparse.py, site_baseline.py
reference/     colleagues' *_processed.jsonl
results/       <model>/<split>/..., slurm/, archive/  (not in git)
tests/
cluster only (not in git): data/ hf_cache/ pyuser/<model>/ third_party/ containers/*.sif
```

Check a checkout: `python3 tools/layout.py [--cluster] [ROOT]`.

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

Once, on the cluster, to move its own files into the new layout (the first run only prints the plan):

```
bash tools/migrate_cluster.sh
bash tools/migrate_cluster.sh --apply
```

Fetch results back to `./results/` (without `archive/`):

```
bash tools/fetch_results.sh
```
