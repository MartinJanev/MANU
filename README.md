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
tools/         layout.py (layout check)
reference/     colleagues' *_processed.jsonl
results/       <model>/<split>/..., slurm/, archive/  (not in git)
archive/code/  old RadFM runners
tests/
cluster only (not in git): data/ hf_cache/ pyuser/<model>/ third_party/ containers/*.sif
```

Check a checkout: `python3 tools/layout.py [--cluster] [ROOT]`.

## Splits

`splits.json` is the only place that defines which patients are used.
`dev` is the small smoke-test set, `eval` the full comparison set.

## Running a model

(dev, then eval; commands are filled in with the runner, Task 7)

## Cluster sync

(sync and fetch commands are filled in with the sync scripts, Task 8)
