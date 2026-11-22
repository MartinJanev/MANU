"""jobs/run_all.sh: one command per block."""
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
FOLDERS = sorted(d.name for d in (REPO / "models").iterdir()
                 if (d / "run.py").exists() and d.name != "prima")


def plan(stage):
    p = subprocess.run(["bash", "jobs/run_all.sh", stage], cwd=REPO, capture_output=True, text=True,
                       env={"DRY_RUN": "1", "PATH": "/usr/bin:/bin"})
    assert p.returncode == 0, p.stderr
    return [line for line in p.stdout.splitlines() if line.startswith("sbatch")]


def model_of(job):
    return job.split("jobs/run_model.sh ")[1].split()[0]


def test_dev_covers_every_model_and_every_task():
    jobs = plan("dev")
    assert sorted({model_of(j) for j in jobs}) == FOLDERS and len(FOLDERS) == 12
    assert len(jobs) == 12 + 6 + 6 + 1  # native, qna, prior, RadFM's own example
    assert all("--split dev" in j and "--fresh" in j for j in jobs if "--demo" not in j)
    assert sum("--demo" in j for j in jobs) == 1


def test_main_runs_the_original_datasets_first_and_never_wipes():
    jobs = plan("main")
    native, flair = jobs[:12], jobs[12:]
    assert sorted(map(model_of, native)) == sorted(map(model_of, flair)) == FOLDERS
    assert all("--split paper --controls" in j and "--fresh" not in j for j in jobs)
    assert all("--datasets ms --ms-flair" in j for j in flair) and not any("--ms-flair" in j for j in native)


def test_second_is_qna_and_prior_for_the_six_full_prompt_models():
    jobs = plan("second")
    qna = [j for j in jobs if "--task qna" in j]
    prior = [j for j in jobs if "--task prior" in j]
    assert (len(qna), len(prior), len(jobs)) == (6, 6, 12)
    assert all("--datasets molab,ms" in j for j in prior) and all("--fresh" not in j for j in jobs)


def test_job_names_are_unique():
    for stage in ("dev", "main", "second"):
        names = [j.split(" -J ")[1].split()[0] for j in plan(stage)]
        assert len(names) == len(set(names))


def test_an_unknown_stage_fails():
    p = subprocess.run(["bash", "jobs/run_all.sh", "nope"], cwd=REPO, capture_output=True, text=True,
                       env={"DRY_RUN": "1", "PATH": "/usr/bin:/bin"})
    assert p.returncode != 0 and "dev|main|second" in p.stderr


@pytest.mark.parametrize("log,ok", [(None, False), ("SETUP FINISHED: FAILED\n", False), ("downloading\n", False),
                                    ("SETUP FINISHED: ok\n", True)])
def test_submits_only_after_the_downloads_finished(tmp_path, log, ok):
    (tmp_path / "jobs").mkdir()
    (tmp_path / "bin").mkdir()
    shutil.copy(REPO / "jobs/run_all.sh", tmp_path / "jobs")
    fake = tmp_path / "bin/sbatch"
    fake.write_text('#!/bin/bash\necho "Submitted batch job 1"\n')
    fake.chmod(0o755)
    if log is not None:
        path = tmp_path / "results/slurm/setup_paper.log"
        path.parent.mkdir(parents=True)
        path.write_text(log)
    p = subprocess.run(["bash", "jobs/run_all.sh", "dev"], cwd=tmp_path, capture_output=True, text=True,
                       env={"PATH": f"{tmp_path / 'bin'}:/usr/bin:/bin"})
    assert (p.returncode == 0) is ok
    assert ("Submitted" in p.stdout) is ok
