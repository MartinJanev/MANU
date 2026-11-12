import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
RUN_MODEL = REPO / "jobs" / "run_model.sh"
SETUP_ENV = REPO / "jobs" / "setup_env.sh"

ENV_SH = """SIF=containers/ngc2412.sif
SIF_SOURCE=docker://nvcr.io/nvidia/pytorch:24.12-py3
HF_REPOS="org/model-a org/model-b"
GIT_REPOS="https://github.com/example/tool.git@abc123"
"""


def run(cmd, env=None, cwd=None):
    """Run a command with DRY_RUN etc. on top of a minimal environment."""
    import os
    full_env = {"PATH": os.environ["PATH"], "HOME": os.environ.get("HOME", "/tmp")}
    full_env.update(env or {})
    return subprocess.run(cmd, env=full_env, cwd=cwd or REPO, capture_output=True, text=True)


@pytest.fixture
def testmodel():
    d = REPO / "models" / "_testmodel"
    try:
        d.mkdir(parents=True, exist_ok=True)
        (d / "env.sh").write_text(ENV_SH)
        (d / "run.py").write_text("print('hi')\n")
        (d / "requirements.txt").write_text("numpy==1.0\n")
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_run_model_dry_run(testmodel):
    p = run(["bash", "jobs/run_model.sh", "_testmodel", "--split", "dev"], env={"DRY_RUN": "1"})
    assert p.returncode == 0, p.stderr
    for s in ["singularity exec --nv", "./pyuser/_testmodel:/pyuser", "containers/ngc2412.sif",
              "/manu/models/_testmodel/run.py --split dev"]:
        assert s in p.stdout


def test_run_model_dry_run_from_other_directory(testmodel, tmp_path):
    p = run(["bash", str(RUN_MODEL), "_testmodel", "--split", "dev"],
            env={"DRY_RUN": "1"}, cwd=tmp_path)
    assert p.returncode == 0, p.stderr
    assert "/manu/models/_testmodel/run.py --split dev" in p.stdout
    assert not (tmp_path / "results").exists()


def test_dry_run_creates_nothing(testmodel):
    before = (REPO / "pyuser").exists()
    run(["bash", "jobs/run_model.sh", "_testmodel"], env={"DRY_RUN": "1"})
    run(["bash", "jobs/setup_env.sh", "_testmodel"], env={"DRY_RUN": "1"})
    assert (REPO / "pyuser").exists() == before
    assert not (REPO / "pyuser" / "_testmodel").exists()
    assert not (REPO / "third_party" / "tool").exists()


def test_unknown_model_fails():
    for script in ("jobs/run_model.sh", "jobs/setup_env.sh"):
        p = run(["bash", script, "nope"], env={"DRY_RUN": "1"})
        assert p.returncode != 0
        assert "models/nope/env.sh" in p.stderr


def test_missing_run_py_fails(testmodel):
    (testmodel / "run.py").unlink()
    p = run(["bash", "jobs/run_model.sh", "_testmodel"], env={"DRY_RUN": "1"})
    assert p.returncode != 0
    assert "models/_testmodel/run.py" in p.stderr


def test_setup_env_dry_run(testmodel):
    p = run(["bash", "jobs/setup_env.sh", "_testmodel"], env={"DRY_RUN": "1"})
    assert p.returncode == 0, p.stderr
    assert "pip install --user" in p.stdout
    assert " -c " in p.stdout
    assert "singularity pull containers/ngc2412.sif docker://nvcr.io/nvidia/pytorch:24.12-py3" in p.stdout
    assert "org/model-a" in p.stdout and "org/model-b" in p.stdout
    assert "git clone https://github.com/example/tool.git third_party/tool" in p.stdout
    assert "checkout abc123" in p.stdout


def test_scripts_parse():
    for script in (RUN_MODEL, SETUP_ENV):
        assert subprocess.run(["bash", "-n", str(script)]).returncode == 0


def test_model_dirs_complete():
    for d in (REPO / "models").iterdir():
        if d.is_dir() and d.name not in ("radfm", "__pycache__") and not d.name.startswith("_"):
            assert all((d / f).exists() for f in ["__init__.py", "run.py", "requirements.txt", "env.sh", "NOTES.md"])
            reqs = [l for l in (d / "requirements.txt").read_text().splitlines()
                    if l.strip() and not l.startswith("#")]
            assert all("==" in l or "@ git+" in l for l in reqs)
