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


def test_requirements_common_pins_nibabel():
    reqs = [l.strip() for l in (REPO / "jobs" / "requirements_common.txt").read_text().splitlines()
            if l.strip() and not l.startswith("#")]
    assert "nibabel==5.3.3" in reqs
    assert all("==" in l for l in reqs)


def test_setup_env_dry_run_installs_common_requirements_and_checks_import(testmodel):
    p = run(["bash", "jobs/setup_env.sh", "_testmodel"], env={"DRY_RUN": "1"})
    assert p.returncode == 0, p.stderr
    (install,) = [l for l in p.stdout.splitlines() if "pip install --user" in l]
    # one pip call, the same torch pins for both files
    assert "-c /pyuser/torch_pins.txt -r /manu/jobs/requirements_common.txt -r /manu/models/_testmodel/requirements.txt" \
        in install
    last = p.stdout.strip().splitlines()[-1]
    assert last.startswith("singularity exec")
    assert "env PYTHONPATH=/manu PYTHONUSERBASE=/pyuser python3 -c import models._testmodel.run" in last


# ---------------------------------------------------------------------------
# Real (not dry) runs in a scratch copy of the repo, with fake singularity / nvidia-smi on PATH.
# ---------------------------------------------------------------------------
FAKE_SINGULARITY = """#!/bin/bash
echo "$*" >> "$FAKE_LOG"
case "$*" in
    *"pip freeze"*) echo "pkg==1.0"; exit 0 ;;
    *"import models."*) exit "${FAKE_IMPORT_STATUS:-0}" ;;
    *"run.py"*) exit "${FAKE_RUN_STATUS:-0}" ;;
esac
exit 0
"""


@pytest.fixture
def scratch(tmp_path):
    """A scratch repo with the job scripts and one model (no HF or git repos), and a bin/ of fakes."""
    repo = tmp_path / "repo"
    (repo / "jobs").mkdir(parents=True)
    for f in ("run_model.sh", "setup_env.sh", "requirements_common.txt"):
        shutil.copy(REPO / "jobs" / f, repo / "jobs" / f)
    model = repo / "models" / "_t"
    model.mkdir(parents=True)
    (model / "env.sh").write_text("SIF=containers/fake.sif\nSIF_SOURCE=docker://example/image\n")
    (model / "run.py").write_text("print('hi')\n")
    (model / "requirements.txt").write_text("numpy==1.0\n")
    (repo / "containers").mkdir()
    (repo / "containers" / "fake.sif").write_text("")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (("singularity", FAKE_SINGULARITY), ("nvidia-smi", "#!/bin/bash\nexit 0\n")):
        (bin_dir / name).write_text(body)
        (bin_dir / name).chmod(0o755)
    import os
    env = {"PATH": f"{bin_dir}:{os.environ['PATH']}", "FAKE_LOG": str(tmp_path / "calls.log")}
    return repo, env, tmp_path / "calls.log"


def test_setup_env_fails_when_the_adapter_does_not_import(scratch):
    repo, env, log = scratch
    p = run(["bash", str(repo / "jobs" / "setup_env.sh"), "_t"], env=dict(env, FAKE_IMPORT_STATUS="1"))
    assert p.returncode != 0
    assert "import models._t.run" in log.read_text()
    assert "models/_t/run.py" in p.stderr
    ok = run(["bash", str(repo / "jobs" / "setup_env.sh"), "_t"], env=env)
    assert ok.returncode == 0, ok.stderr


def test_run_model_writes_freeze_to_slurm_folder_even_when_run_fails(scratch):
    repo, env, _ = scratch
    p = run(["bash", str(repo / "jobs" / "run_model.sh"), "_t", "--split", "dev"],
            env=dict(env, FAKE_RUN_STATUS="3", SLURM_JOB_NAME="lingshu"))
    assert p.returncode == 3
    freeze = repo / "results" / "slurm" / "lingshu_manual_freeze.txt"
    assert freeze.read_text() == "pkg==1.0\n"
    assert not (repo / "results" / "_t").exists()  # the runner makes results/<adapter name>/<split>/ itself


def test_run_model_freeze_name_defaults_to_model(scratch):
    repo, env, _ = scratch
    p = run(["bash", str(repo / "jobs" / "run_model.sh"), "_t", "--split", "dev"], env=env)
    assert p.returncode == 0, p.stderr
    assert (repo / "results" / "slurm" / "_t_manual_freeze.txt").read_text() == "pkg==1.0\n"


def test_run_model_dry_run_shows_freeze_path(testmodel):
    p = run(["bash", "jobs/run_model.sh", "_testmodel", "--split", "dev"],
            env={"DRY_RUN": "1", "SLURM_JOB_NAME": "x", "SLURM_JOB_ID": "42"})
    assert p.returncode == 0, p.stderr
    assert "> results/slurm/x_42_freeze.txt" in p.stdout
    assert "requirements_freeze" not in p.stdout
