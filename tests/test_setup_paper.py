"""jobs/setup_paper.sh: the one command of Block 1 on the login node."""
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture
def login_node(tmp_path):
    repo, bin_dir = tmp_path / "repo", tmp_path / "bin"
    (repo / "jobs").mkdir(parents=True)
    bin_dir.mkdir()
    shutil.copy(REPO / "jobs/setup_paper.sh", repo / "jobs")
    log = tmp_path / "calls.log"
    fakes = {
        # As on the real login node (Block 1, 22 Nov): python3 is there, huggingface_hub is not.
        "python3": f'echo "login-python $*" >> "{log}"\n'
                   'case "$*" in *huggingface_hub*|*prezemi_ct_rate*)\n'
                   "  echo \"ModuleNotFoundError: No module named 'huggingface_hub'\" >&2; exit 1 ;;\n"
                   "esac\nexit 0",
        "singularity": f'echo "container $*" >> "{log}"\n'
                       'case "$*" in\n'
                       '  *get_hf_file_metadata*) exit "${FAKE_ACCESS_STATUS:-0}" ;;\n'
                       '  *"import huggingface_hub"*) exit "${FAKE_IMAGE_STATUS:-0}" ;;\n'
                       "esac\nexit 0",
        "nohup": f'echo "nohup $*" >> "{log}"\nexit 0',
        "df": 'echo "Filesystem 1024-blocks Used Available Capacity Mounted"\n'
              'echo "fs 999999999 1 ${FAKE_FREE_KB:-200000000} 1% /home"',
    }
    for name, body in fakes.items():
        (bin_dir / name).write_text(f"#!/bin/bash\n{body}\n")
        (bin_dir / name).chmod(0o755)

    def go(**env):
        p = subprocess.run(["bash", str(repo / "jobs/setup_paper.sh")], capture_output=True, text=True,
                           env={"PATH": f"{bin_dir}:/usr/bin:/bin", **env})
        return p, log.read_text() if log.exists() else ""
    return go


def test_needs_the_token_before_doing_anything(login_node):
    p, calls = login_node()
    assert p.returncode != 0 and "HF_TOKEN" in p.stderr and "make_split" not in calls and "nohup" not in calls


def test_works_when_the_login_python_has_no_huggingface_hub(login_node):
    p, calls = login_node(HF_TOKEN="x")
    assert p.returncode == 0, p.stderr
    login = [line for line in calls.splitlines() if line.startswith("login-python")]
    assert login == ["login-python tools/make_split.py --paper --out results/splits_proposed.json"]  # stdlib only
    assert "containers/ngc2412.sif" in calls and "./pyuser/hlip:/pyuser" in calls


def test_stops_when_the_image_cannot_download(login_node):
    p, calls = login_node(HF_TOKEN="x", FAKE_IMAGE_STATUS="1")
    assert p.returncode != 0 and "STOP" in p.stderr and "nohup" not in calls


def test_a_bad_token_stops_before_the_background_job(login_node):
    p, calls = login_node(HF_TOKEN="x", FAKE_ACCESS_STATUS="1")
    assert p.returncode != 0 and "CT-RATE" in p.stderr and "nohup" not in calls and "make_split" not in calls


def test_stops_when_the_disk_is_too_full(login_node):
    p, calls = login_node(HF_TOKEN="x", FAKE_FREE_KB="10000000")
    assert p.returncode != 0 and "GB" in p.stderr and "nohup" not in calls


def test_makes_the_split_then_starts_downloads_in_the_background(login_node):
    p, calls = login_node(HF_TOKEN="secret-token")
    assert p.returncode == 0, p.stderr
    assert calls.index("tools/make_split.py --paper") < calls.index("nohup")
    (background,) = [line for line in calls.split("nohup ")[1:]]
    for part in ("singularity exec", "prezemi_ct_rate.py --split paper --splits /results/splits_proposed.json",
                 "--data /data", "setup_env.sh m3dlamed_phi3", "setup_env.sh m3dlamed_llama2"):
        assert part in background
    assert "secret-token" not in calls + p.stdout + p.stderr  # never in a command line or a log
