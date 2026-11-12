import os
import shutil
import stat
import subprocess
from datetime import date
from pathlib import Path

import pytest

from tools.layout import CLUSTER_LAYOUT, check

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = ["tools/migrate_cluster.sh", "tools/sync_to_cluster.sh", "tools/fetch_results.sh"]
SKIP = shutil.ignore_patterns(".git", ".venv", ".superpowers", "__pycache__", ".pytest_cache",
                              "results", "data", "hf_cache", "pyuser", "third_party")


def touch(root: Path, rel: str, text: str = "x"):
    p = root / rel
    if rel.endswith("/"):
        p.mkdir(parents=True, exist_ok=True)
    else:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)


@pytest.fixture
def old_tree(tmp_path):
    """The cluster before migration: old flat layout + cluster-only folders, then the synced repo."""
    root = tmp_path / "manu"
    for rel in ["code/run_radfm.py", "job_radfm.sh", "Dockerfile", "prezemi_molab.py",
                "results/radfm_healthy_processed.jsonl", "results/radfm_healthy_details.jsonl",
                "results/3d-radfm_1946.out", "RadFM/Quick_demo/x",
                "pyuser/lib/python3.10/site-packages/x", "data/ixi/T1/a.nii.gz",
                "data/molab/nifti/", "data/open_ms_data/cross_sectional/raw/",
                "containers/radfm.sif", "hf_cache/", "notes.txt"]:
        touch(root, rel, "old:" + rel)
    shutil.copytree(REPO, root, ignore=SKIP, dirs_exist_ok=True)
    return root


def run_migrate(root: Path, *args: str):
    return subprocess.run(["bash", "tools/migrate_cluster.sh", *args], cwd=root,
                          capture_output=True, text=True)


def snapshot(root: Path):
    return {str(p.relative_to(root)): (p.read_bytes() if p.is_file() else None)
            for p in sorted(root.rglob("*"))}


def test_dry_run_changes_nothing(old_tree):
    before = snapshot(old_tree)
    r = run_migrate(old_tree)
    assert r.returncode == 0, r.stderr
    assert "mv " in r.stdout
    assert snapshot(old_tree) == before


def test_apply_produces_cluster_layout(old_tree):
    r = run_migrate(old_tree, "--apply")
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.splitlines()[-1] == "OK"
    assert check(old_tree, CLUSTER_LAYOUT) == []
    assert (old_tree / "third_party/RadFM/Quick_demo/x").exists()
    assert (old_tree / "pyuser/radfm/lib/python3.10/site-packages/x").exists()
    assert (old_tree / "results/radfm/full/radfm_healthy_processed.jsonl").exists()
    assert (old_tree / "results/radfm/full/radfm_healthy_details.jsonl").exists()
    assert (old_tree / "results/slurm/3d-radfm_1946.out").exists()
    assert (old_tree / "data/ixi/T1/a.nii.gz").exists()
    assert (old_tree / "archive/code/cluster/run_radfm.py").exists()
    assert (old_tree / "archive/cluster_root/Dockerfile").read_text() == "old:Dockerfile"
    assert (old_tree / "archive/cluster_root/job_radfm.sh").exists()
    assert (old_tree / "data_tools/prezemi_molab.py").exists()
    # the synced new files were not overwritten by the old root files
    assert (old_tree / "containers/radfm.Dockerfile").read_text() == (REPO / "containers/radfm.Dockerfile").read_text()
    # originals of the copied results stay archived
    archived = old_tree / f"results/archive/cluster_{date.today():%Y-%m-%d}"
    assert (archived / "radfm_healthy_processed.jsonl").exists()
    assert not (old_tree / "results/radfm_healthy_processed.jsonl").exists()


def test_apply_is_idempotent(old_tree):
    run_migrate(old_tree, "--apply")
    before = snapshot(old_tree)
    r = run_migrate(old_tree, "--apply")
    assert r.returncode == 0, r.stderr
    assert "mv " not in r.stdout and "cp " not in r.stdout
    assert snapshot(old_tree) == before


def test_unknown_files_left(old_tree):
    r = run_migrate(old_tree, "--apply")
    assert (old_tree / "notes.txt").exists()
    assert "left in place: notes.txt" in r.stdout
    left = [l for l in r.stdout.splitlines() if l.startswith("left in place")]
    assert left == ["left in place: notes.txt"]


def test_existing_destination_is_never_overwritten(old_tree):
    touch(old_tree, "third_party/RadFM/keep", "new")
    touch(old_tree, "results/slurm/3d-radfm_1946.out", "new")
    r = run_migrate(old_tree, "--apply")
    assert "skip (exists): third_party/RadFM" in r.stdout
    assert "skip (exists): results/slurm/3d-radfm_1946.out" in r.stdout
    assert (old_tree / "RadFM/Quick_demo/x").exists()
    assert (old_tree / "third_party/RadFM/keep").read_text() == "new"
    assert not (old_tree / "third_party/RadFM/Quick_demo").exists()
    assert (old_tree / "results/slurm/3d-radfm_1946.out").read_text() == "new"
    assert (old_tree / "results/3d-radfm_1946.out").read_text() == "old:results/3d-radfm_1946.out"


def test_refuses_outside_repo_root(tmp_path):
    r = subprocess.run(["bash", str(REPO / "tools/migrate_cluster.sh"), "--apply"],
                       cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 1
    assert "repo root" in r.stderr
    assert list(tmp_path.iterdir()) == []


def test_sync_scripts_parse():
    for s in SCRIPTS:
        assert subprocess.run(["bash", "-n", str(REPO / s)]).returncode == 0


@pytest.fixture
def fake_rsync(tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    rs = bindir / "rsync"
    rs.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$RSYNC_ARGS"\n')
    rs.chmod(rs.stat().st_mode | stat.S_IEXEC)
    args = tmp_path / "args.txt"
    env = {**os.environ, "PATH": f"{bindir}:{os.environ['PATH']}", "RSYNC_ARGS": str(args)}
    return env, args


def test_sync_rsync_arguments(fake_rsync):
    env, args = fake_rsync
    env.update(CLUSTER="u@h", REMOTE="/r/manu", RSYNC_DRY="1")
    subprocess.run(["bash", str(REPO / "tools/sync_to_cluster.sh")], env=env, check=True, cwd="/")
    a = args.read_text().splitlines()
    assert "-n" in a and "--exclude-from=.gitignore" in a and ".git" in a and ".superpowers/" in a
    assert not any("delete" in x for x in a)
    assert a[-2:] == ["./", "u@h:/r/manu/"]
    assert a.index("--include=/.manu_commit") < a.index("--exclude-from=.gitignore")
    assert (REPO / ".manu_commit").read_text().strip()


def test_fetch_rsync_arguments(fake_rsync, tmp_path):
    env, args = fake_rsync
    env.update(CLUSTER="u@h", REMOTE="/r/manu")
    work = tmp_path / "local"
    (work / "tools").mkdir(parents=True)
    shutil.copy(REPO / "tools/fetch_results.sh", work / "tools")
    subprocess.run(["bash", "tools/fetch_results.sh"], env=env, check=True, cwd=work)
    a = args.read_text().splitlines()
    assert "-n" not in a and "--delete" not in a
    assert "/archive/" in a
    assert a[-2:] == ["u@h:/r/manu/results/", "results/"]
