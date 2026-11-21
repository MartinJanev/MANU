import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = ["tools/sync_to_cluster.sh", "tools/fetch_results.sh"]


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
    assert "-avu" in a  # -u: local files re-scored by tools/reparse.py are not overwritten by older cluster copies
    assert "/archive/" in a
    assert a[-2:] == ["u@h:/r/manu/results/", "results/"]
