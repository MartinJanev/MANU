import subprocess
import sys
import json
from pathlib import Path

from tools.layout import CODE_LAYOUT, check

REPO = Path(__file__).resolve().parent.parent


def test_repo_has_code_layout():
    assert check(REPO, CODE_LAYOUT) == []


def test_old_paths_are_gone():
    for p in ["code", "Dockerfile", "job_radfm.sh", "prezemi_podatoci.sh", "proveri_podatoci.py"]:
        assert not (REPO / p).exists()


def test_reference_has_colleague_files():
    assert len(list((REPO / "reference").glob("*_processed.jsonl"))) == 14


def test_splits_json():
    s = json.loads((REPO / "splits.json").read_text())
    assert s["dev"] == {"ixi": ["IXI002"], "molab": ["010005N"], "ms": ["patient05"], "ct_rate": ["valid_1"]}
    assert s["eval"]["molab"] == ["010020N", "010024N", "010029N"]
    assert s["eval"]["ct_rate"] == ["valid_2", "valid_3", "valid_4"]


def test_cli_prints_ok_or_missing(tmp_path):
    cli = [sys.executable, str(REPO / "tools/layout.py")]
    ok = subprocess.run(cli + [str(REPO)], capture_output=True, text=True)
    assert (ok.returncode, ok.stdout.splitlines()[-1]) == (0, "OK")
    bad = subprocess.run(cli + [str(tmp_path)], capture_output=True, text=True)
    assert bad.returncode == 1 and "README.md" in bad.stdout and "OK" not in bad.stdout
