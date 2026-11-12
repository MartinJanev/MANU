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
    assert {k: s["dev"][k] for k in ("ixi", "molab", "ms")} == {"ixi": ["IXI002"], "molab": ["010005N"], "ms": ["patient05"]}
    assert s["eval"]["molab"] == ["010020N", "010024N", "010029N"]
