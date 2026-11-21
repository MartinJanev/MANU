import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_reference_has_colleague_files():
    assert len(list((REPO / "reference").glob("*_processed.jsonl"))) == 14


def test_splits_json():
    s = json.loads((REPO / "splits.json").read_text())
    assert s["dev"] == {"ixi": ["IXI002"], "molab": ["010005N"], "ms": ["patient05"], "ct_rate": ["valid_1"]}
    # tools/make_split.py: 30 per dataset (MS has 30 patients, one is dev), never a dev patient, and the
    # original eval patients are kept so their recorded results resume instead of rerunning
    assert {d: len(p) for d, p in s["eval"].items()} == {"ixi": 30, "molab": 30, "ms": 29, "ct_rate": 30}
    original = {"ixi": ["IXI012", "IXI013", "IXI014"], "molab": ["010020N", "010024N", "010029N"],
                "ms": ["patient09", "patient10", "patient11"], "ct_rate": ["valid_2", "valid_3", "valid_4"]}
    for d, patients in s["eval"].items():
        assert set(original[d]) <= set(patients) and not set(patients) & set(s["dev"][d])
