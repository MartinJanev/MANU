from tools.make_split import propose

SPLITS = {"dev": {"ixi": ["IXI001"], "molab": [], "ms": [], "ct_rate": []},
          "eval": {"ixi": ["IXI002"], "molab": [], "ms": [], "ct_rate": []}}


def _tree(tmp_path, n_ixi=10):
    files = [f"ixi/T1/IXI{i:03d}-HH-0001-T1.nii.gz" for i in range(1, n_ixi + 1)]
    files += ["molab/nifti/10020_19010822_TRA_3D_T1+GD_img.nii", "molab/nifti/10020_19011029_TRA_3D_T1_img.nii"]
    for rel in files:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
    return tmp_path


def test_keeps_eval_excludes_dev_caps_and_is_deterministic(tmp_path):
    data = _tree(tmp_path)
    proposed, counts = propose(data, SPLITS, per_dataset=4, seed=0)
    ixi = proposed["eval"]["ixi"]
    assert len(ixi) == 4 and "IXI002" in ixi and "IXI001" not in ixi
    assert proposed["dev"] == SPLITS["dev"]
    assert propose(data, SPLITS, per_dataset=4, seed=0)[0] == proposed
    assert proposed["eval"]["molab"] == ["010020N"] and counts["molab"] == (1, 2)  # fewer than 4: all, scans counted
    assert counts["ms"] == (0, 0)
    assert SPLITS["eval"]["ixi"] == ["IXI002"]  # the input is not modified
