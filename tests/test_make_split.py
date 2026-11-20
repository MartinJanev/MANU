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


def test_ct_rate_draws_from_labels_csv_not_disk(tmp_path):
    data = _tree(tmp_path)
    csv_path = data / "ct_rate/dataset/multi_abnormality_labels/valid_predicted_labels.csv"
    csv_path.parent.mkdir(parents=True)
    names = [f"valid_{i}_a_1.nii.gz" for i in range(1, 11)] + ["valid_5_a_2.nii.gz", "valid_11_b_1.nii.gz"]
    csv_path.write_text("VolumeName,Lung nodule\n" + "".join(f"{n},0\n" for n in names))
    splits = {"dev": {**SPLITS["dev"], "ct_rate": ["valid_1"]}, "eval": {**SPLITS["eval"], "ct_rate": ["valid_2"]}}
    proposed, counts = propose(data, splits, per_dataset=5, seed=0)
    ct = proposed["eval"]["ct_rate"]
    assert len(ct) == 5 and "valid_2" in ct and "valid_1" not in ct and "valid_11" not in ct
    assert counts["ct_rate"] == (5, 5)
