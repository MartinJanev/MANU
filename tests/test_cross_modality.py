import json
import nibabel as nib
import numpy as np
import pytest

from common.datasets import Scan
from common.runner import PredictionError
from common.schema import full_prompt, slice_preface
from extensions.cross_modality import backends, native, run
from extensions.cross_modality.prepare_source import replace_exact


def test_mri_proxy_preserves_affine_and_is_not_applied_to_ct(tmp_path):
    shape = (5, 6, 8)
    affine = np.diag([0.8, 1.5, 3., 1.])
    mri = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    path = tmp_path / "volume.nii.gz"
    nib.save(nib.Nifti1Image(mri, affine), path)
    image, bounds = native.read_volume(path, ct=False, blank=False)
    assert bounds == pytest.approx(np.percentile(mri[mri > 0], [0.5, 99.5]))
    assert image.affine == pytest.approx(affine)
    assert image.get_fdata().min() == -1000
    assert image.get_fdata().max() == 1000
    with pytest.raises(ValueError, match="CT intensity"):
        native.read_volume(path, ct=True, blank=False)

    ct = np.linspace(-1100, 500, num=mri.size, dtype=np.float32).reshape(shape)
    nib.save(nib.Nifti1Image(ct, affine), path)
    image, bounds = native.read_volume(path, ct=True, blank=False)
    assert bounds is None
    np.testing.assert_array_equal(image.get_fdata(), ct)
    control, _ = native.read_volume(path, ct=True, blank=True)
    assert control.affine == pytest.approx(affine)
    assert np.all(control.get_fdata() == -1000)


def test_slice_transfer_uses_shared_ct_window_and_blank():
    vol = np.zeros((35, 6, 6), dtype=np.float32)
    vol[0] = -1000
    vol[-1] = 500
    images, idx = backends.slice_frames(vol, ct=True, blank=False)
    assert len(set(idx)) == len(images) == 32
    assert (idx[0], idx[-1]) == (0, 34)
    assert images[0].getpixel((0, 0)) == (0, 0, 0)
    assert images[1].getpixel((0, 0)) == (102, 102, 102)
    assert images[-1].getpixel((0, 0)) == (255, 255, 255)
    controls, blank_idx = backends.slice_frames(vol, ct=True, blank=True)
    assert blank_idx == idx
    assert all(image.getbbox() is None for image in controls)
    with pytest.raises(ValueError, match="32 finite"):
        backends.slice_frames(vol[:12], ct=True, blank=False)


def test_structured_adapter_uses_neutral_prompt_and_preserves_parse_failures():
    calls = []

    class Backend:
        def generate(self, scan, volumes, prompt, budget):
            calls.append((prompt, budget))
            return "unstructured answer", {"token_ids": [[42]]}

    adapter = run.StructuredAdapter("sgmri", Backend())
    scan = Scan("ms", "patient05", "scan", ("scan.nii",))
    with pytest.raises(PredictionError) as e:
        adapter.predict(scan, [])
    assert calls == [(slice_preface() + full_prompt("ixi"), 512)]
    assert e.value.details["raw_text"] == "unstructured answer"
    assert e.value.details["token_ids"] == [[42]]
    assert e.value.details["problems"] == ["no_json"]
    assert adapter.asker(scan, [])("Question?") == ("unstructured answer", None)


@pytest.mark.parametrize("model, extras", [
    ("sgmri", ("--split", "paper")),
    ("sgmri", ("--task", "prior")),
    ("neurovfm", ("--task", "qna")),
    ("nv", ("--datasets", "ixi,ixi")),
])
def test_extension_rejects_non_dev_or_unsupported_conditions(model, extras):
    with pytest.raises(SystemExit) as e:
        run.parse_args([model, "--assets", "unused.json", "--results", "unused", *extras])
    assert e.value.code == 2


def test_cli_uses_stock_dev_split_and_never_resumes(tmp_path, monkeypatch):
    model_id, revision = run.MODELS["sgmri"]
    assets = tmp_path / "assets.json"
    out = tmp_path / "run"
    assets.write_text(json.dumps({"model": str(tmp_path), "revision": revision}))
    calls = []

    def capture(adapter, argv):
        calls.append((adapter.name, adapter.model_id, argv))

    monkeypatch.setattr(run.runner, "main", capture)
    # Restore the runner's environment changes when this test is done.
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE",
                "TORCH_FORCE_WEIGHTS_ONLY_LOAD", "ATTENTION_BACKEND", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    argv = ["sgmri", "--assets", str(assets), "--results", str(out), "--task", "qna", "--controls"]
    run.main(argv)
    adapter, recorded_id, forwarded = calls[0]
    assert adapter == "ext-sgmri"
    assert recorded_id == f"{model_id}@{revision}"
    assert forwarded[:4] == ["--split", "dev", "--splits", str(run.REPO / "splits.json")]
    assert forwarded[-3:] == ["--controls", "--task", "qna"]
    with pytest.raises(FileExistsError):
        run.main(argv)
    assert len(calls) == 1


@pytest.mark.parametrize("task, expected_suffix", [("full", ""), ("qna", "-qna")])
def test_extension_writes_runner_records_with_synthetic_ct(tmp_path, task, expected_suffix):
    data = tmp_path / "data"
    path = data / "ct_rate/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz"
    path.parent.mkdir(parents=True)
    nib.save(nib.Nifti1Image(np.ones((4, 4, 36), dtype=np.float32), np.eye(4)), path)
    calls = []

    class FakeBackend:
        def load(self):
            pass

        def generate(self, scan, volumes, prompt, budget):
            calls.append((scan.image_path, float(volumes[0].max()), prompt))
            return ('{"modality":"CT","diagnosis_name":"normal"}' if budget == 512 else "A"), {}

    out = run.runner.run(
        run.StructuredAdapter("hulu", FakeBackend()), split="dev", datasets=["ct_rate"],
        data_dir=data, reference_dir=tmp_path / "reference", results_dir=tmp_path / "results",
        splits={"dev": {"ct_rate": ["valid_1"]}}, controls=True, task=task,
    )
    assert out.parent.name == f"ext-hulu{expected_suffix}"
    processed, = out.glob("*_processed.jsonl")
    controls, = out.glob("*_controls.jsonl")
    assert json.loads(processed.read_text())["output"]["status"] == "success"
    assert json.loads(controls.read_text())["output"]["status"] == "success"
    assert all(maximum == (0. if name.startswith("BLANK__") else 1.) for name, maximum, _ in calls)
    if task == "qna":
        assert calls[0][2] != calls[1][2]  # The option order alternates through the shared runner.


def test_source_edits_reject_upstream_context_drift():
    assert replace_exact("abc", "b", "d") == "adc"
    with pytest.raises(ValueError, match="context changed"):
        replace_exact("abbc", "b", "d")


def test_asset_config_rejects_wrong_revision(tmp_path):
    config = tmp_path / "assets.json"
    config.write_text(json.dumps({"model": str(tmp_path), "revision": "wrong"}))
    with pytest.raises(ValueError, match="documented revision"):
        run.read_assets(config, "sgmri")


def test_asset_config_preserves_radsight_route_alias(tmp_path):
    alias = tmp_path / "VL3-SigLIP-NaViT"
    alias.symlink_to(tmp_path, target_is_directory=True)
    config = tmp_path / "assets.json"
    (tmp_path / run.SOURCE_RECEIPT).write_text(json.dumps({
        "model": "radsight", "upstream_revision": run.PINS["radsight"][1],
    }))
    config.write_text(json.dumps({"model": str(tmp_path), "source": str(tmp_path), "siglip": str(alias),
                                  "revision": run.MODELS["radsight"][1]}))
    assert run.read_assets(config, "radsight")["siglip"] == alias
