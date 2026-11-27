"""Run an extension model on the repository's dev split only."""
import argparse
import json
import os
from pathlib import Path

from common import runner
from common.parse import parse_json_answer
from common.schema import full_prompt, normalize_values, slice_preface
from extensions.cross_modality.prepare_source import PINS, SOURCE_RECEIPT

MODELS = {
    "sgmri": ("SpatialGroundingVQA/Qwen3-VL-8B-SGMRIVQA", "006aa50811a7d70bb3d40704b93cf5f2aae2ecd3"),
    "hulu": ("ZJU-AI4H/Hulu-Med-7B", "258594714a0d3835eb2c9e4cc165a4242e606d71"),
    "nv": ("nvidia/NV-Reason-CT", "386b93e034983f6c1fc841a43833a1b6a0cd9c13"),
    "photon": ("Alibaba-DAMO-Academy/Photon-S1", "785376939e3d42a16a9aa523fef7471119901801"),
    "radsight": ("unstoppableljq/RadSight-4B", "0175f8ae838d5276955d4897ad9f7deb25e01965"),
    "clinfusion": ("Alibaba-DAMO-Academy/ClinFusion-8B", "884c6ad8c3a970080640daca72deb2d216482aa5"),
    "neurovfm": ("mlinslab/neurovfm-llm", "814dbd11f0748acac49e98edee531a9b0ca7037b"),
}
EXTRA_PATHS = {
    "photon": ("source",), "radsight": ("source", "siglip"),
    "clinfusion": ("source", "qwen", "dinov2", "convnext"), "neurovfm": ("source",),
}
REPO = Path(__file__).resolve().parents[2]


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", choices=MODELS)
    p.add_argument("--assets", required=True, type=Path, help="JSON with local, reviewed asset directories")
    p.add_argument("--results", required=True, type=Path, help="new local run directory (no resume)")
    p.add_argument("--data", default="/data")
    p.add_argument("--reference", default="/reference")
    p.add_argument("--datasets", default=",".join(runner.DATASETS))
    p.add_argument("--controls", action="store_true")
    p.add_argument("--ms-flair", action="store_true")
    p.add_argument("--task", choices=("full", "qna"), default=None)
    args = p.parse_args(argv)
    datasets = args.datasets.split(",")
    if len(set(datasets)) != len(datasets) or not set(datasets) <= set(runner.DATASETS):
        p.error(f"--datasets must be distinct keys from {', '.join(runner.DATASETS)}")
    if args.ms_flair and datasets != ["ms"]:
        p.error("--ms-flair requires --datasets ms")
    if args.model == "neurovfm" and args.task:
        p.error("NeuroVFM only supports its native constrained report, not full/Q&A")
    return args


def read_assets(path, kind):
    cfg = json.loads(path.read_text())
    keys = {"model", "revision", *EXTRA_PATHS.get(kind, ())}
    if not isinstance(cfg, dict) or set(cfg) != keys or cfg["revision"] != MODELS[kind][1]:
        raise ValueError(f"{kind} assets must contain {sorted(keys)} and the documented revision")
    out = {}
    for key in keys - {"revision"}:
        asset = Path(cfg[key]).expanduser()
        if not asset.is_absolute() or not asset.is_dir():
            raise ValueError(f"{key} must be an existing absolute directory")
        out[key] = asset
    if "source" in out:
        receipt = json.loads((out["source"] / SOURCE_RECEIPT).read_text())
        if receipt != {"model": kind, "upstream_revision": PINS[kind][1]}:
            raise ValueError(f"Prepare the pinned {kind} source with prepare_source first")
    return out


class StructuredAdapter:
    mode = "full_json"

    def __init__(self, kind, backend):
        self.name = f"ext-{kind}"
        self.model_id = "@".join(MODELS[kind])
        self.backend = backend
        self.slices = kind in ("sgmri", "hulu")

    def load(self):
        self.backend.load()

    def asker(self, scan, volumes):
        def ask(prompt):
            if self.slices:
                prompt = slice_preface() + prompt
            text, _ = self.backend.generate(scan, volumes, prompt, 128)
            return text, None
        return ask

    def predict(self, scan, volumes):
        prompt = full_prompt(scan.dataset)
        if self.slices:
            prompt = slice_preface() + prompt
        text, detail = self.backend.generate(scan, volumes, prompt, 512)
        values, problems = parse_json_answer(text)
        detail = dict(detail, prompt=prompt, raw_text=text, problems=problems)
        if values is None or not any(v is not None for v in normalize_values(scan.dataset, values).values()):
            raise runner.PredictionError("No usable full JSON answer", detail)
        return values, detail


def main(argv=None):
    args = parse_args(argv)
    assets = read_assets(args.assets, args.model)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_DATASETS_OFFLINE="1", TORCH_FORCE_WEIGHTS_ONLY_LOAD="1")
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        os.environ.pop(key, None)
    os.environ["ATTENTION_BACKEND"] = "TORCH_SDPA"
    from extensions.cross_modality.backends import Native, NeuroVFM, Slices

    if args.model == "neurovfm":
        adapter = NeuroVFM(assets, "@".join(MODELS["neurovfm"]))
    else:
        cls = Slices if args.model in ("sgmri", "hulu") else Native
        adapter = StructuredAdapter(args.model, cls(args.model, assets))
    args.results.mkdir(mode=0o700, parents=True, exist_ok=False)
    (args.results / "extension_config.json").write_text(json.dumps({
        "model": adapter.model_id,
        "split": "dev",
        "task": args.task or adapter.mode,
        "assets": {k: str(v) for k, v in assets.items()},
        "revision_verified_by_runner": False,
    }, indent=2) + "\n")
    forwarded = ["--split", "dev", "--splits", str(REPO / "splits.json"),
                 "--datasets", args.datasets, "--data", args.data,
                 "--reference", args.reference, "--results", str(args.results)]
    if args.controls:
        forwarded.append("--controls")
    if args.ms_flair:
        forwarded.append("--ms-flair")
    if args.task:
        forwarded.extend(("--task", args.task))
    runner.main(adapter, forwarded)


if __name__ == "__main__":
    main()
