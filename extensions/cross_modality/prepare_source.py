"""Copy a clean pinned upstream checkout and apply local inference-only changes."""
import argparse
import ast
import json
from pathlib import Path
import shutil
import subprocess

PINS = {
    "photon": ("alibaba-damo-academy/Photon", "88c37554a05ca1bc36a0e22f99a86d87dd40fea3"),
    "radsight": ("alibaba-damo-academy/damo-RadSight", "b506fbbbe9199cabd9132432f4bbede5cee25ed3"),
    "clinfusion": ("alibaba-damo-academy/ClinFusion", "2db9d000f52479b42be75c952c1bac17d4868d07"),
    "neurovfm": ("MLNeurosurg/neurovfm", "eb41760f200e3a542fd7bce627513d9a625f0997"),
}
SOURCE_RECEIPT = ".manu-extension-source.json"


def replace_exact(text, original, replacement, count=1):
    if text.count(original) != count:
        raise ValueError(f"Upstream source context changed: {original}")
    return text.replace(original, replacement)


def edit(path, *changes):
    text = path.read_text()
    for change in changes:
        text = replace_exact(text, *change)
    compile(text, str(path), "exec")
    path.write_text(text)


def lazy_exports(path):
    """Defer NeuroVFM's training imports until those symbols are requested."""
    export_map, assignments = {}, []
    text = path.read_text()
    for node in ast.parse(text).body:
        if isinstance(node, ast.ImportFrom):
            for name in node.names:
                if name.name == "*":
                    raise ValueError(f"Unexpected star import: {path}")
                export_map[name.asname or name.name] = ("." * node.level + (node.module or ""), name.name)
        elif isinstance(node, ast.Assign):
            assignments.append(ast.get_source_segment(text, node))
        elif not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)):
            raise ValueError(f"Unexpected initializer statement: {path}")
    code = '"""Inference-only lazy upstream exports."""\nimport importlib\n'
    code += "\n".join(assignments) + "\n_EXPORTS = " + repr(export_map) + """

def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, attribute = _EXPORTS[name]
    value = getattr(importlib.import_module(module, __name__), attribute)
    globals()[name] = value
    return value
"""
    compile(code, str(path), "exec")
    path.write_text(code)


def apply(kind, out):
    if kind == "clinfusion":
        edit(out / "custom_model/medevalkit_adapter_qwen3_vl.py",
             ('attn_implementation="flash_attention_2"',
              'attn_implementation="sdpa", local_files_only=True, use_safetensors=True'),
             ("load_info = load_sharded_checkpoint(self.llm, model_path, strict=False, prefer_safe=True)",
              "from extensions.cross_modality.native import strict_shards\n"
              "        strict_shards(self.llm, model_path)\n"
              "        load_info = load_sharded_checkpoint(self.llm, model_path, strict=True, prefer_safe=True)\n"
              "        self.checkpoint_load_info = load_info._asdict()"),
             ("return torch.zeros(1, *target_size, dtype=torch.float32)",
              'raise RuntimeError("MONAI required; no zero fallback")'))
        edit(out / "custom_model/openclip_encoder.py",
             ("import deepspeed\n", ""),
             ("    if is_deepspeed_zero3_enabled():\n\n        error_msgs = []",
              "    if is_deepspeed_zero3_enabled():\n        import deepspeed\n\n        error_msgs = []"))
    elif kind == "radsight":
        path = out / "radsight/mm_utils.py"
        text = path.read_text()
        imports = {
            "load_video_from_ids": ["import cv2", "import imageio", "from decord import VideoReader, cpu"],
            "load_video": ["import ffmpeg"],
        }
        for lines in imports.values():
            for line in lines:
                text = replace_exact(text, line + "\n", "")
        nodes = {node.name: node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef)}
        lines = text.splitlines(keepends=True)
        for name in sorted(imports, key=lambda n: nodes[n].lineno, reverse=True):
            # Insert before the first statement as in the native CT pilot.
            start = nodes[name].body[0].lineno - 1
            lines[start:start] = ["    " + line + "\n" for line in imports[name]]
        code = "".join(lines)
        compile(code, str(path), "exec")
        path.write_text(code)
    elif kind == "neurovfm":
        pkg = out / "neurovfm"
        for directory in ("", "models", "data", "systems", "pipelines"):
            lazy_exports(pkg / directory / "__init__.py")
        edit(pkg / "models/vlm.py",
             ("self.llm: PreTrainedModel = AutoModelForCausalLM.from_pretrained(",
              "self.llm, self.loading_info = AutoModelForCausalLM.from_pretrained("),
             ("trust_remote_code=True,", "trust_remote_code=False, local_files_only=True,", 2),
             ("attn_implementation=attn_implementation,\n        )",
              "attn_implementation=attn_implementation,\n            output_loading_info=True, use_safetensors=True,\n        )"),
             ("        self.tokenizer = AutoTokenizer.from_pretrained(",
              "        if any(self.loading_info.get(k) for k in "
              '("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):\n'
              '            raise RuntimeError(f"Incomplete NeuroVFM LM restoration: {self.loading_info}")\n'
              "        self.tokenizer = AutoTokenizer.from_pretrained("))
        edit(pkg / "pipelines/generator.py",
             ('torch.load(enc_path, map_location="cpu")', 'torch.load(enc_path, map_location="cpu", weights_only=True)'),
             ('torch.load(conn_path, map_location="cpu")', 'torch.load(conn_path, map_location="cpu", weights_only=True)'),
             ("model.vision_encoder.load_state_dict(enc_state, strict=False)",
              "info = model.vision_encoder.load_state_dict(enc_state, strict=True)\n"
              "        model.encoder_loading_info = info._asdict()"),
             ("model.vision_connector.load_state_dict(conn_state)",
              "info = model.vision_connector.load_state_dict(conn_state, strict=True)\n"
              "        model.connector_loading_info = info._asdict()"),
             ('logging.warning("vision_encoder.pt not found. Vision encoder initialized randomly.")',
              'raise FileNotFoundError("vision_encoder.pt required")'),
             ('logging.warning("vision_connector.pt not found. Connector initialized randomly.")',
              'raise FileNotFoundError("vision_connector.pt required")'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", choices=PINS)
    p.add_argument("checkout", type=Path)
    p.add_argument("output", type=Path, help="new local directory; do not commit it")
    args = p.parse_args()
    checkout = args.checkout.resolve()
    output = args.output.resolve()
    if output.is_relative_to(checkout):
        p.error("output must be outside the source checkout")

    def git(*command):
        return subprocess.check_output(["git", "-C", str(checkout), *command], text=True).strip()

    if (Path(git("rev-parse", "--show-toplevel")) != checkout
            or git("rev-parse", "HEAD") != PINS[args.model][1]
            or git("status", "--porcelain", "--untracked-files=all", "--ignored")):
        p.error("expected a clean checkout of the documented source commit")
    shutil.copytree(checkout, output, symlinks=True, ignore=shutil.ignore_patterns(".git"))
    apply(args.model, output)
    with (output / SOURCE_RECEIPT).open("x") as f:
        json.dump({"model": args.model, "upstream_revision": PINS[args.model][1]}, f)


if __name__ == "__main__":
    main()
