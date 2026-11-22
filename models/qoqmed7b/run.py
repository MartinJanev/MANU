"""QoQ-Med-VL-7B (Qwen2.5-VL based; MIT, research use only): 4 axial slices, full-JSON prompt, JSON answer.

QoQ-Med was trained to answer as: a ```json block of "bbox_2d" boxes, reasoning in <think>...</think>, the final
answer in \\boxed{...} (examples/format_prompt/medical_format.jinja in its repository). So only the text after the
last </think> is parsed, and within it the last \\boxed{...} if there is one; an unclosed <think> gives no values.

Usage: sbatch -J qoqmed7b jobs/run_model.sh qoqmed7b --split dev|eval|full [--controls] [--limit N]
       (jobs/run_model.sh runs `python3 /manu/models/qoqmed7b/run.py --split ...` inside the container)
"""
from common import runner
from common.parse import parse_json_answer
from common.schema import full_prompt, slice_preface
from common.volume import pick_slices, to01, to_rgb_images

N_SLICES = 4
IMAGE_SIZE = 448
MAX_NEW_TOKENS = 2048  # it reasons in <think> before answering
THINK_OPEN, THINK_CLOSE, BOXED = "<think>", "</think>", "\\boxed{"


def _last_boxed(text):
    """Content of the last \\boxed{...} (balanced braces, braces inside JSON strings ignored), or None."""
    start = text.rfind(BOXED)
    if start == -1:
        return None
    start += len(BOXED)
    depth, in_string, escaped = 1, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif c == "\\":
                escaped = True
            elif c == '"':
                in_string = False
        elif c == '"':
            in_string = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i]
    return text[start:]  # unclosed (truncated answer): let parse_json_answer reject it


def _answer_part(raw):
    """(text to parse, where it came from): "boxed", "after_think" or "whole"; (None, None) if <think> never closes."""
    if raw.rfind(THINK_OPEN) > raw.rfind(THINK_CLOSE):
        return None, None  # generation stopped inside the reasoning: any JSON there is not an answer
    text, source = raw, "whole"
    close = raw.rfind(THINK_CLOSE)
    if close != -1:
        text, source = raw[close + len(THINK_CLOSE):], "after_think"
    boxed = _last_boxed(text)
    if boxed is not None:
        return boxed, "boxed"
    return text, source


class QoQMedAdapter:
    name = "qoq-med-vl-7b"
    model_id = "ddvd233/QoQ-Med-VL-7B"
    mode = "full_json"

    def load(self):
        from models._qwen25vl import load_qwen25vl

        self.model, self.processor = load_qwen25vl(self.model_id)

    def _generate(self, images, text):
        from models._qwen25vl import generate_images

        return generate_images(self.model, self.processor, images, text, MAX_NEW_TOKENS)

    def predict(self, scan, volumes):
        vol01 = to01(volumes[0], scan.dataset)
        slices = pick_slices(vol01, N_SLICES)
        images = to_rgb_images(vol01, slices, IMAGE_SIZE)
        text = slice_preface() + full_prompt(scan.dataset)
        raw = self._generate(images, text)
        answer, parsed_from = _answer_part(raw)
        if answer is None:
            obj, problems = None, ["truncated_think"]
        else:
            obj, problems = parse_json_answer(answer)
        details = {"prompt": text, "raw": raw, "problems": problems, "slices": slices, "parsed_from": parsed_from}
        return obj or {}, details


if __name__ == "__main__":
    runner.main(QoQMedAdapter())
