"""QoQ-Med-VL-7B (Qwen2.5-VL based, research-only licence): 4 axial slices, the colleagues' full-JSON prompt, JSON answer.

Usage (inside the job): python3 -m models.qoqmed7b.run <runner args>
"""
from common import runner
from common.parse import parse_json_answer
from common.schema import full_prompt, slice_preface
from common.volume import pick_slices, to01, to_rgb_images

N_SLICES = 4
IMAGE_SIZE = 448
MAX_NEW_TOKENS = 1024


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
        text = slice_preface(N_SLICES) + full_prompt(scan.dataset)
        raw = self._generate(images, text)
        obj, problems = parse_json_answer(raw)
        return obj or {}, {"prompt": text, "raw": raw, "problems": problems, "slices": slices}


if __name__ == "__main__":
    runner.main(QoQMedAdapter())
