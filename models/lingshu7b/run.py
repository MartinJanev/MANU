"""Lingshu-7B (Qwen2.5-VL based medical MLLM): 8 axial slices, the colleagues' full-JSON prompt, JSON answer.

Usage: sbatch -J lingshu7b jobs/run_model.sh lingshu7b --split dev|eval|full [--controls] [--limit N]
       (jobs/run_model.sh runs `python3 /manu/models/lingshu7b/run.py --split ...` inside the container)
"""
from common import runner
from common.parse import parse_json_answer
from common.schema import full_prompt, slice_preface
from common.volume import pick_slices, to01, to_rgb_images

N_SLICES = 8
IMAGE_SIZE = 448
QNA_MAX_NEW_TOKENS = 128  # one question
MAX_NEW_TOKENS = 1024


class LingshuAdapter:
    name = "lingshu-7b"
    model_id = "lingshu-medical-mllm/Lingshu-7B"
    mode = "full_json"

    def load(self):
        from models._qwen25vl import load_qwen25vl

        self.model, self.processor = load_qwen25vl(self.model_id)

    def _generate(self, images, text, max_new_tokens=MAX_NEW_TOKENS):
        from models._qwen25vl import generate_images

        return generate_images(self.model, self.processor, images, text, max_new_tokens)

    def predict(self, scan, volumes):
        vol01 = to01(volumes[0], scan.dataset)
        slices = pick_slices(vol01, N_SLICES)
        images = to_rgb_images(vol01, slices, IMAGE_SIZE)
        text = slice_preface() + full_prompt(scan.dataset, scan.prior)
        raw = self._generate(images, text)
        obj, problems = parse_json_answer(raw)
        return obj or {}, {"prompt": text, "raw": raw, "problems": problems, "slices": slices}

    def asker(self, scan, volumes):
        """One question per call, over the same slices (runner --task qna)."""
        vol01 = to01(volumes[0], scan.dataset)
        images = to_rgb_images(vol01, pick_slices(vol01, N_SLICES), IMAGE_SIZE)
        return lambda question: (self._generate(images, slice_preface() + question, QNA_MAX_NEW_TOKENS), None)


if __name__ == "__main__":
    runner.main(LingshuAdapter())
