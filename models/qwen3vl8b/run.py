"""Qwen3-VL-8B-Instruct (general VLM, Apache-2.0): 10 axial slices as one video, full-JSON prompt, JSON answer.

The general baseline for the Qwen-based medical models. Slices and frames as HealthGPT-Pro-27B (the other video
adapter): pick_slices, 10 frames of 448x448, `"sample_fps": 2.0`; the calls follow the video example of the Qwen3-VL
README (github.com/QwenLM/Qwen3-VL): process_vision_info(image_patch_size=16, return_video_kwargs=True,
return_video_metadata=True), the processor with do_resize=False, generate, trim the prompt, batch_decode.
Greedy decoding, max_new_tokens 1024 (an Instruct model: no reasoning block). A failed parse raises
PredictionError: an error record, raw text in details.

Usage: sbatch -J qwen3vl8b jobs/run_model.sh qwen3vl8b --split dev|eval|full [--controls] [--limit N]
       (jobs/run_model.sh runs `python3 /manu/models/qwen3vl8b/run.py --split ...` inside the container)
"""
from common import runner
from common.parse import parse_json_answer
from common.schema import full_prompt, normalize_values, slice_preface
from common.volume import pick_slices, to01, to_rgb_images

N_SLICES = 10  # as HealthGPT-Pro-27B
IMAGE_SIZE = 448  # a multiple of 32 (patch 16 x merge 2): qwen-vl-utils keeps 448x448
MAX_NEW_TOKENS = 1024
SAMPLE_FPS = 2.0
IMAGE_PATCH_SIZE = 16  # README: 16 for Qwen3-VL


class Qwen3VLAdapter:
    name = "qwen3-vl-8b"
    model_id = "Qwen/Qwen3-VL-8B-Instruct"
    mode = "full_json"

    def load(self):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda")
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.model_id)

    def _generate(self, frames, text):
        import torch
        from qwen_vl_utils import process_vision_info

        # The README's video inference, with our frames and question
        messages = [{
            "role": "user",
            "content": [
                {"type": "video", "video": frames, "sample_fps": SAMPLE_FPS},
                {"type": "text", "text": text},
            ],
        }]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos, video_kwargs = process_vision_info(
            messages, image_patch_size=IMAGE_PATCH_SIZE, return_video_kwargs=True, return_video_metadata=True)
        videos, video_metadatas = zip(*videos)  # always one video here: (frames tensor, metadata)
        videos, video_metadatas = list(videos), list(video_metadatas)
        inputs = self.processor(text=prompt, images=images, videos=videos, video_metadata=video_metadatas,
                                return_tensors="pt", do_resize=False, **video_kwargs).to(self.model.device)
        with torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, num_beams=1)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    def predict(self, scan, volumes):
        vol01 = to01(volumes[0], scan.dataset)
        slices = pick_slices(vol01, N_SLICES)
        frames = to_rgb_images(vol01, slices, IMAGE_SIZE)
        text = slice_preface() + full_prompt(scan.dataset)
        raw = self._generate(frames, text)
        obj, problems = parse_json_answer(raw)
        details = {"prompt": text, "raw": raw, "problems": problems, "slices": slices}
        if obj is None:
            raise runner.PredictionError(f"no usable JSON in the answer ({', '.join(problems)})", details)
        if not any(v is not None for v in normalize_values(scan.dataset, obj).values()):
            raise runner.PredictionError("JSON without any valid schema value", details)
        return obj, details


if __name__ == "__main__":
    runner.main(Qwen3VLAdapter())
