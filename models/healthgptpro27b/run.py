"""HealthGPT-Pro-27B (Qwen3.6-27B based medical MLLM; Apache-2.0): 10 axial slices as one video, full-JSON prompt.

The calls follow the 3D-volume example of the model card (the same call sequence as inference/infer_3d_volume.py
of github.com/ZJU4HealthCare/HealthGPT, written there for the Qwen3-VL based 4B/8B): the slices go in as one video
of PIL frames (`"sample_fps": 2.0`), `process_vision_info` of qwen-vl-utils prepares it, the processor gets
`do_resize=False`.
If the chat template's generation prompt ends with an open <think> (Qwen3.6 thinking mode), the decoded output
may start inside the reasoning: only the text after the last </think> is parsed. An answer counts as cut off inside
the reasoning (no values, `truncated_think`) only when the prompt opened <think> AND generation used all
max_new_tokens: in job 2777 the prompt opened <think> yet every answer was plain JSON with no </think> (~200 tokens,
stopped on its own), so the tag alone is no evidence of truncation.

Usage: sbatch -J healthgptpro27b jobs/run_model.sh healthgptpro27b --split dev|eval|full [--controls] [--limit N]
       (jobs/run_model.sh runs `python3 /manu/models/healthgptpro27b/run.py --split ...` inside the container)
"""
from common import runner
from common.parse import parse_json_answer
from common.schema import full_prompt, slice_preface
from common.volume import pick_slices, to01, to_rgb_images

N_SLICES = 10
IMAGE_SIZE = 448  # a multiple of 32 (patch 16 x merge 2): qwen-vl-utils keeps 448x448, 196 tokens per frame pair
MAX_NEW_TOKENS = 2048  # it reasons in <think> before answering
QNA_MAX_NEW_TOKENS = 512  # one question: room for a short reasoning
SAMPLE_FPS = 2.0  # the card's 3D example
IMAGE_PATCH_SIZE = 16  # process_vision_info argument of the card (the vision tower's patch size)
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"


def _answer_part(raw, cut_off):
    """Text after the last </think>, or None if the reasoning never closes. `cut_off`: the generation prompt
    ended in <think> and the output used all max_new_tokens, so it stopped inside the reasoning."""
    # ponytail: if the tokenizer strips </think> AND the model reasons before answering, a draft JSON inside the
    # reasoning could be parsed; decode the ids after the </think> token if details.raw ever shows prose before JSON.
    text = (THINK_OPEN if cut_off else "") + raw
    close = text.rfind(THINK_CLOSE)
    if text.rfind(THINK_OPEN) > close:
        return None  # generation stopped inside the reasoning: any JSON there is not an answer
    return text if close == -1 else text[close + len(THINK_CLOSE):]  # -1: no reasoning at all


def answer_values(raw, cut_off):
    """(JSON object or None, problems) of one decoded output; also used by tools/reparse.py."""
    answer = _answer_part(raw, cut_off)
    if answer is None:
        return None, ["truncated_think"]
    return parse_json_answer(answer)


class HealthGPTProAdapter:
    name = "healthgpt-pro-27b"
    model_id = "lintw/HealthGPT-Pro-27B"
    mode = "full_json"

    def load(self):
        import torch
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

        # The card's loading code, on one GPU (the card: device_map="auto", flash_attention_2)
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            self.model_id,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map="cuda",
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.model_id)

    def _generate(self, frames, text, max_new_tokens):
        from qwen_vl_utils import process_vision_info

        # The card's 3D-volume inference, with our frames and question
        messages = [{
            "role": "user",
            "content": [
                {"type": "video", "video": frames, "sample_fps": SAMPLE_FPS},
                {"type": "text", "text": text},
            ],
        }]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos, video_kwargs = process_vision_info(
            messages,
            image_patch_size=IMAGE_PATCH_SIZE,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        videos, video_metadatas = zip(*videos)  # always one video here: (frames tensor, metadata)
        videos, video_metadatas = list(videos), list(video_metadatas)
        inputs = self.processor(
            text=prompt,
            images=images,
            videos=videos,
            video_metadata=video_metadatas,
            return_tensors="pt",
            do_resize=False,
            **video_kwargs,
        ).to(self.model.device)
        generated = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
        raw = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return raw, prompt.rstrip().endswith(THINK_OPEN), len(trimmed[0])

    def predict(self, scan, volumes):
        vol01 = to01(volumes[0], scan.dataset)
        slices = pick_slices(vol01, N_SLICES)
        frames = to_rgb_images(vol01, slices, IMAGE_SIZE)
        text = slice_preface() + full_prompt(scan.dataset, scan.prior)
        raw, opened, n_tokens = self._generate(frames, text, MAX_NEW_TOKENS)
        obj, problems = answer_values(raw, cut_off=opened and n_tokens >= MAX_NEW_TOKENS)
        return obj or {}, {"prompt": text, "raw": raw, "problems": problems, "slices": slices, "think_opened": opened,
                           "n_tokens": n_tokens}

    def asker(self, scan, volumes):
        """One question per call, over the same frames (runner --task qna)."""
        vol01 = to01(volumes[0], scan.dataset)
        frames = to_rgb_images(vol01, pick_slices(vol01, N_SLICES), IMAGE_SIZE)

        def ask(question):
            raw, opened, n = self._generate(frames, slice_preface() + question, QNA_MAX_NEW_TOKENS)
            return _answer_part(raw, opened and n >= QNA_MAX_NEW_TOKENS) or "", None

        return ask


if __name__ == "__main__":
    runner.main(HealthGPTProAdapter())
