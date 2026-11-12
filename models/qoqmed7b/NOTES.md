# QoQ-Med-VL-7B

## Setup

Sources read (2026-11-12):
- QoQ_Med repository https://github.com/DDVD233/QoQ_Med, cloned at commit 06a3bccf8749953cc579e87acb6d5876ef35356f (2025-07-31): `README.md`, `LICENSE`, `requirements.txt`, `examples/format_prompt/medical_format.jinja`, `examples/grpo_climb_engaging.yaml`, `examples/reward_function/medical.py`, `verl/utils/dataset.py`, `verl/trainer/data_loader.py`.
- Model card https://huggingface.co/ddvd233/QoQ-Med-VL-7B/blob/main/README.md, read through a web-fetch tool (page text; huggingface.co is blocked for git/curl on the build machine): front matter `license: mit`, `base_model: Qwen/Qwen2.5-VL-7B-Instruct`; no transformers version.
- QwenLM/Qwen2.5-VL README history (https://github.com/QwenLM/Qwen2.5-VL): the Quickstart line `pip install transformers==4.51.3 accelerate` was added in commit 6e98a0a (2025-05-13) and removed in fa5cb51 (2025-09-24), when the README became the Qwen3-VL one.
- PyPI JSON API (https://pypi.org/pypi/<package>/<version>/json) for transformers, accelerate, qwen-vl-utils, tokenizers; the qwen-vl-utils 0.0.11 wheel (`qwen_vl_utils/vision_process.py`); the transformers 4.51.3 wheel (`modeling_utils.py`).

The checkpoint loads as a plain `Qwen2_5_VLForConditionalGeneration` with `AutoProcessor` (no custom code, no `trust_remote_code`); the README says models loaded this way support only vision and text inputs (no ECG / time series).

Copied from the README: `Qwen2_5_VLForConditionalGeneration.from_pretrained(..., torch_dtype=torch.bfloat16, ...)`, `AutoProcessor.from_pretrained`, `processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)`, `process_vision_info(messages)`, `processor(text=[text], images=..., videos=..., padding=True, return_tensors="pt")`, `inputs.to("cuda")`, `model.generate(**inputs, max_new_tokens=...)`, trimming of the prompt tokens, `processor.batch_decode(..., skip_special_tokens=True, clean_up_tokenization_spaces=False)` (all in `models/_qwen25vl.py`). `torch_dtype=` is the keyword `from_pretrained` reads in transformers 4.51.3 (`modeling_utils.py`).
Verified in source: passing PIL images directly as `{"type": "image", "image": <PIL.Image>}`: qwen-vl-utils 0.0.11 `fetch_image` takes a `PIL.Image` as is (`if isinstance(image, Image.Image): image_obj = image`), converts it to RGB and rounds its size to a multiple of 28, so 448x448 stays 448x448 (256 visual tokens per slice).
Best guess: `attn_implementation="sdpa"` with bf16 and `.to("cuda")` (the README shows `device_map="auto"` and flash-attention 2); `do_sample=False, num_beams=1` passed explicitly (the checkpoint's `generation_config.json` was not read).

Input recipe:
- 4 slices: the task brief's choice. In the repo, `extract_video_frames(video_path, num_frames=4)` (`verl/utils/dataset.py`) samples 4 evenly spaced frames by default, but `RLHFDataset.__init__` defaults `video_frames=2` and `verl/trainer/data_loader.py` does not pass `video_frames`, so at this commit the training loader extracts 2 frames per video. The repo does not show how 3D scans were turned into videos or images.
- 448x448: training resized images into `min_pixels: 65536` .. `max_pixels: 524288` (`examples/grpo_climb_engaging.yaml`); 448x448 = 200704 pixels is inside that range (and equals the README's suggested `min_pixels = 256 * 28 * 28`; the README's `max_pixels` is not used here).
- Answer format: training appended `examples/format_prompt/medical_format.jinja` to every question (`format_prompt:` in `grpo_climb_engaging.yaml`): first a ```json block of `"bbox_2d"` boxes, then reasoning in `<think> </think>`, the final answer in `\boxed{}` (the reward `medical_compute_score` reads `\boxed{}`). Our prompt does not ask for that format, but the model may still use it, so `run.py` parses only the text after the last `</think>`, and within it the last `\boxed{...}` if there is one; a `<think>` that never closes (cut off by `max_new_tokens`) gives no values and the problem `truncated_think`. `details.parsed_from` is `boxed`, `after_think`, `whole` or null. `max_new_tokens=2048` because it reasons first; training allowed responses of up to 4000 tokens (`max_response_length`), so `truncated_think` in the dev run means 2048 is too short.

Pins (`requirements.txt`):
- transformers==4.51.3: the version the Qwen2.5-VL README documents (above); satisfies the QoQ_Med repo's `transformers>=4.51.0` (`requirements.txt`). Neither the QoQ_Med README (`pip install transformers qwen-vl-utils torch`) nor the model card names a version. PyPI: released 2025-04-14, `requires_python >=3.9`, pure-Python wheel; needs tokenizers >=0.21,<0.22 (cp39-abi3 wheels, so Python 3.12 is fine) and huggingface-hub >=0.30,<1.0.
- accelerate==1.6.0: the newest accelerate release before transformers 4.51.3 (2025-04-01; transformers asks `accelerate>=0.26.0`). PyPI: `requires_python >=3.9`, `torch>=2.0.0`, pure-Python wheel.
- qwen-vl-utils==0.0.11 (2025-04-21): the release current on PyPI during the whole time the Qwen2.5-VL README recommended transformers 4.51.3 (the next release, 0.0.13, came on 2025-09-20). PyPI: `requires_python >=3.8`; depends only on av, packaging, pillow, requests (no transformers dependency).

Licence: the repository code is MIT (top-level `LICENSE`, (c) 2025 Wei Dai); the model card's metadata tags the weights `license: mit`; both the GitHub README and the model card say the model is "intended for research purposes only" and "not suitable for clinical deployment". No separate licence file for the weights was read.

Before running: none known (the model card mentions no gating; if the download asks for a token, export `HF_TOKEN` on the login node).

## Dev run

## Eval run

## Pitfalls
