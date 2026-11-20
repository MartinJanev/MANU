# Qwen3-VL-8B-Instruct

## Setup

Sources read (2026-11-20):
- README of https://github.com/QwenLM/Qwen3-VL (cloned at 96588727e44c78b25ba03ea03b8e12f7e64fd0da):
  "The Qwen3-VL model requires transformers >= 4.57.0"; `AutoModelForImageTextToText` + `AutoProcessor`; video
  example: `process_vision_info(messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)`,
  split `(video, metadata)` pairs, `processor(text=..., images=..., videos=..., video_metadata=..., return_tensors="pt",
  do_resize=False, **video_kwargs)`, `generate`, trim the prompt tokens, `batch_decode(skip_special_tokens=True,
  clean_up_tokenization_spaces=False)`; `pip install qwen-vl-utils==0.0.14`.
- qwen-vl-utils 0.0.14 (`src/qwen_vl_utils/vision_process.py` of the same clone): a list of PIL frames is padded to
  an even count (FRAME_FACTOR 2), `sample_fps` defaults to 2.0; 448x448 frames stay 448x448 (between
  VIDEO_MIN_TOKEN_NUM 128 and VIDEO_MAX_TOKEN_NUM 768 tokens of 32x32 pixels).

Copied from documentation: the README's video inference calls (`models/qwen3vl8b/run.py`, `_generate`).
Same as HealthGPT-Pro-27B (fair comparison): `pick_slices`, 10 frames, 448x448, `"sample_fps": 2.0`,
`slice_preface(10)` + the full-JSON prompt.
Choices: `dtype=torch.bfloat16`, `attn_implementation="sdpa"`, `device_map="cuda"` (README: `dtype="auto"`,
`device_map="auto"`, flash-attention 2 recommended but not installed); greedy `do_sample=False, num_beams=1`,
max_new_tokens 1024 (an Instruct model, no reasoning block). No JSON, or JSON without one valid schema value:
`PredictionError` -> status "error" with the reason; the raw text stays in `*_details.jsonl`.

Pins (`requirements.txt`): transformers==4.57.6 (>= 4.57.0 as the README asks; the 4.57 release hlip already runs
with in containers/ngc2412.sif, torch_compat shim applies), accelerate==1.12.0, huggingface-hub==0.36.2 (hlip's),
qwen-vl-utils==0.0.14 (README).

Before running: none (Apache-2.0, not gated). `nohup bash jobs/setup_env.sh qwen3vl8b > setup_qwen3vl8b.log 2>&1 &`

## Dev run

Smoke test (1 IXI, 1 MS): `sbatch -J qwen3vl8b jobs/run_model.sh qwen3vl8b --split dev --datasets ixi,ms --controls`

## Eval run

`sbatch -J qwen3vl8b jobs/run_model.sh qwen3vl8b --split eval --controls`

## Pitfalls

- The base model of several medical models in the benchmark (ClinFusion-8B, Qwen3-VL-8B-SGMRIVQA): compare their
  blank agreement and diagnoses with this one.
