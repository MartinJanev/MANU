# Lingshu-7B

## Setup

Sources read (2026-11-12):
- Model card https://huggingface.co/lingshu-medical-mllm/Lingshu-7B (main; latest repo commit b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9 when read), read through a web-fetch tool (page text; huggingface.co is blocked for git/curl on the build machine): licence MIT; no gating text; section "Usage": "Using transformers (version 4.52.1 is recommended)"; the transformers quick start loads `Qwen2_5_VLForConditionalGeneration.from_pretrained("lingshu-medical-mllm/Lingshu-7B", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="auto")` and `AutoProcessor`, builds `{"type": "image", ...}` then `{"type": "text", ...}` content, calls `apply_chat_template(..., add_generation_prompt=True)`, `process_vision_info(messages)`, the processor, `model.generate(**inputs, max_new_tokens=128)`, trims the prompt tokens and decodes with `batch_decode`.
- Hugging Face Space https://huggingface.co/spaces/warshanks/Lingshu-7B (`app.py`, same tool): the same loading calls (bf16, `device_map="auto"`, flash-attention commented out).
- QoQ-Med README (https://github.com/DDVD233/QoQ_Med, cloned at 06a3bcc): the same Qwen2.5-VL pattern verbatim, including the `batch_decode` arguments.
- PyPI JSON API (https://pypi.org/pypi/<package>/<version>/json) for transformers, accelerate, qwen-vl-utils, tokenizers; the qwen-vl-utils 0.0.11 wheel (`qwen_vl_utils/vision_process.py`); the transformers 4.52.1 wheel (`modeling_utils.py`, `models/qwen2_5_vl/modeling_qwen2_5_vl.py`).

Copied from documentation: `Qwen2_5_VLForConditionalGeneration.from_pretrained(..., torch_dtype=torch.bfloat16, ...)`, `AutoProcessor.from_pretrained`, `processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)`, `process_vision_info(messages)`, `processor(text=[text], images=..., videos=..., padding=True, return_tensors="pt")`, `model.generate(**inputs, max_new_tokens=...)`, trimming of the prompt tokens, `processor.batch_decode(..., skip_special_tokens=True, clean_up_tokenization_spaces=False)` (all in `models/_qwen25vl.py`). `torch_dtype=` is the keyword `from_pretrained` reads in transformers 4.52.1 (`modeling_utils.py`).
Verified in source: passing PIL images directly as `{"type": "image", "image": <PIL.Image>}`: qwen-vl-utils 0.0.11 `fetch_image` takes a `PIL.Image` as is (`if isinstance(image, Image.Image): image_obj = image`), converts it to RGB and rounds its size to a multiple of 28, so 448x448 stays 448x448 (256 visual tokens per slice).
Best guess: `attn_implementation="sdpa"` with bf16 and `.to("cuda")` (the card shows flash-attention 2 and `device_map="auto"`); `do_sample=False, num_beams=1` passed explicitly (the checkpoint's `generation_config.json` was not read).

Pins (`requirements.txt`):
- transformers==4.52.1: the version the model card recommends. PyPI: released 2025-05-20 (4.52.0 is yanked, 4.52.1 is not), `requires_python >=3.9`, pure-Python wheel; needs tokenizers >=0.21,<0.22 (cp39-abi3 wheels, so Python 3.12 is fine) and huggingface-hub >=0.30,<1.0. (QoQ-Med-VL-7B pins 4.51.3, the Qwen2.5-VL README's version, because its own sources name none.)
- accelerate==1.7.0: the newest accelerate release before transformers 4.52.1 (2025-05-15; transformers asks `accelerate>=0.26.0`). PyPI: `requires_python >=3.9`, `torch>=2.0.0`, pure-Python wheel.
- qwen-vl-utils==0.0.11 (2025-04-21): the release current on PyPI when transformers 4.52.1 came out (the next release, 0.0.13, came on 2025-09-20). PyPI: `requires_python >=3.8`; depends only on av, packaging, pillow, requests (no transformers dependency).

Before running: none (MIT, the model card shows no gating; if the download asks for a token, export `HF_TOKEN` on the login node).

## Dev run

## Eval run

## Pitfalls

- Lingshu-32B scored higher on a black image than on real MRI (the colleagues' model notes): compare blank agreement of this model in the dev/eval report.
