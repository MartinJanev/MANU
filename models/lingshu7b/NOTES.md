# Lingshu-7B

## Setup

Sources read: transformers Qwen2.5-VL documentation (https://huggingface.co/docs/transformers/en/model_doc/qwen2_5_vl); a Hugging Face Space running this checkpoint (https://huggingface.co/spaces/warshanks/Lingshu-7B, via a mirror; it loads `Qwen2_5_VLForConditionalGeneration` and `AutoProcessor` from `lingshu-medical-mllm/Lingshu-7B` and uses `process_vision_info`); the QoQ-Med repository README for the shared loading pattern. The Lingshu-7B model card itself (huggingface.co) could not be fetched from the build machine, so the model card's quick start and its stated transformers minimum were NOT read directly.

Copied from documentation: `Qwen2_5_VLForConditionalGeneration.from_pretrained`, `AutoProcessor.from_pretrained`, `processor.apply_chat_template(..., add_generation_prompt=True)`, `process_vision_info`, `model.generate`, trimming of the prompt tokens, `processor.batch_decode` (all in `models/_qwen25vl.py`).
Best guess: `attn_implementation="sdpa"` with `torch_dtype=torch.bfloat16` then `.to("cuda")` (the examples use `device_map="auto"` / flash-attention); passing PIL images directly as `{"type": "image", "image": <PIL>}`; `do_sample=False, num_beams=1` passed explicitly to `generate`.

Pins (`requirements.txt`): transformers==4.57.1 (newest 4.x release confirmed on PyPI that has Qwen2.5-VL, which needs >= 4.49; 5.x deliberately avoided: it requires Python >= 3.11 APIs and torch >= 2.6 final while the image has torch 2.6.0a0), accelerate==1.10.1 (a release confirmed on PyPI that supports transformers 4.57), qwen-vl-utils==0.0.14 (newest on PyPI). Martin: if the model card names a different transformers version, change the pin.

Before running: none (MIT-licensed, not gated as far as seen; if the download asks for a token, export `HF_TOKEN` on the login node).

## Dev run

## Eval run

## Pitfalls

- Lingshu-32B scored higher on a black image than on real MRI (the colleagues' model notes): compare blank agreement of this model in the dev/eval report.
