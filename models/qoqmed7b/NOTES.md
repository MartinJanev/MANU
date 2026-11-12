# QoQ-Med-VL-7B

## Setup

Sources read: the QoQ-Med repository README (https://github.com/DDVD233/QoQ_Med, via a mirror), whose usage section loads `ddvd233/QoQ-Med-VL-7B` as a plain `Qwen2_5_VLForConditionalGeneration` with `AutoProcessor` and `process_vision_info` (no custom code, no `trust_remote_code`; the vision+text path ignores the ECG input); the transformers Qwen2.5-VL documentation (https://huggingface.co/docs/transformers/en/model_doc/qwen2_5_vl). The Hugging Face model card itself could not be fetched from the build machine.

Copied from documentation: the loading and generation calls in `models/_qwen25vl.py` (shared with Lingshu-7B; see `models/lingshu7b/NOTES.md`). Best guess: `attn_implementation="sdpa"` with bf16 and `.to("cuda")` (the README shows `device_map="auto"` / flash-attention); explicit greedy arguments; passing PIL images directly. The README gave no image-count recipe: 4 slices comes from the task brief (training recipe), not from a source read here. The README suggests `min_pixels=256*28*28, max_pixels=1280*28*28` for the processor; not used (448x448 images are 256 visual tokens each at the default settings).

Pins: same as Lingshu-7B (transformers==4.57.1, accelerate==1.10.1, qwen-vl-utils==0.0.14); the README names no transformers version (`pip install transformers qwen-vl-utils torch`).

License: the model is for research use only (README: "intended for research purposes only", not for clinical deployment); the licence text itself was not found.

Before running: none known (if the download asks for a token, export `HF_TOKEN` on the login node).

## Dev run

## Eval run

## Pitfalls
