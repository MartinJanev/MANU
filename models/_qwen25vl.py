"""Shared loading and generation for the Qwen2.5-VL based medical models (Lingshu-7B, QoQ-Med-VL-7B).

Both checkpoints load as a plain Qwen2_5_VLForConditionalGeneration, so the calls follow the quick starts of the
QoQ-Med README and the Lingshu-7B model card (sources and deviations: models/<m>/NOTES.md). torch / transformers are
imported inside the functions, so a model's run.py imports on a CPU-only machine.
"""


def load_qwen25vl(model_id):
    """(model, processor): bf16, SDPA attention, on cuda."""
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to("cuda")
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def generate_images(model, processor, images, text, max_new_tokens):
    """One user message (the images, then the text), chat template, greedy decoding; returns the answer text."""
    import torch
    from qwen_vl_utils import process_vision_info

    messages = [{
        "role": "user",
        "content": [{"type": "image", "image": image} for image in images] + [{"type": "text", "text": text}],
    }]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[prompt], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to("cuda")
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
