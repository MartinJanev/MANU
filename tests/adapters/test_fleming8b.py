import numpy as np
from PIL import Image

from common.datasets import Scan
from common.schema import full_prompt
from common.volume import resize, to01
from models.fleming8b.run import REASONING_PROMPT, FlemingAdapter, slice_images

ANSWER = '<think>An axial T1 MRI; no lesion is seen.</think><answer>{"modality": "MRI"}</answer>'
CARD_SLICES = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 31]  # np.linspace(0, 31, 11, dtype=int) of the card


def _scan(dataset="molab"):
    return Scan(dataset, "p1", "p1.nii.gz", ["p1.nii.gz"])


def _volume():
    return np.random.default_rng(0).random((40, 64, 64), dtype=np.float32)


def _run(monkeypatch, output, dataset="molab", volume=None):
    adapter = FlemingAdapter()
    calls = []

    def fake_generate(volume, text, max_new_tokens):
        calls.append((volume, text, max_new_tokens))
        return output

    monkeypatch.setattr(adapter, "_generate", fake_generate)
    values, details = adapter.predict(_scan(dataset), [_volume() if volume is None else volume])
    return values, details, calls


def test_npy_shape_and_range(monkeypatch):
    _, _, calls = _run(monkeypatch, ANSWER)
    volume = calls[0][0]
    assert volume.shape == (32, 256, 256)
    assert volume.dtype == np.float32
    assert volume.min() >= 0.0 and volume.max() <= 1.0


def test_volume_is_the_resized_canonical_stack(monkeypatch):
    # No axis change or flip: the canonical axial stack goes in as is (orientation: models/fleming8b/NOTES.md).
    vol = np.random.default_rng(2).normal(0.0, 400.0, (40, 64, 64)).astype(np.float32)
    for dataset in ("molab", "ct_rate"):
        _, _, calls = _run(monkeypatch, ANSWER, dataset=dataset, volume=vol)
        np.testing.assert_array_equal(calls[0][0], resize(to01(vol, dataset), (32, 256, 256)))


def test_parses_answer_tag(monkeypatch):
    raw = '<think>…</think><answer>{"modality": "MRI"}</answer>'
    values, details, _ = _run(monkeypatch, raw)
    assert values["modality"] == "MRI"
    assert details["problems"] == []
    assert details["raw"] == raw


def test_max_new_tokens_4096(monkeypatch):
    _, _, calls = _run(monkeypatch, ANSWER)
    assert calls[0][2] == 4096


def test_prompt_is_reasoning_prompt_then_11_images_then_question(monkeypatch):
    _, details, calls = _run(monkeypatch, ANSWER, dataset="ms")
    text = calls[0][1]
    # f"{prompt}\n{image_prefix}{question}" of the card's inference_3d_medical_image
    assert text == REASONING_PROMPT + "\n" + "<image>\n" * 11 + full_prompt("ms")
    assert details["prompt"] == text
    assert details["slices"] == CARD_SLICES
    # REASONING_PROMPT of the card's Quick Start, verbatim
    assert REASONING_PROMPT == (
        "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
        "The assistant first thinks about the reasoning process in the mind and then provides the user a concise "
        "final answer in a short word or phrase. The reasoning process and answer are enclosed within "
        "<think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here "
        "</think><answer> answer here </answer>"
    )


def test_unclosed_think_is_empty_with_problem(monkeypatch):
    values, details, _ = _run(monkeypatch, '<think>A draft {"modality": "CT"}, but the')
    assert values == {}
    assert details["problems"] == ["truncated_think"]


# ---------------------------------------------------------------------------
# The card's 3D loader (demo/demo.py of github.com/UbiquantAI/Fleming-VL at ec25225e), verbatim except that
# it takes an array instead of an .npy path and stops before the torchvision transform (torch is not installed
# in the test venv).
# ---------------------------------------------------------------------------
def _card_normalize_image(image):
    img_min = np.min(image)
    img_max = np.max(image)

    if img_max - img_min == 0:
        return np.zeros_like(image, dtype=np.uint8)

    return ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)


def _card_find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _card_dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = _card_find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def _card_slice_tiles(data, input_size=448, max_num=1, num_slices=11):
    indices = np.linspace(0, 31, num_slices, dtype=int)
    tiles = []
    for idx in indices:
        slice_img = data[idx]
        normalized = _card_normalize_image(slice_img)
        rgb_img = np.stack([normalized, normalized, normalized], axis=-1)
        img = Image.fromarray(rgb_img)
        img = _card_dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        assert len(img) == 1  # num_patches_list entry of this slice
        tiles.append(img[0])
    return tiles


def test_slice_images_match_card_loader():
    rng = np.random.default_rng(3)
    volume = rng.random((32, 256, 256), dtype=np.float32) * rng.random((32, 1, 1), dtype=np.float32)
    volume[6] = 0.25  # a constant sampled slice: the card makes it black
    images = slice_images(volume)
    expected = _card_slice_tiles(volume)
    assert len(images) == 11
    assert all(im.mode == "RGB" and im.size == (448, 448) for im in images)
    for ours, card in zip(images, expected):
        np.testing.assert_array_equal(np.asarray(ours), np.asarray(card))
    assert not np.asarray(slice_images(np.zeros((32, 256, 256), np.float32))[0]).any()
