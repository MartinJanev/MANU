# Fleming-VL-8B

## Setup

Sources read (2026-11-13):
- Model card https://huggingface.co/IQuestLab/Fleming-VL-8B/blob/main/README.md, through a web-fetch tool that returns a summary of the page, not its text (huggingface.co is blocked for git/curl on the build machine), plus the same card under `UbiquantAI/Fleming-VL-8B` through a mirror (hg.istephan-squireso.workers.dev): front matter `license: apache-2.0`, `library_name: transformers`, `pipeline_tag: image-text-to-text`; install `pip install -U transformers accelerate` (no versions); the Quick Start is the script of the GitHub README below (its code loads `MODEL_PATH = "UbiquantAI/Fleming-VL-8B"`, its vLLM/SGLang lines `IQuestLab/Fleming-VL-8B`); 3D example: (32, 256, 256) or (1, 32, 256, 256) `.npy`, 11 slices, one tile per slice, `do_sample=False`, `max_new_tokens=1024`.
- GitHub repository https://github.com/UbiquantAI/Fleming-VL, cloned at commit ec25225e7bcd2ce8220d3d41643b8a8ba081d25b (2025-11-06): `README.md` (Quick Start = `demo/demo.py`), `demo/demo.py`, `demo/resource/test.npy`, `requirements.txt` (a full environment freeze), `INSTALLATION.md` (InternVL's generic install notes), `LICENSE` (Apache 2.0).
- InternVL `chat()`: https://github.com/OpenGVLab/InternVL at commit 2410d1dbf208f0e799459aff9376e5747dbf41a2, `internvl_chat/internvl/model/internvl_chat/modeling_internvl_chat.py` and `modeling_intern_vit.py` (how `<image>` placeholders and `num_patches_list` expand into image tokens, eos = template separator, `batch_decode(..., skip_special_tokens=True)` then split at the separator; `use_flash_attn=False` gives naive vision attention and `attn_implementation='eager'` in the LLM).
- Paper arXiv 2511.00916 (HTML v1, web fetch): names M3D (M3D-VQA) as its 3D CT data; "Each 3D volume is converted into a sequence of 2D RGB images through evenly-spaced sampling along the depth dimension."
- M3D repository https://github.com/BAAI-DCAI/M3D at commit 63e72005a2304eda74579870239bfc7b04b0edae: `Data/process/m3d_cap_data_prepare.py` (how M3D's (1, 32, 256, 256) arrays are built).
- https://huggingface.co/IQuestLab/Fleming-VL-38B/blob/main/generation_config.json (web fetch): `{"_from_model_config": true, "transformers_version": "4.49.0"}`. The 8B repo's `config.json`, `generation_config.json`, tokenizer and remote-code files could not be read.
- PyPI JSON API (https://pypi.org/pypi/<package>/<version>/json) for transformers 4.49.0, accelerate 1.7.0, timm 1.0.15, einops 0.8.1.

Copied from the card (`demo/demo.py`): `load()` = `load_model` (`AutoModel.from_pretrained(..., torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, use_flash_attn=False, trust_remote_code=True).eval().cuda()`, `AutoTokenizer.from_pretrained(..., trust_remote_code=True, use_fast=False)`; `use_flash_attn=False` as the 3D example passes it); `REASONING_PROMPT` verbatim; `build_transform(448)` (RGB, bicubic resize to 448x448, `ToTensor`, ImageNet mean/std); `convert_npy_to_images`: shape (32, 256, 256), slices `np.linspace(0, 31, 11, dtype=int)` = 0, 3, 6, ..., 27, 31, per-slice `normalize_image` (min-max to uint8, truncated; a constant slice becomes black), grey repeated into RGB, `dynamic_preprocess(max_num=1, use_thumbnail=True)` = one 448x448 tile per slice; the question `f"{prompt}\n{image_prefix}{question}"` with one `"<image>\n"` per slice; `model.chat(tokenizer, pixel_values.to(torch.bfloat16).cuda(), question, generation_config, num_patches_list=..., history=None, return_history=True)`.

Deviations:
- The volume comes from memory (no `np.load`). `slice_images` writes `dynamic_preprocess` with `max_num=1` as its only possible result, a resize to 448x448 with Pillow's default filter (bicubic); `test_slice_images_match_card_loader` checks it against the card's functions verbatim. A scratch check (not in the repo) with CPU torch 2.6.0 / torchvision 0.21.0 ran `_generate` with a stand-in model: its `pixel_values` equal the card's `convert_npy_to_images` output (max abs difference 0, bf16) on random, per-slice-scaled and all-zero volumes and on the card's `test.npy`; `chat()` received `num_patches_list=[1]*11`, `history=None`, `return_history=True`.
- `max_new_tokens=2048` (global constraints: it thinks first) instead of the 3D example's 1024; `num_beams=1` passed explicitly (the card passes only `do_sample=False`).
- The question is `full_prompt(dataset)`, without a slice preface (the card's 3D path adds none). The card's `REASONING_PROMPT` asks for "a concise final answer in a short word or phrase", which pulls against our JSON request: the model may put a phrase in `<answer>` (problem `no_json`).
- Parsing: `parse_json_answer` drops closed `<think>` blocks and reads `<answer>...</answer>`. A `<think>` that never closes (the 2048 tokens ran out while reasoning) gives no values and the problem `truncated_think`, as in qoqmed7b.

Best guesses (not verifiable here):
- The 8B repo ships InternVL3's remote code (`config.json` `auto_map` to its own `modeling_internvl_chat.py`, whose `chat()` matches the InternVL GitHub version read above). If `auto_map` names another repo (e.g. `OpenGVLab/InternVL3-8B`), the offline job fails at `load()` and that repo must be added to `HF_REPOS`.
- `<think>`/`<answer>` survive decoding: `chat()` decodes with `skip_special_tokens=True`; if the tokenizer marks these tags as special tokens they vanish and `parse_json_answer` takes the first JSON object of the whole text.
- The tokenizer is InternVL3's Qwen2 BPE tokenizer, which needs no sentencepiece (the repo's freeze has `sentencepiece==0.2.0` if `use_fast=False` asks for it).

Orientation:
- The card's loader treats axis 0 as depth (`data[idx]` for the 11 indices) and shows each slice with rows = axis 1, columns = axis 2. The card and the paper state no anatomical orientation.
- The card's example volume `demo/resource/test.npy` ((1, 32, 256, 256) float32 in [0.10, 0.65], a chest CT with a pacemaker) is not an axial stack. Read off rendered slices and reformats: axis 0 runs across the patient, patient right → left (sagittal slices; the middle ones show the spine lengthwise; liver at low indices, heart and pacemaker at high ones), axis 1 anterior → posterior (table at the bottom), axis 2 inferior → superior (abdomen at low indices, trachea at high ones): a CT voxel array in LPS (x, y, z) order, resized, not reoriented.
- M3D, the paper's named 3D data, stacks Radiopaedia display slices in file order (`m3d_cap_data_prepare.py`): depth is the series' own plane (axial, coronal or sagittal) and order, in-plane as displayed. For axial series that is the radiological display, rows anterior → posterior and columns patient right → left, the same in-plane layout as our stack.
- So `predict` passes our canonical axial stack (D feet → head, H front → back, W patient right → left, `common.volume.load_canonical`) resized to (32, 256, 256), with no flip or axis change: no source fixes a depth direction, M3D's axial in-plane layout matches ours, and the global constraints require the axial stack. The layout of the card's example would be `vol.transpose(2, 1, 0)` (sagittal slices); it is not used. `test_volume_is_the_resized_canonical_stack` pins this.

Pins (`requirements.txt`):
- transformers==4.49.0: the GitHub repo's `requirements.txt` and the `transformers_version` of Fleming-VL-38B's `generation_config.json`; the card itself says only `pip install -U transformers accelerate`. PyPI: released 2025-02-17, `requires_python >=3.9`, pure-Python wheel; needs tokenizers >=0.21,<0.22 and huggingface-hub >=0.26,<1.0.
- accelerate==1.7.0: the repo's `requirements.txt`; `low_cpu_mem_usage=True` needs accelerate. PyPI: `requires_python >=3.9`, `torch>=2.0.0`.
- timm==1.0.15, einops==0.8.1: the repo's `requirements.txt`; InternVL's vision tower imports `timm.models.layers.DropPath` and `einops.rearrange` (`modeling_intern_vit.py`), and transformers refuses remote code whose top-level imports are missing.
- Not listed: flash-attn (the 3D example loads with `use_flash_attn=False`), decord (video only), torchvision (the image's).
- `uv pip compile` for Python 3.12 with `torch==2.6.0`, `torchvision==0.21.0` as constraints resolves these pins (tokenizers 0.21.4, huggingface-hub 0.36.2).

Licence: Apache-2.0 (card front matter; `LICENSE` of the GitHub repo). The card and README say research and non-clinical use only.

Before running: nothing known (the card shows no access gate; if the download asks for a token, export `HF_TOKEN` on the login node).

## Dev run

## Eval run

## Pitfalls
