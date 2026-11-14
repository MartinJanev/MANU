# BrainGemma3D

## Setup

Sources read: the model card (https://huggingface.co/praiselab-picuslab/BrainGemma3D, via the mirror hf-cn0.pages.dev, which also showed the file tree, `model_config.json`, `language_model/config.json`, `language_model/generation_config.json`, `vision_model/config.json` and the snapshot's `braingemma3d_architecture.py`); the GitHub repository https://github.com/PRAISELab-PicusLab/BrainGemma3D at commit 97a63df9ef842b60faeacb2004a16d398a427851 (`braingemma3d_architecture.py`, the same 29 kB file as in the snapshot with the same key lines; `braingemma3d_inference.py`, `braingemma3d_training.py`, `preprocess_healthy_brains.py`, `requirements.txt`); the paper (arXiv 2602.22098, via a mirror).

The snapshot (12.2 GB) holds the code and all weights: `braingemma3d_architecture.py` (class `BrainGemma3D`, `CANONICAL_PROMPT`), `vision_model/` (a full MedSigLIP-448 `SiglipModel`), `language_model/` (a full `Gemma3ForConditionalGeneration`, fp16, tokenizer included, no adapter files, so the LoRA looks merged), `projector_vis_scale.pt`, `model_config.json` (depth 2, 32 vision tokens). The architecture loads both models from these folders with `local_files_only=True`, so `env.sh` lists only this repo: no `google/medgemma-1.5-4b-it` or `google/medsiglip-448` download, no `GIT_REPOS` clone.

Copied from the card's Quick Start: `snapshot_download(repo_id)`, `sys.path.append(local_dir)`, the `BrainGemma3D(vision_model_dir=..., language_model_dir=..., depth=2, num_vision_tokens=32, freeze_vision=True, freeze_language=True, device_map={"": 0})` call, `torch.load(projector_vis_scale.pt)` into `vision_projector` and `vis_scale`, `model.eval()`, the (1, 1, D, H, W) input, `generate_report(volume, prompt=..., max_new_tokens=..., temperature=..., top_p=...)` under `torch.no_grad()`. The architecture itself loads MedGemma in 4-bit nf4 (bitsandbytes) and prepends 32 visual tokens to the prompt embeddings.

Deviations and best guesses (nothing here was executed):
- The card imports `from medgemma3d_architecture import MedGemma3D`; the snapshot has no such file. Its module is `braingemma3d_architecture.py` with class `BrainGemma3D` (same constructor), which is what `load()` imports.
- Decoding: the card samples (`max_new_tokens=256, temperature=0.1, top_p=0.9`). We pass `max_new_tokens=512, temperature=None, top_p=None`; `generate_report` sets `do_sample = temperature is not None and temperature > 0`, so decoding is greedy, and `num_beams` stays 1 (not passed, not in `generation_config.json`). Its own defaults `min_new_tokens=10`, `repetition_penalty=1.2`, `no_repeat_ngram_size=3` stay (the paper's decoding controls).
- Prompt: `CANONICAL_PROMPT` verbatim, `"Generate a radiology report for this brain MRI FLAIR scan. \n"`; the trailing space and newline are part of the training prompt.
- Depth: the card's Quick Start uses `target_size=(32, 128, 128)`; the paper, the training script (`--target-size` default `64 128 128`) and the healthy-control preprocessing use 64 slices. The published code does not require 64 (depth stride 1, depth positions rebuilt from the number of depth patches, adaptive pooling to 32 tokens), so we use 32; to try 64, change `INPUT_SHAPE`.
- Orientation: `load_nifti_volume` (BraTS branch) feeds `np.transpose(RAS array, (2, 1, 0))` = (inferior→superior, posterior→anterior, left→right). Our `load_canonical` stack is (inferior→superior, anterior→posterior, right→left), so `to_model_axes` reverses H and W (`vol[:, ::-1, ::-1]`, a 180° in-plane rotation). `test_volume_is_in_model_loader_axes` checks this against the loader's own lines on an LPS NIfTI.
- Intensities: `to01` clips at percentiles 0.5/99.5 and `resize` is scipy order-1 zoom; the model's loader clips at 1/99 and resizes with `F.interpolate(mode="trilinear", align_corners=False)`. The card's "z-score" is not in the code (min-max only).
- Training data: BraTS 2020 FLAIR (skull-stripped) plus 99 healthy controls that went through a different preprocessing path (`preprocess_healthy_brains.py`: ANTs skull stripping, LPS, square padding, brain-voxel percentiles). Our IXI/MOLAB/open_ms_data T1 inputs are not skull-stripped.
- The package restores only the projector and `vis_scale`; the 3D patch embedding is re-inflated from `vision_model/` at every load (as in the card), although phase 1 of the training script trains it.

Pins (`requirements.txt`), all from the GitHub repo's `requirements.txt` (a full environment freeze), keeping only what the architecture module and `load()` import: transformers==4.57.3 (also the `transformers_version` in the snapshot's configs), accelerate==1.12.0 (`device_map`), bitsandbytes==0.48.2 (4-bit; needs torch >= 2.3), huggingface-hub==0.36.0 (`snapshot_download`), nibabel==5.3.3 (imported by the architecture module). The card says "PyTorch 2.0, Transformers 4.40+"; the repo froze torch 2.9.0, the image keeps its 2.6.0a0.

Before running: nothing known. The card states CC-BY-4.0 and shows no access gate, but the language model is a MedGemma derivative (HAI-DEF terms): if the download is refused, accept the terms on the model page and export `HF_TOKEN` on the login node. The card asks for >= 16 GB VRAM.

Extra run on the training sequence: `sbatch -J braingemma3d jobs/run_model.sh braingemma3d --split eval --datasets ms --ms-flair --controls` → `results/braingemma3d-flair/eval/`.

## Dev run

## Eval run

## Pitfalls

- Job 2555 (dev, 2026-11-14) stopped in `load()`: transformers 4.57.3 imports `TransformGetItemToIndex` from `torch._dynamo._trace_wrapped_higher_order_op`, which the image's 2.6.0a0 lacks (it has the class in `torch._higher_order_ops.flex_attention`). `common/torch_compat.py` aliases it.
- The prompt names FLAIR, so `specialized_sequence` = FLAIR in a report is an echo of the prompt, not a reading of the image.
