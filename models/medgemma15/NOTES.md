# MedGemma 1.5 4B-it

## Setup

Sources read (2026-11-20):
- Notebook `notebooks/high_dimensional_ct_hugging_face.ipynb` of https://github.com/Google-Health/medgemma
  (cloned at a60a66024f6153496dfa9490dba12d4f1fbd092e): 85-slice sampling
  `[dicom_instances[int(round(i /(MAX_SLICE) * (len(dicom_instances)-1)))] for i in range(1, MAX_SLICE + 1)]`
  when there are more than 85; `norm()` / `window()` with `window_clips = [(-1024, 1024), (-135, 215), (0, 80)]`,
  `np.round(...).astype(np.uint8)`; `_encode()` (JPEG data URI); content = instruction text, then
  `{"type": "image"}` + `{"type": "text", "text": f"SLICE {n}"}` per slice, then the query text, in one `user`
  message; `AutoProcessor.from_pretrained(model_id, use_fast=True)`, `AutoModelForImageTextToText` with
  `dtype=torch.bfloat16, device_map="auto", offload_buffers=True`; `apply_chat_template(messages,
  add_generation_prompt=True, continue_final_message=False, return_tensors="pt", tokenize=True, return_dict=True)`,
  `inputs.to(model.device, dtype=torch.bfloat16)`, `generate(do_sample=False, max_new_tokens=2000)`.
- MedGemma 1.5 technical report, arXiv 2604.05081v2, CT/MRI section: at most 85 axial slices, sampled
  equidistantly along z; MRI "no windowing", min-max normalization per volume, R, G, B set to the same value;
  CT windows R (-1024, 1024), G (-135, 215), B (0, 80); slices interleaved with "SLICE {index}"; each slice rescaled
  to 896x896. The notebook's markdown says 135 to 215 HU for green; its code and the report say -135: code followed.
- Model card (copy of the google/medgemma-1.5-4b-it card; huggingface.co is not reachable from the build machine):
  "Gemma 3 is supported starting from transformers 4.50.0"; the direct example decodes only the new tokens with
  `skip_special_tokens=True`; greedy decoding is the checkpoint's default since 2026-01-23. HAI-DEF terms, gated.
- transformers 4.57.3 wheel: `image_utils.load_image` decodes `data:image/...;base64,` strings;
  `from_pretrained(dtype=...)` exists; `huggingface-hub>=0.34.0,<1.0`.

Copied from the notebook: slice sampling, CT windows and rounding, JPEG data URIs, message layout, loading,
tokenization and generate arguments (max_new_tokens 1024 instead of 2000, as for the other full-JSON models).
From the report: MRI = one min-max over the whole volume, grey in R = G = B.
Deviations / choices:
- Decoding: the model card's `processor.decode(generated[0][input_len:], skip_special_tokens=True)` instead of the
  notebook's `post_process_image_text_to_text` + stripping the echoed prompt (same text).
- Prompt: the colleagues' full-JSON prompt of `common/prompts/` (as every other model). The text before
  "Output ONLY this JSON schema below" (role and task lines, i.e. the system text at the start of the user text)
  plus `slice_preface()` is the instruction; the rest (the schema) comes after the slices.
- Slice order: our canonical stack, inferior -> superior (the notebook orders by DICOM InstanceNumber).
- Blank control (zeros): MRI min-max has no range, so the slices are black; CT zeros are 0 HU through the windows.
- `diagnosis_name` goes through `evaluation.evaluate.norm_diag` ("Normal" -> normal, "Brain tumor" -> tumor) before
  the common normaliser; the model's own spelling is kept as `diagnosis_name_raw` in the details.
- No JSON, or JSON without one valid schema value: `PredictionError` -> status "error" with the reason; the raw
  text stays in `*_details.jsonl`.

Pins (`requirements.txt`): transformers==4.57.3, accelerate==1.12.0, huggingface-hub==0.36.0, the versions
BrainGemma3D (built on MedGemma 1.5) already runs with in containers/ngc2412.sif (torch_compat shim applies).

Before running: export `HF_TOKEN` (an account that accepted the HAI-DEF terms) on the login node, then
`nohup bash jobs/setup_env.sh medgemma15 > setup_medgemma15.log 2>&1 &`.

## Dev run

- Jobs 2908 / 2914 / 2920 (2026-11-23): full prompt and prior parse; in Q&A it explains its choice in prose ("... is **C. T2-weighted**"), which parse_choice now reads (88 -> 121 of 126 dev answers).


Smoke test (1 IXI, 1 MS): `sbatch -J medgemma15 jobs/run_model.sh medgemma15 --split dev --datasets ixi,ms --controls`

- Job 2759 (2026-11-20, dev): IXI answered in JSON; the 4 MOLAB scans, the MS scan and the CT-RATE scan got a
  FINDINGS/IMPRESSION report instead ("... IMPRESSION: Normal brain."), so 6 of 11 rows were `no_json` errors. Every
  blank control answered in JSON. Such reports are now read with `common.parse.extract_from_report`, the extractor
  the report models share (`details.parsed_from`: `json` or `report`); the dev rows were re-scored with
  `python3 -I tools/reparse.py medgemma-1.5-4b dev`.

## Eval run

`sbatch -J medgemma15 jobs/run_model.sh medgemma15 --split eval --controls`

## Pitfalls

- 85 slices x 256 image tokens = 21,760 tokens per prompt; a volume with 85 slices or fewer sends all of them.
