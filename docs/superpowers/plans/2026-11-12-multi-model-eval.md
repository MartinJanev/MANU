# MANU: eight-model evaluation pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run BrainGemma3D, Lingshu-7B, QoQ-Med-VL-7B, Fleming-VL-8B, HLIP, MedM-VL-CT-3B-en, Med-2E3 and HealthGPT-Pro-27B through one shared pipeline on fixed dev/eval patients from IXI, MOLAB, open_ms_data and CT-RATE, producing records in the colleagues' JSONL format, one evaluation table and a list of pitfalls per model.

**Architecture:** A shared package `common/` owns everything that must be identical across models: which scans belong to a split, how a NIfTI becomes a canonical axial array, prompts, parsing, the output record and the run loop (resume, error isolation, blank-image control). Each model is one `models/<name>/run.py` that implements a small `Adapter` (`load` + `predict`) and calls `common.runner.main`. One generic SLURM script runs any model in a Singularity image with its own package folder; a migration script replicates the new layout on the cluster.

**Tech Stack:** Python 3.10+ (`radfm.sif`) / 3.12 (NGC 24.12), numpy, nibabel, scipy, Pillow, pytest; PyTorch + transformers inside Singularity on the FINKI GPGPU cluster (SLURM, one A100 80 GB per job).

**Spec:** No separate spec doc; the requirements are listed below (Martin, 2026-11-12). Model facts (input sizes, prompt formats, version pins) come from `../cld/Medical VLLMs - Prompts and Q&A.md` ("the model notes"); data facts from the project doc `claude/Проверка на податоците (IXI, MOLAB, open_ms_data).md`. Executors read both where a task points to them.

## Requirements

- **R1** Test the 8 models on IXI, MOLAB and open_ms_data, and on a few CT-RATE chest CTs (downloaded on the cluster).
- **R2** A separate script per model, with the same process for all (RadFM is the precedent).
- **R3** Dev split (developing and checking the method): IXI002, 010005N, patient05.
- **R4** Eval split (results): IXI012, IXI013, IXI014, 010020N, 010024N, 010029N, patient09, patient10, patient11.
- **R5** Reorganize the folder.
- **R6** A script that replicates the new layout on the cluster.
- **R7** Report results and pitfalls per model.

## Global Constraints

- Repo root = local `~/МАНУ/MANU` = cluster `/home/hpc/users/martin.janev/manu` (login node `10.10.65.6`, EduVPN). All paths in this plan are relative to it.
- Splits live only in `splits.json`. A MOLAB "patient" means **all** of its scans (dev: 4; eval: 3 + 9 + 7 = 19). Expected records per model: dev 7 (1 IXI + 4 MOLAB + 1 MS + 1 CT-RATE), eval 28 (3 + 19 + 3 + 3).
- The model input is the T1 volume for every dataset: `data/ixi/T1/*.nii.gz`, `data/molab/nifti/*_img.nii*`, `data/open_ms_data/cross_sectional/raw/patientNN/T1W.nii.gz`; CT-RATE: `valid_N_a_1.nii.gz` (first scan, first reconstruction). FLAIR only in the explicitly labelled variant `--ms-flair` (results under `<model>-flair`).
- Every volume passes through `common.volume.load_canonical` (RAS via `nib.as_closest_canonical`, axial stack). IXI is acquired sagittally; no model may receive a non-canonical array. Intensities are scaled to [0, 1] only by `common.volume.to01(vol, dataset)`: CT window [-1000, 1000] HU for CT-RATE, percentile scaling for MRI.
- Output = the colleagues' format (`reference/radfm_*_processed.jsonl`), built only by `common.schema.build_record`. `input.system_prompt_type` ∈ `full_json_v1`, `qna_per_field_v1`, `findings_then_extract_v1`, `clip_or_heads_v1`. Ground truth goes only into `input.metadata`; no `parsed_response` value may come from metadata or be a constant.
- Decoding: greedy (`do_sample=False`, `num_beams=1`), seed 0. `max_new_tokens`: 1024 full_json (2048 for Fleming-VL, which thinks first), 512 reports, 16 MCQ.
- Results: `results/<model>/<split>/<model>_<tag>_{processed,details,controls}.jsonl` + `run_info.json`; `<tag>` ∈ `healthy`, `brain_metastases`, `ms`, `ct_rate`. `results/` is never committed.
- Containers: RadFM keeps `containers/radfm.sif`. The 8 new models use `containers/ngc2412.sif` = `docker://nvcr.io/nvidia/pytorch:24.12-py3` (CUDA 12.6.3, needs driver ≥560; the cluster has 560.35.05; torch 2.6.0a0, Python 3.12). Per-model packages go to `pyuser/<model>` via `PYTHONUSERBASE`, installed with torch/torchvision/torchaudio pinned to the image's versions (never reinstall torch). Each `requirements.txt` uses exact `==` pins, at the versions the model card / repo names.
- Jobs: partition `openlab-queue`, `--gres=gpu:1`, 16 CPUs, 96G, ≤ 1 day; `HF_HUB_OFFLINE=1` inside jobs; weights and repos are fetched on the login node by `jobs/setup_env.sh`.
- `common/` imports with only numpy, nibabel, scipy and Pillow (no torch at import time) and runs on Python 3.10.
- Tests run on CPU without weights: `.venv/bin/python -m pytest tests -q`. The only GPU-touching code in an adapter is one method (`_generate` or `_score`), which tests replace.
- Code and comments in English, in the style of `models/radfm/run.py`. RadFM itself is not ported to `common/`; its full-run results are compared by filtering them to the split (Task 9).

## Review Focus

1. **The model ignores the image** and answers the same for a blank volume as for the real scan. → Task 6 writes one blank control per dataset (`test_controls_written_once`); Task 9 reports blank agreement per model (`test_blank_agreement`).
2. **Orientation as a shortcut:** IXI (sagittal acquisition) must give the same axial layout as the axially acquired datasets, or a model can tell healthy from sick by orientation. → Task 3 `test_sagittal_affine_gives_axial_stack`.
3. **Schema echo:** the model copies `"<MRI|CT|null>"` or the whole option list instead of answering. → Task 4 `test_echoed_placeholder_becomes_null`; Task 4 `test_parse_choice` (echo case); Task 5 `test_echo_of_all_options_is_none`.
4. **One bad scan or an OOM kills the job; a restarted job duplicates records.** → Task 6 `test_error_scan_is_recorded_and_run_continues`, `test_resume_skips_done`.
5. **A split ID with no matching file is silently skipped**, so a model is evaluated on fewer patients than intended. → Task 2 `test_missing_patient_raises`.

---

## Target layout

```
MANU/
├── README.md  splits.json  .gitignore
├── common/            shared pipeline: datasets.py volume.py schema.py parse.py qna.py runner.py prompts/
├── models/            radfm/run.py (moved) + one folder per new model: run.py requirements.txt env.sh NOTES.md
├── jobs/              run_model.sh (generic) setup_env.sh job_radfm.sh
├── containers/        radfm.Dockerfile (+ *.sif on the cluster only)
├── data_tools/        prezemi_podatoci.sh proveri_podatoci.py prezemi_ct_rate.py
├── evaluation/        evaluate.py check_schema.py
├── tools/             layout.py migrate_cluster.sh sync_to_cluster.sh fetch_results.sh
├── reference/         colleagues' *_processed.jsonl (from ../Outputs)
├── results/           <model>/<split>/…, slurm/, archive/   (not in git)
├── archive/code/      run_radfm_old.py run_radfm_nov.py
├── tests/  docs/superpowers/plans/
└── cluster only (not in git): data/ hf_cache/ pyuser/<model>/ third_party/ containers/*.sif
```

---

### Task 1: Reorganize the local folder

**Files:**
- Create: `tools/__init__.py`, `tools/layout.py`, `tests/conftest.py` (puts the repo root on `sys.path`), `tests/test_layout.py`, `splits.json`, `README.md`, `.gitignore`, `common/__init__.py`, `evaluation/__init__.py`
- Move:

  | from | to |
  |---|---|
  | `code/run_radfm.py` | `models/radfm/run.py` |
  | `code/archive/*` | `archive/code/` |
  | `code/evaluate.py`, `code/check_schema.py` | `evaluation/` |
  | `Dockerfile` | `containers/radfm.Dockerfile` |
  | `job_radfm.sh` | `jobs/job_radfm.sh` |
  | `prezemi_podatoci.sh`, `proveri_podatoci.py` | `data_tools/` |
  | `results/old_results_02_11`, `results/old_results_04_11` | `results/archive/` |
  | `../Outputs/*.jsonl` (14 files) | `reference/` |
- Delete: `code/__pycache__/`, `__pycache__/`, then the empty `code/`
- Modify: `jobs/job_radfm.sh`, `data_tools/prezemi_podatoci.sh`, `data_tools/proveri_podatoci.py:522`, `evaluation/evaluate.py` (`main`)

**Interfaces:**
- Produces: `tools.layout.CODE_LAYOUT: list[str]`, `tools.layout.CLUSTER_LAYOUT: list[str]` (an entry ending in `/` is a directory); `tools.layout.check(root: Path, layout: list[str]) -> list[str]` (missing entries, in layout order); CLI `python3 tools/layout.py [--cluster] [ROOT]` (prints missing entries, exit 1 if any); `splits.json`.

Exact values:
- `CODE_LAYOUT = ["README.md", "splits.json", ".gitignore", "common/__init__.py", "models/radfm/run.py", "jobs/job_radfm.sh", "containers/radfm.Dockerfile", "data_tools/prezemi_podatoci.sh", "data_tools/proveri_podatoci.py", "evaluation/evaluate.py", "evaluation/check_schema.py", "reference/", "results/", "tools/layout.py", "tests/"]`
- `CLUSTER_LAYOUT = CODE_LAYOUT + ["data/ixi/T1/", "data/molab/nifti/", "data/open_ms_data/cross_sectional/raw/", "containers/radfm.sif", "hf_cache/", "pyuser/radfm/", "third_party/RadFM/Quick_demo/", "results/slurm/", "results/radfm/full/"]`
- `splits.json`: `{"dev": {"ixi": ["IXI002"], "molab": ["010005N"], "ms": ["patient05"]}, "eval": {"ixi": ["IXI012", "IXI013", "IXI014"], "molab": ["010020N", "010024N", "010029N"], "ms": ["patient09", "patient10", "patient11"]}}`
- `.gitignore`: `data/`, `hf_cache/`, `pyuser/`, `third_party/`, `containers/*.sif`, `results/`, `.idea/`, `.venv/`, `__pycache__/`, `*.pyc`, `.DS_Store`, `.manu_commit`
- `jobs/job_radfm.sh`: `--output=results/slurm/%x_%j.out`; `mkdir -p results/slurm results/radfm/full` before the run; binds `-B ./models/radfm:/code -B ./data:/data -B ./results/radfm/full:/results -B ./reference:/reference -B ./hf_cache:/hf_cache -B ./pyuser/radfm:/pyuser -B ./third_party/RadFM/Quick_demo:/radfm`; command `python3 /code/run.py "$DATASET" "${@:2}"`; freeze to `results/radfm/full/requirements_radfm.txt`.
- `data_tools/prezemi_podatoci.sh`: `BASE="${BASE:-$(cd "$(dirname "$0")/.." && pwd)}"`; the final check calls `"$BASE/data_tools/proveri_podatoci.py"`.
- `data_tools/proveri_podatoci.py:522`: `--data` default `Path(__file__).resolve().parent.parent / "data"`.
- `evaluation/evaluate.py` `main()`: `--outputs` default `str(MANU / "reference")`.
- `README.md`: four short sections: Layout (the tree above), Splits, Running a model (dev → eval, Task 7 commands), Cluster sync (Task 8 commands).

- [ ] **Step 1: Snapshot before moving.** `cd ~/МАНУ/MANU && git init`, write `.gitignore`, `git add -A && git commit -m "chore: snapshot before reorganization"`; then `python3 -m venv .venv && .venv/bin/pip install numpy nibabel scipy pillow pytest`.

- [ ] **Step 2: Write the failing test** `tests/test_layout.py`

```python
def test_repo_has_code_layout():
    assert check(REPO, CODE_LAYOUT) == []

def test_old_paths_are_gone():
    for p in ["code", "Dockerfile", "job_radfm.sh", "prezemi_podatoci.sh", "proveri_podatoci.py"]:
        assert not (REPO / p).exists()

def test_reference_has_colleague_files():
    assert len(list((REPO / "reference").glob("*_processed.jsonl"))) == 14

def test_splits_json():
    s = json.loads((REPO / "splits.json").read_text())
    assert {k: s["dev"][k] for k in ("ixi", "molab", "ms")} == {"ixi": ["IXI002"], "molab": ["010005N"], "ms": ["patient05"]}
    assert s["eval"]["molab"] == ["010020N", "010024N", "010029N"]
```

- [ ] **Step 3: Run it to see it fail.** `.venv/bin/python -m pytest tests/test_layout.py -q` → FAIL (`No module named 'tools.layout'`).
- [ ] **Step 4: Implement** `tools/layout.py`, make the moves and edits above, write `splits.json` and `README.md`.
- [ ] **Step 5: Verify.** `.venv/bin/python -m pytest tests -q` → PASS; `python3 -I evaluation/evaluate.py --selftest` → `selftest OK`; `bash -n jobs/job_radfm.sh data_tools/prezemi_podatoci.sh` → no output.
- [ ] **Step 6: Commit.** `git add -A && git commit -m "refactor: reorganize MANU into common/models/jobs/evaluation layout"`

---

### Task 2: Scans of a split — `common/datasets.py`

**Files:** Create `common/datasets.py`, `tests/test_datasets.py`

**Interfaces:**
- Produces:
  - `DATASETS: dict[str, dict]`: `ixi` → `{"glob": "ixi/T1/*.nii.gz", "tag": "healthy", "experiment": "healthy"}`; `molab` → `{"glob": "molab/nifti/*_img.nii*", "tag": "brain_metastases", "experiment": "brain_metastases"}`; `ms` → `{"glob": "open_ms_data/cross_sectional/raw/patient*/T1W.nii.gz", "tag": "ms", "experiment": "multiple_sclerosis"}`
  - `@dataclass(frozen=True) class Scan: dataset: str; patient: str; image_path: str; files: tuple[str, ...]` (`image_path` in the colleagues' spelling; `files` are absolute paths fed to the model)
  - `load_splits(path: str) -> dict`
  - `patient_id(dataset: str, image_path: str) -> str`
  - `molab_name(name: str) -> str` (copy of `models/radfm/run.py`)
  - `find_scans(data_dir: str, dataset: str, split: str, splits: dict, ms_flair: bool = False) -> list[Scan]` — `split="full"` means every scan; sorted by `image_path`; raises `ValueError` naming every split patient with no file. MS: `image_path = "patientNN_T1.nii.gz"`, `files = (…/T1W.nii.gz,)` or `(…/FLAIR.nii.gz,)` with `ms_flair`.
  - `ground_truth(scan: Scan, reference_dir: str, data_dir: str) -> dict` — metadata from `reference/radfm_<tag>_processed.jsonl` via the moved `load_reference` / `metadata_for` logic of `models/radfm/run.py` (cached per dataset). Task 10 adds `ct_rate`.

- [ ] **Step 1: Write the failing tests** (a fake tree of empty files under `tmp_path`: `ixi/T1/IXI002-Guys-0828-T1.nii.gz`, `ixi/T1/IXI012-HH-1211-T1.nii.gz`, `molab/nifti/10005_19010618_eTRA-3D-T1Gd_img.nii`, `molab/nifti/10005_19011011_eTRA-3D-T1Gd_img.nii`, `molab/nifti/10020_19010822_TRA_3D_T1+GD_img.nii`, `open_ms_data/cross_sectional/raw/patient05/{T1W,FLAIR}.nii.gz`, `…/patient09/T1W.nii.gz`)

```python
def test_dev_split_resolves_expected_scans(tree):
    dev = {"dev": {"ixi": ["IXI002"], "molab": ["010005N"], "ms": ["patient05"]}}
    assert [s.image_path for s in find_scans(tree, "molab", "dev", dev)] == [
        "010005N__010005_19010618_eTRA-3D-T1Gd_img.nii", "010005N__010005_19011011_eTRA-3D-T1Gd_img.nii"]
    (ms,) = find_scans(tree, "ms", "dev", dev)
    assert (ms.patient, ms.image_path) == ("patient05", "patient05_T1.nii.gz")
    assert ms.files[0].endswith("patient05/T1W.nii.gz")

def test_ms_flair_variant(tree): ...   # files[0] ends with "patient05/FLAIR.nii.gz", image_path unchanged
def test_full_split_returns_all(tree): ...   # len(find_scans(tree, "ixi", "full", {})) == 2
def test_missing_patient_raises(tree):
    with pytest.raises(ValueError, match="IXI999"):
        find_scans(tree, "ixi", "dev", {"dev": {"ixi": ["IXI999"]}})

def test_patient_id():
    assert patient_id("ixi", "IXI012-HH-1211-T1.nii.gz") == "IXI012"
    assert patient_id("molab", "010024N__010024_19001219_TRA_3D_T1+GD_img.nii") == "010024N"
    assert patient_id("ms", "patient10_T1.nii.gz") == "patient10"

def test_real_splits_are_disjoint():   # reads REPO/splits.json: no patient in both dev and eval

def test_ground_truth_from_reference():
    # real reference/: 010005N → primary_origin "breast"; patient09 → ms_type "SMRR"; IXI002 → class "normal"
```

- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_datasets.py -q` → FAIL (import error).
- [ ] **Step 3: Implement** `common/datasets.py`.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat(common): resolve dev/eval scans and ground truth`

---

### Task 3: Volume preprocessing — `common/volume.py`

**Files:** Create `common/volume.py`, `tests/test_volume.py`

**Interfaces:**
- Produces:
  - `load_canonical(path: str) -> np.ndarray` — float32 `(D, H, W)`: D axial slices inferior→superior, H rows anterior→posterior, W columns patient right→left (radiological view). Same steps 1–2 as `load_volume` in `models/radfm/run.py`, then depth moved to axis 0; NaN → 0; a 4D file → first volume.
  - `normalize01(vol, lo: float = 0.5, hi: float = 99.5) -> np.ndarray` — percentile clip to [0, 1]; a constant volume → zeros.
  - `resize(vol, shape: tuple[int, int, int]) -> np.ndarray` — `scipy.ndimage.zoom`, `order=1`, exact output shape.
  - `foreground_range(vol01, thr: float = 0.1, min_frac: float = 0.05) -> tuple[int, int]` — first/last slice with ≥ `min_frac` of voxels above `thr`; the whole range if none.
  - `pick_slices(vol01, n: int) -> list[int]` — `round(linspace(a, b, n + 2))[1:-1]` over `foreground_range`.
  - `to_rgb_images(vol01, indices: list[int], size: int) -> list[PIL.Image.Image]` — RGB, `size × size`, uint8.
  - `to_pseudo_hu(vol01) -> np.ndarray` (`vol01 * 2000 - 1000`); `ct_to01(vol_hu) -> np.ndarray` (clip [-1000, 1000] → [0, 1]).
  - `to01(vol, dataset: str) -> np.ndarray` — `ct_to01` for `ct_rate`, `normalize01` for every other dataset. Adapters call this rather than `normalize01`/`ct_to01` (MedM-VL is the one exception, see its task).

- [ ] **Step 1: Write the failing tests** (tiny NIfTIs written with nibabel into `tmp_path`)

```python
def test_canonical_axes():
    data = np.zeros((4, 5, 6), np.float32); data[0, 4, 5] = 1   # most left, most anterior, most superior
    out = load_canonical(save(data, np.eye(4)))
    assert out.shape == (6, 5, 4) and out[5, 0, 3] == 1

def test_sagittal_affine_gives_axial_stack():
    ras = np.random.default_rng(0).random((4, 5, 6)).astype(np.float32)
    sag = ras.transpose(1, 2, 0)                                   # sag[i, j, k] = ras[k, i, j]
    affine = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float)
    np.testing.assert_array_equal(load_canonical(save(sag, affine)), load_canonical(save(ras, np.eye(4))))

def test_normalize01_range(): ...        # random → min 0, max 1; constant → all zeros
def test_resize_exact_shape(): ...       # (10, 20, 30) → resize(·, (32, 64, 64)).shape == (32, 64, 64)
def test_pick_slices_inside_foreground():
    vol = np.zeros((40, 8, 8), np.float32); vol[10:30] = 1.0
    idx = pick_slices(vol, 4)
    assert len(idx) == 4 and all(10 <= i <= 29 for i in idx) and idx == sorted(set(idx))
def test_to_rgb_images(): ...            # 2 images, mode "RGB", size (448, 448)
def test_hu_mappings():
    assert to_pseudo_hu(np.array([0.0, 1.0])).tolist() == [-1000.0, 1000.0]
    assert ct_to01(np.array([-2000.0, 0.0, 3000.0])).tolist() == [0.0, 0.5, 1.0]
def test_to01_dispatch():
    v = np.array([-2000.0, 0.0, 3000.0])
    assert to01(v, "ct_rate").tolist() == [0.0, 0.5, 1.0]
    assert to01(v, "ixi").tolist() == normalize01(v).tolist()
```

- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_volume.py -q` → FAIL.
- [ ] **Step 3: Implement** `common/volume.py`.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat(common): canonical volume loading and model input helpers`

---

### Task 4: Record, prompts and parsing — `common/schema.py`, `common/parse.py`

**Files:** Create `common/schema.py`, `common/parse.py`, `common/prompts/full_json_ixi.txt` (Appendix A), `common/prompts/full_json_molab.txt` (Appendix B), `common/prompts/full_json_ms.txt` (Appendix C), `common/prompts/full_json_ct_rate.txt` (Appendix A with `a medical head scan (CT or MRI)` → `a medical scan (CT or MRI)`), `tests/test_schema.py`, `tests/test_parse.py`

**Interfaces:**
- Consumes: `common.datasets.DATASETS`; `evaluation.check_schema.key_paths` (tests).
- Produces (`common/schema.py`):
  - `PARSED_KEYS: dict[str, list[str]]` — copy of `PARSED_KEYS` in `models/radfm/run.py`, plus `"ct_rate": PARSED_KEYS["ixi"]`.
  - `VOCAB: dict[str, tuple[str, ...]]` — the allowed values of each categorical field, exactly as listed in Appendices A–C (`modality`, `specialized_sequence`, `plane`, `diagnosis_name`, `diagnosis_detailed`, `primary_origin` (spelled `bladder`, `ovary`), `primary_origin_subtype`, `ms_type`, `ms_lesion_location`, `ms_lesion_burden`).
  - `full_prompt(dataset: str) -> str` (reads `common/prompts/full_json_<dataset>.txt`); `slice_preface(n: int) -> str` = `f"The images are {n} axial slices of the same 3D scan, ordered from inferior to superior.\n"`.
  - `normalize_values(dataset: str, raw: dict) -> dict` — returns every key of `PARSED_KEYS[dataset]` except `scan_name`; case-insensitive vocabulary match plus synonyms (`blader`→`bladder`, `overy`→`ovary`, `T1 with contrast`/`T1-weighted with contrast`/`T1Gd`/`T1 post-contrast`→`T1C+`, `relapsing-remitting`→`SMRR`, `secondary progressive`→`SMSP`, `primary progressive`→`SMPP`); anything else → `None`; numeric fields → float clamped to [0, 1], non-numeric → `None`; `ms_active_lesions` → bool or `None`; `icd10_code` kept only if it matches `^[A-Z]\d{2}(\.\d+)?$`. Idempotent.
  - `build_record(model: str, dataset: str, mode: str, row_index: int, image_path: str, metadata: dict, values: dict, error: str | None) -> dict` — the structure of `build_record` in `models/radfm/run.py`; `experiment_id = f"{model}_{DATASETS[dataset]['experiment']}"`, `system_prompt_type = f"{mode}_v1"`, `model_requested` and `api_response.model` = `model`; `scan_name = image_path` for MOLAB only.
- Produces (`common/parse.py`):
  - `parse_json_answer(text: str) -> tuple[dict | None, list[str]]` — uses the `<answer>…</answer>` content if present, strips ``` fences, takes the first balanced `{…}`, drops trailing commas, `json.loads`; problems ⊂ `{"no_json", "invalid_json", "echo:<field>"}`; a string value containing `<` or `|` becomes `None` and adds `echo:<field>`.
  - `parse_choice(text: str, options: list[str], style: str) -> int | None` — `None` if every option text appears (echo); else a leading letter (`style "A"`: `^\s*\(?([A-D])[.)\s]`; `style "a"`: `^\s*\(([a-d])\)`); else the index of the one option text that appears (case-insensitive); else `None`.
  - `extract_from_report(report: str, dataset: str) -> dict` — one fixed keyword extractor shared by every report model; returns raw schema values (the runner normalizes). Rules, applied to sentences split on `[.;\n]`, skipping for findings any sentence matching `\b(no|without|negative for|absence of|not)\b`:
    - `diagnosis_name`: first match of tumor `metasta|tumou?r|neoplas|\bmass\b|glioma|meningioma|glioblastoma|lymphoma` → `tumor`; `multiple sclerosis|demyelinat` → `multiple sclerosis`; `infarct|isch(a)?emi|stroke|h(a)?emorrhag` → `stroke`; else (negated sentences included) `no (acute |significant )?(intracranial )?abnormalit|unremarkable|within normal limits|normal (brain|study|mri)` → `normal`; else `lesion|hyperintens|enhanc|abnormal` → `other abnormalities`; else `None`.
    - `diagnosis_detailed`: tumor → first of glioma, meningioma, pituitary (`pituitary_tumor`), schwannoma, medulloblastoma, germinoma, neurocytoma, papilloma, granuloma, tuberculoma, `carcinoma|metasta` (`carcinoma`); stroke → `hemorrhagic` if `h(a)?emorrhag`, else `ischemic`.
    - `modality`: `\bMRI?\b|magnetic resonance` → MRI; `\bCT\b|computed tomography` → CT. `specialized_sequence`, `plane`: only explicit words (FLAIR; `post-contrast|gadolinium|T1C` → T1C+; T2; T1; axial/sagittal/coronal).
    - MOLAB `primary_origin`: `small cell` + lung → `lung-SCLC`, other lung → `lung-NSCLC`, breast, melanoma, `kidney|renal`, `colon|colorectal`, bladder, `ovar`, `uter|endometri`, `pharyn`, `sarcoma` → `bone-sarcoma`. MS: `relapsing` → SMRR, `secondary progressive` → SMSP, `primary progressive` → SMPP; `ms_lesion_location` first of periventricular, juxtacortical, infratentorial, deep white matter.
    - All confidence and severity fields `None`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_schema.py
@pytest.mark.parametrize("ds,tag", [("ixi", "healthy"), ("molab", "brain_metastases"), ("ms", "ms")])
def test_record_matches_reference_structure(ds, tag):
    ref = first_success_row(REPO / f"reference/radfm_{tag}_processed.jsonl")
    rec = build_record("lingshu-7b", ds, "full_json", 0, "x.nii.gz", ref["input"]["metadata"],
                       normalize_values(ds, {}), None)
    assert key_paths(rec) == key_paths(ref)

def test_experiment_and_mode():
    rec = build_record("lingshu-7b", "molab", "full_json", 0, "a", {}, {}, None)
    assert rec["experiment_id"] == "lingshu-7b_brain_metastases"
    assert rec["input"]["system_prompt_type"] == "full_json_v1"

def test_error_record(): ...   # parsed_response is None, status "error", error text kept

def test_normalize_values():
    v = normalize_values("molab", {"modality": "mri", "specialized_sequence": "T1 with contrast",
                                   "diagnosis_name": "Tumor", "primary_origin": "overy",
                                   "severity_score": "1.4", "extra": 1})
    assert (v["modality"], v["specialized_sequence"], v["diagnosis_name"], v["primary_origin"],
            v["severity_score"]) == ("MRI", "T1C+", "tumor", "ovary", 1.0)
    assert set(v) == set(PARSED_KEYS["molab"]) - {"scan_name"}
    assert normalize_values("molab", v) == v
    assert normalize_values("ixi", {"diagnosis_name": "brain lesion"})["diagnosis_name"] is None

@pytest.mark.parametrize("ds", ["ixi", "molab", "ms", "ct_rate"])
def test_full_prompt_lists_dataset_keys(ds):
    assert all(f'"{k}"' in full_prompt(ds) for k in PARSED_KEYS[ds] if k != "scan_name")

# tests/test_parse.py
def test_plain_json():      assert parse_json_answer('{"modality": "MRI"}') == ({"modality": "MRI"}, [])
def test_fenced_answer_tag():
    t = '<think>x</think><answer>```json\n{"plane": "axial",}\n```</answer>'
    assert parse_json_answer(t) == ({"plane": "axial"}, [])
def test_no_json():         assert parse_json_answer("The image shows") == (None, ["no_json"])
def test_invalid_json():    assert parse_json_answer('{"modality": MRI}') == (None, ["invalid_json"])
def test_echoed_placeholder_becomes_null():
    obj, problems = parse_json_answer('{"modality": "<MRI|CT|null>", "plane": "axial"}')
    assert obj == {"modality": None, "plane": "axial"} and "echo:modality" in problems

OPTS = ["Axial", "Sagittal", "Coronal", "Oblique"]
def test_parse_choice():
    assert parse_choice("B. Sagittal", OPTS, "A") == 1
    assert parse_choice("(c) Coronal", OPTS, "a") == 2
    assert parse_choice("sagittal", OPTS, "A") == 1
    assert parse_choice("A. Axial B. Sagittal C. Coronal D. Oblique", OPTS, "A") is None
    assert parse_choice("I don't know", OPTS, "A") is None

def test_extract_negation():
    v = extract_from_report("No evidence of metastasis. Unremarkable brain MRI.", "molab")
    assert (v["diagnosis_name"], v["modality"]) == ("normal", "MRI")
def test_extract_tumor():
    v = extract_from_report("Enhancing mass in the left frontal lobe consistent with metastasis "
                            "from lung cancer.", "molab")
    assert (v["diagnosis_name"], v["diagnosis_detailed"], v["primary_origin"]) == ("tumor", "carcinoma", "lung-NSCLC")
def test_extract_ms():
    v = extract_from_report("Multiple periventricular hyperintense lesions suggestive of demyelinating disease.", "ms")
    assert (v["diagnosis_name"], v["ms_lesion_location"]) == ("multiple sclerosis", "periventricular")
def test_extract_confidences_are_null(): ...   # every *_confidence and severity_score is None
```

- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_schema.py tests/test_parse.py -q` → FAIL.
- [ ] **Step 3: Implement** `common/schema.py`, `common/parse.py`, the three prompt files.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat(common): record format, prompts, JSON/report/choice parsing`

---

### Task 5: Per-field questions — `common/qna.py`

**Files:** Create `common/qna.py`, `tests/test_qna.py`

**Interfaces:**
- Consumes: `common.parse.parse_choice`.
- Produces:
  - `@dataclass(frozen=True) class Question: field: str; text: str; options: tuple[str, ...]; values: tuple` — `values[i]` is the schema value for `options[i]`; `None` means "Other / Cannot be determined" (follow-up level or null).
  - `BANK: dict[str, Question | list[Question]]` — question and option text **verbatim** from the "Шаблони по поле" table in the model notes (M3D column, exactly four options each); hierarchical fields (`diagnosis_detailed` for tumor R1–R3, `primary_origin` R1–R4 and the lung-type question) are lists. Severity values `0.0, 0.25, 0.5, 0.75`; tumor R1 option "Metastasis" → `carcinoma`.
  - `format_mcq(q: Question, style: str, order: tuple[int, ...]) -> str` — style `"A"`: `"{text} Choices: A. {o} B. {o} C. {o} D. {o}"`; style `"a"`: `"{text} (a) {o} (b) {o} (c) {o} (d) {o}"`, options in `order`.
  - `AskFn = Callable[[str], tuple[str, list[float] | None]]` — question text → (raw answer, probabilities of the four letters in displayed order, or `None`).
  - `ask(q: Question, ask_fn: AskFn, style: str) -> dict` — asks in order `(0, 1, 2, 3)` and reversed `(3, 2, 1, 0)`; returns `{"value", "confidence", "agree", "answers", "method"}`; `value`/`confidence` come from the original order (confidence `None` without probabilities); `agree` = both orders gave the same option; `method` ∈ `letter`, `option_text`, `none`.
  - `interrogate(dataset: str, ask_fn: AskFn, style: str) -> tuple[dict, dict]` — (raw values, log per field). Order and rules: `modality`; `specialized_sequence` only if modality ≠ CT; `plane`; `diagnosis_name` (+ `diagnosis_confidence`); `diagnosis_detailed` only for tumor (R1→R2→R3 on "Other") or stroke, else `None`; `severity_score` (+ `severity_confidence`); MOLAB: `primary_origin` R1→R4 (+ lung type when Lung), `origin_confidence` = product of confidences along the path (`None` if any is `None`), `primary_origin_subtype` and `subtype_confidence` `None`; MS: `ms_type`, `ms_lesion_location`, `ms_lesion_burden`, `ms_active_lesions = None` (no contrast in open_ms_data), `ms_confidence` = probability of the "Multiple sclerosis" option in the diagnosis question (`None` without probabilities); `icd10_code = None`.

- [ ] **Step 1: Write the failing tests**

```python
def test_format_mcq_styles():
    q = BANK["plane"]
    assert format_mcq(q, "A", (0, 1, 2, 3)) == \
        "Which plane is displayed in the image? Choices: A. Axial B. Sagittal C. Coronal D. Oblique"
    assert format_mcq(q, "a", (0, 1, 2, 3)) == \
        "Which plane is displayed in the image? (a) Axial (b) Sagittal (c) Coronal (d) Oblique"

def test_reverse_order_maps_back():          # ask_fn always answers "Sagittal" → value "sagittal", agree True
def test_position_bias_flagged():            # ask_fn always answers "A" → value "axial", agree False
def test_echo_of_all_options_is_none():      # ask_fn echoes the question → value None, method "none"
def test_ct_skips_sequence():                # "B" to the modality question → specialized_sequence None, not in log
def test_ms_fields():                        # always "A" → ms keys present, ms_active_lesions None
def test_origin_hierarchy():
    # R1 → "D" (Other), R2 → "A" (Kidney); probabilities always [0.1, 0.2, 0.3, 0.4]
    values, _ = interrogate("molab", fake, "A")
    assert values["primary_origin"] == "kidney"
    assert values["origin_confidence"] == pytest.approx(0.4 * 0.1)
```

- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_qna.py -q` → FAIL.
- [ ] **Step 3: Implement** `common/qna.py`.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat(common): per-field MCQ bank with order-reversal check`

---

### Task 6: The run loop — `common/runner.py`

**Files:** Create `common/runner.py`, `tests/test_runner.py`

**Interfaces:**
- Consumes: `datasets.find_scans`, `datasets.ground_truth`, `datasets.DATASETS`, `volume.load_canonical`, `schema.normalize_values`, `schema.build_record`.
- Produces:
  - `MODES = ("full_json", "qna_per_field", "findings_then_extract", "clip_or_heads")`
  - `class Adapter(Protocol)`: attributes `name: str`, `model_id: str`, `mode: str`; `load(self) -> None`; `predict(self, scan: Scan, volumes: list[np.ndarray]) -> tuple[dict, dict]` (raw values, JSON-serialisable details). `volumes` are `load_canonical` arrays of `scan.files`.
  - `run(adapter, split: str, datasets: list[str], data_dir: str, reference_dir: str, results_dir: str, splits: dict, limit: int | None = None, controls: bool = False, ms_flair: bool = False) -> Path` (returns the output folder)
  - `main(adapter, argv: list[str] | None = None) -> None` — CLI: `--split {dev,eval,full}` (required), `--datasets` (comma list, default: every key of `DATASETS`), `--limit N`, `--controls`, `--ms-flair`, `--data /data`, `--reference /reference`, `--results /results`, `--splits /manu/splits.json`.

Behavior: output folder `results_dir/<name>/<split>/` with `name = adapter.name + ("-flair" if ms_flair else "")`. Per dataset: `<name>_<tag>_processed.jsonl` (via `build_record`, metadata from `ground_truth`), `<name>_<tag>_details.jsonl` (`image_path, patient, files, seconds, details, error`), and with `--controls` one record in `<name>_<tag>_controls.jsonl`: the first scan with every volume replaced by zeros of the same shape, `image_path = "BLANK__" + first.image_path`, `row_index = -1`, written only if the file has no record yet. Resume by `image_path`. `adapter.load()` only if there is work. An exception in one scan → error record (`repr(e)`), `torch.cuda.empty_cache()` if torch is already imported, and the run continues. `run_info.json` (rewritten at start): `model, model_id, mode, split, datasets, ms_flair, started, host, git_commit` (env `MANU_GIT_COMMIT`), `versions` (torch, transformers, numpy, nibabel when importable). Progress lines like RadFM: `[i/N] <image_path> <seconds>s <diagnosis_name or error>`.

- [ ] **Step 1: Write the failing tests** (a `FakeAdapter`, name `"fake"`, mode `"full_json"`, counting `load()` calls, recording the volumes it receives, returning `({"modality": "MRI", "diagnosis_name": "tumor"}, {"raw": "x"})` and raising `RuntimeError("boom")` for IXI002; a tmp data tree with 4×4×4 NIfTIs for the dev patients; the repo's `reference/`)

```python
def test_dev_run_writes_one_record_per_scan(tmp):   # 1 + 2 + 1 records; key_paths == reference; MOLAB metadata primary_origin "breast"
def test_error_scan_is_recorded_and_run_continues(tmp):  # IXI002 status "error", "boom" in error; others "success"
def test_resume_skips_done(tmp):                     # second run: no duplicate image_path, load() called once in total
def test_controls_written_once(tmp):                 # controls twice → one record per dataset, "BLANK__" prefix, adapter got all-zero arrays
def test_ms_flair_name_suffix(tmp):                  # ms_flair=True → folder results/fake-flair/dev/
def test_run_info(tmp):                              # model "fake", split "dev", mode "full_json"
def test_limit(tmp):                                 # limit=1 → one record per dataset
```

- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_runner.py -q` → FAIL.
- [ ] **Step 3: Implement** `common/runner.py`.
- [ ] **Step 4: Run** the full suite → PASS.
- [ ] **Step 5: Commit** `feat(common): shared run loop with resume, error isolation and blank controls`

---

### Task 7: Cluster job and environment setup

**Files:** Create `jobs/run_model.sh`, `jobs/setup_env.sh`, `tests/test_jobs.py`

**Interfaces:**
- Produces the per-model folder convention every model task follows: `models/<m>/__init__.py`, `run.py`, `requirements.txt`, `NOTES.md` (headings: Setup, Dev run, Eval run, Pitfalls) and `env.sh` defining `SIF` (e.g. `containers/ngc2412.sif`), `SIF_SOURCE` (`docker://nvcr.io/nvidia/pytorch:24.12-py3`), `HF_REPOS` (space-separated repo ids), `GIT_REPOS` (space-separated `url@commit`, cloned into `third_party/<repo name>`), optional `EXTRA_BINDS`.
- `sbatch -J <m> jobs/run_model.sh <m> <runner args…>`; `bash jobs/setup_env.sh <m>` on the login node. Both honour `DRY_RUN=1` (print the commands, run nothing).

`jobs/run_model.sh`: the `#SBATCH` header of `jobs/job_radfm.sh` (`--output=results/slurm/%x_%j.out`), `set -euo pipefail`, `source models/$MODEL/env.sh`, export `SINGULARITYENV_HF_HOME=/hf_cache`, `SINGULARITYENV_HF_HUB_OFFLINE=1`, `SINGULARITYENV_TRANSFORMERS_OFFLINE=1`, `SINGULARITYENV_PYTHONUSERBASE=/pyuser`, `SINGULARITYENV_PYTHONPATH=/manu`, `SINGULARITYENV_MANU_GIT_COMMIT=$(cat .manu_commit 2>/dev/null || echo unknown)`; `mkdir -p results/slurm pyuser/$MODEL third_party`; then

```bash
CMD=(singularity exec --nv -B .:/manu -B ./data:/data -B ./results:/results -B ./reference:/reference
     -B ./hf_cache:/hf_cache -B "./pyuser/$MODEL:/pyuser" -B ./third_party:/third_party ${EXTRA_BINDS:-}
     "$SIF" python3 "/manu/models/$MODEL/run.py" "$@")
```

then `nvidia-smi`, the command, and `pip freeze` (with the same `PYTHONUSERBASE`) into `results/$MODEL/requirements_freeze.txt`.

`jobs/setup_env.sh <m>`: pull `$SIF` from `$SIF_SOURCE` if missing; inside the image, write the image's `torch`, `torchvision`, `torchaudio` pins to a constraints file and `pip install --user --no-cache-dir -c <pins> -r /manu/models/<m>/requirements.txt` with `PYTHONUSERBASE=/pyuser` (`pyuser/<m>` bound); `snapshot_download` each `HF_REPOS` entry with `HF_HOME=/hf_cache` (gated repos need `HF_TOKEN` exported on the login node); clone/checkout each `GIT_REPOS` entry; finally print `torch.__version__`, `transformers.__version__` from inside the image.

- [ ] **Step 1: Write the failing tests** (`tests/test_jobs.py`; a throwaway `models/_testmodel/env.sh` created and removed by a fixture)

```python
def test_run_model_dry_run(testmodel):
    out = run(["bash", "jobs/run_model.sh", "_testmodel", "--split", "dev"], env={"DRY_RUN": "1"})
    for s in ["singularity exec --nv", "./pyuser/_testmodel:/pyuser", "containers/ngc2412.sif",
              "/manu/models/_testmodel/run.py --split dev"]:
        assert s in out

def test_unknown_model_fails():      # exit code != 0 for "nope"
def test_setup_env_dry_run(testmodel):   # output contains "pip install --user" and " -c "
def test_scripts_parse():            # bash -n on both scripts returns 0
def test_model_dirs_complete():
    for d in (REPO / "models").iterdir():
        if d.is_dir() and d.name not in ("radfm", "__pycache__") and not d.name.startswith("_"):
            assert all((d / f).exists() for f in ["__init__.py", "run.py", "requirements.txt", "env.sh", "NOTES.md"])
            reqs = [l for l in (d / "requirements.txt").read_text().splitlines() if l.strip() and not l.startswith("#")]
            assert all("==" in l or "@ git+" in l for l in reqs)
```

- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_jobs.py -q` → FAIL.
- [ ] **Step 3: Implement** both scripts.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat(jobs): generic model job and login-node environment setup`

---

### Task 8: Replicate the layout on the cluster

**Files:** Create `tools/migrate_cluster.sh`, `tools/sync_to_cluster.sh`, `tools/fetch_results.sh`, `tests/test_migrate.py`

**Interfaces:**
- Consumes: `tools/layout.py` (`CLUSTER_LAYOUT`, CLI `--cluster`).
- Produces: `bash tools/sync_to_cluster.sh` (Mac, EduVPN on): writes `.manu_commit` (`git rev-parse --short HEAD`), then `rsync -av --exclude-from=.gitignore --exclude .git ./ "${CLUSTER:-martin.janev@10.10.65.6}:${REMOTE:-/home/hpc/users/martin.janev/manu}/"` (never `--delete`; `RSYNC_DRY=1` adds `-n`). `bash tools/fetch_results.sh`: `rsync -av` of the remote `results/` into `./results/`, excluding `archive/`. `bash tools/migrate_cluster.sh [--apply]` (run on the cluster in the repo root, after the first sync).

`migrate_cluster.sh` — dry run by default (prints `mv a b` lines, changes nothing); with `--apply` moves only when the source exists and the destination does not; never deletes; idempotent. Moves:

| on the cluster (old) | to |
|---|---|
| `code/` (whole folder) | `archive/code/cluster/` |
| root `job_radfm.sh`, `Dockerfile`, `prezemi_podatoci.sh`, `proveri_podatoci.py`, `a.py` | `archive/cluster_root/` (synced copies now live in `jobs/`, `containers/`, `data_tools/`) |
| root `prezemi_molab.py`, `sporedi_molab.py`, `linkovi.txt`, `molab_373_datoteki.txt`, `*.log` | `data_tools/` |
| `results/*.out` | `results/slurm/` |
| every other top-level file in `results/` | `results/archive/cluster_<YYYY-MM-DD>/`; then **copy** `radfm_*_processed.jsonl` and `radfm_*_details.jsonl` from there to `results/radfm/full/` |
| `RadFM/` | `third_party/RadFM/` |
| contents of `pyuser/` when `pyuser/radfm/` does not exist | `pyuser/radfm/` |

Then `mkdir -p` of `results/slurm results/radfm/full third_party pyuser hf_cache containers`, list unknown root entries as `left in place: <name>`, and run `python3 tools/layout.py --cluster .`.

- [ ] **Step 1: Write the failing tests** (a fake *old* cluster tree in `tmp_path`: `code/run_radfm.py`, `job_radfm.sh`, `Dockerfile`, `prezemi_molab.py`, `results/radfm_healthy_processed.jsonl`, `results/3d-radfm_1946.out`, `RadFM/Quick_demo/x`, `pyuser/lib/python3.10/site-packages/x`, `data/ixi/T1/a.nii.gz`, `data/molab/nifti/`, `data/open_ms_data/cross_sectional/raw/`, `containers/radfm.sif`, `hf_cache/`, `notes.txt`; then the repo's tracked files copied in, as a sync would)

```python
def test_dry_run_changes_nothing(old_tree):      # tree snapshot identical; stdout contains "mv "
def test_apply_produces_cluster_layout(old_tree):
    run_migrate(old_tree, "--apply")
    assert check(old_tree, CLUSTER_LAYOUT) == []
    assert (old_tree / "third_party/RadFM/Quick_demo/x").exists()
    assert (old_tree / "pyuser/radfm/lib/python3.10/site-packages/x").exists()
    assert (old_tree / "results/radfm/full/radfm_healthy_processed.jsonl").exists()
    assert (old_tree / "results/slurm/3d-radfm_1946.out").exists()
    assert (old_tree / "data/ixi/T1/a.nii.gz").exists()
def test_apply_is_idempotent(old_tree):          # second --apply prints no "mv ", tree unchanged
def test_unknown_files_left(old_tree):           # notes.txt still at root, reported as "left in place: notes.txt"
def test_sync_scripts_parse():                   # bash -n on the three scripts
```

- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_migrate.py -q` → FAIL.
- [ ] **Step 3: Implement** the three scripts.
- [ ] **Step 4: Run** the full suite → PASS.
- [ ] **Step 5: Commit** `feat(tools): replicate the new layout on the cluster`
- [ ] **Step 6 [cluster — Martin]:** `bash tools/sync_to_cluster.sh`; on the cluster `bash tools/migrate_cluster.sh` (read the plan), then `bash tools/migrate_cluster.sh --apply` → last line `OK` from `layout.py --cluster`. Check RadFM still runs: `sbatch jobs/job_radfm.sh ms 1` → one new record in `results/radfm/full/radfm_ms_processed.jsonl`.

---

### Task 9: Split-aware evaluation — `evaluation/evaluate.py`

**Files:** Modify `evaluation/evaluate.py`; Create `tests/test_evaluate_split.py`

**Interfaces:**
- Consumes: `common.datasets.load_splits`, `patient_id`; `common.schema.build_record` (tests); existing `class_metrics`, `field`, `norm_diag`, `TRUTH_DIAG`, `describe`.
- Produces:
  - `load_split(results: Path, reference: Path, split: str, splits: dict) -> dict[str, dict[str, list]]` — model → dataset → `[(image_path, parsed_response, metadata)]`, datasets in order ixi, molab, ms, ct_rate. Sources: `results/<m>/<split>/*_processed.jsonl` (model = folder name, dataset from the file tag); `results/<m>/full/*_processed.jsonl` filtered to the split's patients when `results/<m>/<split>/` does not exist (model `"<m> (full run)"`); `reference/*_processed.jsonl` filtered to the split's patients (model from `describe(path, "outputs")`). Inserts the repo root into `sys.path` itself (the script runs with `-I`).
  - `blank_agreement(results: Path, model: str, split: str) -> float | None` — over datasets with a controls file: share of categorical fields (no confidences) where the control's value equals the most common real value for that dataset; mean over datasets.
  - `split_report(runs: dict, results: Path, split: str) -> list[str]` — markdown (records and valid count every dataset; accuracy columns and pooled bal.acc use IXI, MOLAB and MS only): main table `model | mode | records | valid | IXI acc | MOLAB acc (scan) | MOLAB acc (patient majority) | MS acc | pooled bal.acc | blank agreement | median s/scan` (`valid` = status success with at least one non-null field; mode from `system_prompt_type` without `_v1`, `""` → `full_json (colleagues)`; seconds from `*_details.jsonl`), then the existing confusion and collapse sections for the same runs.
  - CLI: `python3 -I evaluation/evaluate.py --split dev|eval [--results results] [--reference reference] [--splits splits.json]` → writes `results/report_<split>.md` and prints it. Without `--split` the current behavior is unchanged.

- [ ] **Step 1: Write the failing tests** (fixture files built with `build_record`)

```python
def test_reference_filtered_to_split(tmp):     # reference rows IXI002 + IXI012, split dev → only IXI002
def test_full_run_filtered(tmp):               # results/radfm/full has eval + other patients → only eval, model "radfm (full run)"
def test_split_report_numbers(tmp):
    # model "m": IXI002 → normal, 4 × 010005N → tumor, patient05 → stroke
    text = "\n".join(split_report(load_split(...), results, "dev"))
    assert "| m | full_json | 6 | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% | 66.7% |" in text
def test_blank_agreement(tmp):                 # identical control → 1.0; all fields different → 0.0
def test_selftest_still_passes():              # selftest() runs without AssertionError
```

- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_evaluate_split.py -q` → FAIL.
- [ ] **Step 3: Implement** the additions.
- [ ] **Step 4: Run** the full suite and `python3 -I evaluation/evaluate.py --selftest` → PASS / `selftest OK`.
- [ ] **Step 5: Commit** `feat(evaluation): per-split report with blank-control agreement`

---

### Task 10: CT-RATE data and its evaluation

Purpose: a small chest-CT control. For Med-2E3 and MedM-VL it is their training domain: if they read CT-RATE correctly but not brain MRI, the problem is domain shift, not our pipeline. For the other six models it checks whether they at least recognise CT and an abnormal scan when the body part is outside their training.

**Files:** Create `data_tools/prezemi_ct_rate.py`, `tests/test_ct_rate.py`; Modify `common/datasets.py`, `splits.json`, `evaluation/evaluate.py`

**Interfaces:**
- Consumes: `common.datasets` (Task 2), `evaluation.evaluate.load_split` / `split_report` (Task 9).
- Produces:
  - `select_files(repo_files: list[str], patients: list[str]) -> dict[str, str]` — patient → repo path of its `valid_N_a_1.nii.gz` matching `^dataset/valid[^/]*/valid_N/valid_N_a/valid_N_a_1\.nii\.gz$`, preferring a `valid_fixed` path when both exist; `ValueError` naming a missing patient.
  - CLI `python3 data_tools/prezemi_ct_rate.py [--split dev|eval|all]` — needs only `huggingface_hub` (token from env `HF_TOKEN`, an account that accepted the dataset terms). Downloads into `data/ct_rate/`, keeping repo paths, the selected volumes plus `dataset/multi_abnormality_labels/valid_predicted_labels.csv`, `dataset/radiology_text_reports/validation_reports.csv`, `dataset/metadata/validation_metadata.csv` from `ibrahimhamamci/CT-RATE` (gated, 21.3 TB in total; only these files are fetched). Then prints per patient: path, number of positive labels, and, if nibabel imports, shape and min/max.
  - `DATASETS["ct_rate"] = {"glob": "ct_rate/dataset/valid*/valid_*/valid_*_a/valid_*_a_1.nii.gz", "tag": "ct_rate", "experiment": "ct_rate"}`; `patient_id("ct_rate", "valid_12_a_1.nii.gz") == "valid_12"`; `image_path` = file basename.
  - `ground_truth` for `ct_rate` → `{"dataset": "ct_rate", "class": None, "modality": "CT", "axial_plane": "axial", "modality_subtype": None, "icd10_code": None, "abnormalities": [label columns equal to 1 for that VolumeName]}`.
  - `splits.json`: `"ct_rate": ["valid_1"]` in dev, `["valid_2", "valid_3", "valid_4"]` in eval.
  - `ct_rate_table(runs: dict) -> list[str]` in `evaluation/evaluate.py`, appended by `split_report` when CT-RATE rows exist: `model | n | modality = CT | plane = axial | normal/abnormal agrees` (truth abnormal = at least one positive label; prediction abnormal = `diagnosis_name` not `normal`; `null` counts as wrong).

- [ ] **Step 1: Write the failing tests**

```python
def test_select_files_prefers_fixed():
    files = ["dataset/valid/valid_1/valid_1_a/valid_1_a_1.nii.gz",
             "dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz",
             "dataset/valid/valid_1/valid_1_a/valid_1_a_2.nii.gz"]
    assert select_files(files, ["valid_1"]) == {"valid_1": files[1]}
def test_select_files_missing_patient_raises():   # ValueError matching "valid_9"
def test_ct_rate_scans(tree):                     # fake tree → Scan("ct_rate", "valid_1", "valid_1_a_1.nii.gz", (...,))
def test_ct_rate_ground_truth(tree):              # tiny labels CSV with Emphysema = 1 → abnormalities == ["Emphysema"], modality "CT"
def test_ct_rate_table():
    # one model, one CT-RATE patient with a positive label; prediction modality "CT", plane "axial", diagnosis "normal"
    assert "| m | 1 | 100.0% | 100.0% | 0.0% |" in "\n".join(ct_rate_table(runs))
```

- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_ct_rate.py -q` → FAIL.
- [ ] **Step 3: Implement** the script, the `ct_rate` branches in `common/datasets.py`, `ct_rate_table`, and the `splits.json` entries.
- [ ] **Step 4: Run** the full suite → PASS.
- [ ] **Step 5: Commit** `feat(data): CT-RATE control set and its evaluation table`
- [ ] **Step 6 [cluster — Martin]:** accept the CT-RATE terms on Hugging Face; on the login node `export HF_TOKEN=…` and `singularity exec -B .:/manu --pwd /manu containers/radfm.sif python3 data_tools/prezemi_ct_rate.py --split all`. Expected: four volumes; min/max HU-like (about -1024…3000). If the values are already scaled to another range, stop and fix `to01` for `ct_rate` before any model runs. If the three eval patients are all normal or all abnormal, swap one in `splits.json` for a patient with the other status and download again.

---

## Model tasks (11–18)

### Model task template

Every model task follows these steps; its own block gives the specifics. The implementer does Steps 1–5; Martin does the `[cluster]` steps (Claude can read the logs with him).

**Files (each model):** `models/<m>/{__init__.py, run.py, requirements.txt, env.sh, NOTES.md}`, `tests/adapters/test_<m>.py` (no `__init__.py` in `tests/adapters/`).

**Adapter shape:** `class <X>Adapter` with `name`, `model_id`, `mode`, `load()`, `predict(scan, volumes)`, and exactly one GPU method (`_generate` or `_score`) that tests replace with `monkeypatch`. Torch and transformers are imported inside `load()` or the GPU method, so `run.py` imports on a CPU machine. `if __name__ == "__main__": runner.main(<X>Adapter())`. Test volume: `np.random.default_rng(0).random((40, 64, 64), dtype=np.float32)`. Intensities: `to01(vol, scan.dataset)` first (CT-RATE gets the CT window); MedM-VL is the one exception (see its task). On CT-RATE every model runs as-is: the full-JSON models get `full_prompt("ct_rate")`, BrainGemma3D and HLIP their fixed brain prompt/classes, the QnA models the same questions.

- [ ] **Step 1:** Write the failing tests listed in the task.
- [ ] **Step 2:** Run `.venv/bin/python -m pytest tests/adapters/test_<m>.py -q` → FAIL (import error).
- [ ] **Step 3:** Implement the adapter, `requirements.txt` (exact pins), `env.sh`, `NOTES.md` (headings only).
- [ ] **Step 4:** Run the task's tests and the full suite → PASS.
- [ ] **Step 5:** Commit `feat(<m>): adapter`.
- [ ] **Step 6 [cluster]:** `bash tools/sync_to_cluster.sh`; on the login node `bash jobs/setup_env.sh <m>` → ends with the torch (2.6.0a0…) and transformers versions.
- [ ] **Step 7 [cluster]:** `sbatch -J <m> jobs/run_model.sh <m> --split dev --controls` → `results/<m>/dev/` has 7 processed records (1 + 4 + 1 + 1) and 4 controls; for each tag `python3 evaluation/check_schema.py results/<m>/dev/<m>_<tag>_processed.jsonl reference/radfm_<tag>_processed.jsonl` → `same structure`.
- [ ] **Step 8:** `bash tools/fetch_results.sh`; `python3 -I evaluation/evaluate.py --split dev`; fill `NOTES.md` › Dev run: peak VRAM (from `nvidia-smi` in the log), median s/scan, valid rate, blank agreement, parse problems (`details`), position-bias rate (QnA models: share of `agree == False`). If the dev run shows a pipeline bug, fix it and repeat Step 7: this is what the dev split is for.
- [ ] **Step 9 [cluster]:** `sbatch -J <m> jobs/run_model.sh <m> --split eval --controls` → 28 processed records and 4 controls.
- [ ] **Step 10:** Fetch, fill `NOTES.md` › Eval run and › Pitfalls, commit `docs(<m>): dev/eval notes`.

### Task 11: Lingshu-7B — `models/lingshu7b` (name `lingshu-7b`, mode `full_json`)

- Also create `models/__init__.py` and `models/_qwen25vl.py`: `load_qwen25vl(model_id: str) -> tuple[model, processor]` (bf16, `attn_implementation="sdpa"`, on cuda) and `generate_images(model, processor, images: list[Image], text: str, max_new_tokens: int) -> str` (one user message: the images, then the text; chat template; greedy). Shared with Task 12.
- `env.sh`: `SIF=containers/ngc2412.sif`, `HF_REPOS="lingshu-medical-mllm/Lingshu-7B"`. Requirements: transformers (≥ 4.49, the version needed for Qwen2.5-VL), accelerate, qwen-vl-utils, all `==`.
- Input: `to01` → `pick_slices(·, 8)` → `to_rgb_images(·, 448)`; text = `slice_preface(8) + full_prompt(scan.dataset)`; `max_new_tokens=1024`.
- `predict`: `obj, problems = parse_json_answer(text)`; values = `obj or {}`; details `{"prompt", "raw", "problems", "slices"}`.
- Tests: `test_inputs_are_8_rgb_448` (captured images: 8, RGB, 448×448; text ends with `full_prompt(dataset)`); `test_parses_fenced_json` (fake output with ``` fences → `values["diagnosis_name"] == "tumor"`); `test_invalid_json_is_empty_with_problem`.
- Pitfall to check: Lingshu-32B scored higher on a black image than on real MRI (model notes) → compare blank agreement.

### Task 12: QoQ-Med-VL-7B — `models/qoqmed7b` (name `qoq-med-vl-7b`, mode `full_json`)

- `HF_REPOS="ddvd233/QoQ-Med-VL-7B"`; uses `models/_qwen25vl.py`; requirements as Task 11. License: research only (note in `NOTES.md`).
- Input: 4 slices (its training recipe), 448, `slice_preface(4) + full_prompt(dataset)`.
- Tests: `test_inputs_are_4_rgb_448`; `test_parses_fenced_json`.

### Task 13: BrainGemma3D — `models/braingemma3d` (name `braingemma3d`, mode `findings_then_extract`)

- `HF_REPOS="praiselab-picuslab/BrainGemma3D"` plus its MedGemma 1.5 base (gated, HAI-DEF: accept on Hugging Face, `HF_TOKEN` on the login node); code and pins from the model card Quick Start (clone into `third_party/` via `GIT_REPOS` if the card loads from GitHub).
- Input: `to01` → `resize(·, (32, 128, 128))` (the card's `target_size`; if the published loader requires 64 slices, use `(64, 128, 128)` and record it in `NOTES.md`). Prompt exactly `Generate a radiology report for this brain MRI FLAIR scan.`; `max_new_tokens=512`.
- `predict`: values = `extract_from_report(report, scan.dataset)`; details `{"raw_report", "input_shape"}`.
- Extra run on its training sequence: `sbatch -J braingemma3d jobs/run_model.sh braingemma3d --split eval --datasets ms --ms-flair --controls` → `results/braingemma3d-flair/eval/`.
- Tests: `test_volume_is_32_128_128_in_unit_range`; `test_prompt_is_exact`; `test_report_goes_through_shared_extractor` (fake report `"No acute intracranial abnormality."` → `diagnosis_name == "normal"`, `details["raw_report"]` equals it).
- Pitfall to check: the prompt says FLAIR, so `specialized_sequence` = FLAIR in a report is an echo, not a reading.

### Task 14: HLIP — `models/hlip` (name `hlip`, mode `clip_or_heads`)

- `GIT_REPOS="https://github.com/Zch0414/hlip@<commit>"`; the brain-MRI checkpoint named in the repo README (project notes mention ViT-L `dualdinotxt1568`); record the exact file in `env.sh`.
- `preprocess(vol: np.ndarray) -> np.ndarray` returns `(48, 224, 224)`, replicating the intensity steps of `src/hlip_test/zeroshot_pubbrain5.py` on our canonical volume.
- Zero-shot: template `This study shows: {c} identified.` over the class strings copied verbatim from `src/hlip/zeroshot_metadata_pubbrain5.py` and `zeroshot_metadata_mrrate.py`, mapped to groups: no significant abnormalities → `normal`; metastasis, glioma, meningioma, "Metastatic malignant neoplasm to brain" → `tumor`; stroke classes → `stroke`; "Demyelinating disease of central nervous system" → `multiple sclerosis`. `_score(volume) -> dict[class, probability]` (softmax over classes).
- `predict`: `diagnosis_name` = group with the largest summed probability, `diagnosis_confidence` = that sum; every other field `None` (no constants).
- Tests: `test_preprocess_shape`; `test_grouping` (fake probabilities with tumor classes summing highest → `tumor`, confidence = the sum); `test_other_fields_null`.

### Task 15: Fleming-VL-8B — `models/fleming8b` (name `fleming-vl-8b`, mode `full_json`)

- `HF_REPOS="IQuestLab/Fleming-VL-8B"` (InternVL3-based, `trust_remote_code`); pins from the model card.
- Input: `to01` → `resize(·, (32, 256, 256))` float32 → the card's own 3D path (it samples 11 slices at 448²); do not use `pick_slices`. Text = `full_prompt(dataset)` plus the card's `<think>/<answer>` instruction if it requires one; `max_new_tokens=2048`.
- Tests: `test_npy_shape_and_range`; `test_parses_answer_tag` (`<think>…</think><answer>{"modality": "MRI"}</answer>` → `values["modality"] == "MRI"`); `test_max_new_tokens_2048`.

### Task 16: Med-2E3 — `models/med2e3` (name `med-2e3`, mode `qna_per_field`)

- `HF_REPOS="shiym2000/Med-2E3-M3D"`, `GIT_REPOS="https://github.com/MSIIP/Med-2E3@<commit>"`; load through the functions of `src/inference.py` (import, not subprocess), `conv_version="phi"`; pins from the repo. In Step 6 check `import flash_attn` inside the image; if it is missing and the model accepts `sdpa`, use it and note it.
- Input: `to01` → `resize(·, (32, 256, 256))` → `(1, 32, 256, 256)`.
- `ask_fn`: prompt `"<image3d>\n" + question`, `max_new_tokens=16`, letter probabilities from the first generated token's scores over the four letter tokens; `predict` = `interrogate(scan.dataset, ask_fn, "A")`, details = its log.
- Tests: `test_input_shape`; `test_prompt_prefix_and_style` (captured prompt starts `<image3d>\n`, contains `Choices: A.`); `test_interrogate_wired` (fake `_generate` always returns `"A"` → `values["modality"] == "MRI"`, `values["plane"] == "axial"`).

### Task 17: MedM-VL-CT-3B-en — `models/medmvl3b` (name `medm-vl-ct-3b`, mode `qna_per_field`)

- `HF_REPOS="shiym2000/MedM-VL-CT-3B-en"` (Llama-3.2-3B), `GIT_REPOS="https://github.com/MSIIP/MedM-VL@<commit>"` (`lvlm` package), `conv_version="llama3"`, `transformers==4.42.4` (repo pin).
- Input: MRI: `normalize01` → `resize(·, (32, 256, 256))` → `to_pseudo_hu` (the model's loader clips to [-1000, 1000]); CT-RATE: the real HU volume resized and clipped to [-1000, 1000] (no pseudo-HU).
- `ask_fn` as Task 16 with style `"a"`: `"<image3d>\n" + "{Q} (a) … (d) …"`.
- Tests: `test_pseudo_hu_range` (min ≥ -1000, max ≤ 1000); `test_prompt_style` (contains `(a) `, no `Choices:`); `test_parses_paren_letter` (fake `"(c) Coronal"` for the plane question → `values["plane"] == "coronal"`).

### Task 18: HealthGPT-Pro-27B — `models/healthgptpro27b` (name `healthgpt-pro-27b`, mode `full_json`)

- `HF_REPOS="lintw/HealthGPT-Pro-27B"` (Qwen3.6-27B base per the model tree; ≈ 54 GB in bf16, fits one A100 80 GB; no quantization); transformers version from the model card.
- Input: `pick_slices(·, 10)` → `to_rgb_images(·, 448)` passed as **one video** of 10 frames in the card's message format; text `slice_preface(10) + full_prompt(dataset)`.
- Tests: `test_ten_frames_as_one_video` (captured message has one video with 10 frames); `test_parses_fenced_json`.
- Pitfall to check: peak VRAM; on OOM, record it in `NOTES.md` rather than silently quantizing.

---

### Task 19: Results and pitfalls

**Files:** Create `docs/results_eval_<YYYY-MM-DD>.md`

- [ ] **Step 1:** After all eval runs: `bash tools/fetch_results.sh`; `python3 -I evaluation/evaluate.py --split eval` → `results/report_eval.md` (includes RadFM and the colleagues' models filtered to the same patients).
- [ ] **Step 2:** Write `docs/results_eval_<date>.md`: the main table copied from `results/report_eval.md`; one short paragraph per model from its `NOTES.md` (mode, input recipe, valid rate, blank agreement, position bias, VRAM, s/scan, pitfalls); a closing line that 12 patients (9 brain MRI + 3 CT-RATE) check the pipeline and are not a benchmark, and that `--split full` runs the full datasets with the same command.
- [ ] **Step 3: Verify:** re-run the evaluation and check that every number in the doc matches `results/report_eval.md`.
- [ ] **Step 4:** Commit `docs: eval-split results and pitfalls`.

---

## Appendix A — `common/prompts/full_json_ixi.txt` (colleagues' "NOV" prompt, verbatim, trailing spaces stripped)

```text
You are an expert neuroradiologist analyzing a medical head scan (CT or MRI).
Your task: generate a strict JSON object describing the findings.
Output ONLY this JSON schema below without any extra text or explanation:
{
  "modality": "<MRI|CT|null>",
  "specialized_sequence": "<FLAIR|T1|T2|T1C+|null>",
  "plane": "<axial|sagittal|coronal|null>",
  "diagnosis_name": "<tumor|stroke|multiple sclerosis|normal|other abnormalities|null>",
  "diagnosis_detailed": "<glioma|meningioma|pituitary_tumor|carcinoma|germinoma|granuloma|medulloblastoma|neurocytoma|papilloma|schwannoma|tuberculoma|ischemic|hemorrhagic|null>",
  "icd10_code": "<ICD-10 or null>",
  "severity_score": <float 0-1 or null>,
  "diagnosis_confidence": <float 0-1>,
  "severity_confidence": <float 0-1>
}
Examples:
{
  "modality": "CT",
  "specialized_sequence": null,
  "plane": "axial",
  "diagnosis_name": "stroke",
  "diagnosis_detailed": "ischemic",
  "icd10_code": "I63",
  "severity_score": 0.65,
  "diagnosis_confidence": 0.92,
  "severity_confidence": 0.89
}
{
  "modality": "MRI",
  "specialized_sequence": "FLAIR",
  "plane": "axial",
  "diagnosis_name": "tumor",
  "diagnosis_detailed": "glioma",
  "icd10_code": "C71",
  "severity_score": 0.80,
  "diagnosis_confidence": 0.95,
  "severity_confidence": 0.90
}
Remember: Output must be a valid JSON EXACTLY matching the schema above.
```

## Appendix B — `common/prompts/full_json_molab.txt` ("Medgemma 1.5 brain metastasis base prompt", verbatim)

```text
You are an expert neuroradiologist analyzing a medical head scan (CT or MRI).
Your task: generate a strict JSON object describing the findings.
Output ONLY this JSON schema below without any extra text or explanation:
{
  "modality": "<MRI|CT|null>",
  "specialized_sequence": "<FLAIR|T1|T2|T1C+|null>",
  "plane": "<axial|sagittal|coronal|null>",
  "diagnosis_name": "<tumor|stroke|multiple sclerosis|normal|other abnormalities|null>",
  "diagnosis_detailed": "<glioma|meningioma|pituitary_tumor|carcinoma|germinoma|granuloma|medulloblastoma|neurocytoma|papilloma|schwannoma|tuberculoma|ischemic|hemorrhagic|null>",
  "severity_score": <float 0-1 or null>,
  "diagnosis_confidence": <float 0-1>,
  "severity_confidence": <float 0-1>,
  "primary_origin": <breast|lung-NSCLC|lung-SCLC|melanoma|kidney|colon|blader|unknown|ovary|pharynx|bone-sarcoma|uterus|null> (in case of metastatic cancer),
  "primary_origin_subtype": <breast:luminal_a|luminal_b|triple_negative|her2; lung:adenocarcinoma|epidermoid|large_cell_carcinoma|large_cell_endocrine; other: null> (in case of metastatic cancer),
  "origin_confidence": <float 0-1 or null>,
  "subtype_confidence": <float 0-1 or null>
}
Remember: Output must be a valid JSON EXACTLY matching the schema above.
```

## Appendix C — `common/prompts/full_json_ms.txt` (reconstructed in the style of B; replace with the colleagues' MS prompt if they have one)

```text
You are an expert neuroradiologist analyzing a medical head scan (CT or MRI).
Your task: generate a strict JSON object describing the findings.
Output ONLY this JSON schema below without any extra text or explanation:
{
  "modality": "<MRI|CT|null>",
  "specialized_sequence": "<FLAIR|T1|T2|T1C+|null>",
  "plane": "<axial|sagittal|coronal|null>",
  "diagnosis_name": "<tumor|stroke|multiple sclerosis|normal|other abnormalities|null>",
  "diagnosis_detailed": "<glioma|meningioma|pituitary_tumor|carcinoma|germinoma|granuloma|medulloblastoma|neurocytoma|papilloma|schwannoma|tuberculoma|ischemic|hemorrhagic|null>",
  "severity_score": <float 0-1 or null>,
  "diagnosis_confidence": <float 0-1>,
  "severity_confidence": <float 0-1>,
  "ms_type": <SMRR|SMSP|SMPP|null> (in case of multiple sclerosis),
  "ms_lesion_location": <periventricular|juxtacortical|infratentorial|deep white matter|null> (in case of multiple sclerosis),
  "ms_lesion_burden": <none|low|moderate|high|null> (in case of multiple sclerosis),
  "ms_active_lesions": <true|false|null> (in case of multiple sclerosis),
  "ms_confidence": <float 0-1 or null>
}
Remember: Output must be a valid JSON EXACTLY matching the schema above.
```
