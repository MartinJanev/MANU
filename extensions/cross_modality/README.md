# Cross-modality development adapters

This extension adds seven checkpoint releases that are not in the frozen
[paper protocol](../../evaluation/PROTOCOL.md). The paper roster, splits,
scoring, and batch jobs are unchanged. M3D-LaMed Phi-3 already has an
[adapter](../../models/m3dlamed_phi3); this extension does not duplicate it.

| CLI model | Input path | Transformers used in the earlier pilots |
|---|---|---|
| `sgmri` | SGMRI-Qwen8B: 32 axial images | 4.57.3 |
| `hulu` | Hulu-Med7B: 32 axial frames as one video; SDPA memory guard | 4.51.2 |
| `nv` | NV-Reason-CT: native 192-cube input; chest crop on CT, centered on MRI; thinking off | 5.6.2 |
| `photon` | Photon-S1: 1 mm RAI volume, center crop/pad, then upstream volume preprocessing | 4.51.3 |
| `radsight` | RadSight-4B: native volume; full-volume 1×1×5 mm resampling, no mask-based crop | 4.57.3 |
| `clinfusion` | ClinFusion-8B: native volume plus four slices from the upstream in-memory route (literal RAS indexing, not an axial claim) | 4.57.0 |
| `neurovfm` | NeuroVFM: native constrained report; one packed window for MRI, three for CT | 4.56.0 |

SGMRI and Hulu share the same slice indices and fixed chest CT window
(`-160` to `240` HU); MRI slice inputs use percentile scaling. The four
CT-focused volume paths are **transfer tests on MRI**. They map the 0.5th to
99.5th percentiles of positive MRI voxels to a CT processor's numeric range.
Those values are **not HU**. No masks or reference text are supplied to the
model. No weights are changed. MRI and chest CT differ in both modality and
anatomy; differences between the tracks do not isolate either factor.

## Run a development test

1. Use a separate GPU Python environment per model. Reuse a reviewed environment
   if available. The versions above are pilot starting points, not lockfiles for
   this extension. For NeuroVFM the pilot used torch `2.7.1+cu126`, FlashAttention
   `2.8.3` with fused dense/MLP and Triton norm support, and a matching `ptxas`.
   Check those native imports before attempting inference. The other ports use
   the SDPA paths; do not replace the existing paper environment.
2. Stage the local checkpoint from its [recorded repository and revision](run.py)
   and verify it against your acquisition record. The `revision` field in the
   local config is a declaration, not a checksum of the files. Hulu and NV
   execute reviewed custom Python from the local checkpoint. Do not use an
   unknown or mutable snapshot.
3. For Photon, RadSight, ClinFusion and NeuroVFM, check out the
   [pinned upstream commit](prepare_source.py) in a separate, clean directory.
   Prepare a local inference copy and inspect the source changes:

   ```bash
   python -m extensions.cross_modality.prepare_source clinfusion \
     /local/upstream/ClinFusion /local/assets/clinfusion-inference
   ```

   RadSight defers unused video imports. ClinFusion uses SDPA, requires MONAI
   and checks the actual final shard keys and shapes before strict loading.
   NeuroVFM defers training imports and requires strict encoder, connector and
   language-model restoration. These are local infrastructure changes, not claims
   of parity with the authors' exact environments.
4. Create a **local** JSON config. SGMRI, Hulu and NV use only `model` and
   `revision`. Photon and NeuroVFM also need `source`, the prepared directory.
   RadSight also needs `siglip`, a directory or symlink whose basename contains
   `VL3-SigLIP-NaViT`. ClinFusion also needs `qwen`, `dinov2` and `convnext`.

   | Auxiliary checkpoint | Revision |
   |---|---|
   | `DAMO-NLP-SG/VL3-SigLIP-NaViT` | `d7dded4d0bf07092c453579b3e324d4fa16105a3` |
   | `Qwen/Qwen3-VL-8B-Instruct` | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` |
   | `facebook/dinov2-large` | `47b73eefe95e8d44ec3623f8890bd894b6ea2d6c` |
   | `laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup` | `654d0f80ff73c58e7281a3ca7dc425589049e2e1` |

   Example SGMRI config:

   ```json
   {
     "model": "/local/assets/sgmri",
     "revision": "006aa50811a7d70bb3d40704b93cf5f2aae2ecd3"
   }
   ```

5. Run from the repository root in the matching environment with one visible
   GPU, authorized local data in the usual layout, and the private reference
   directory. Use corrected CT-RATE volumes verified to contain HU. Start with
   one dataset and a blank control:

   ```bash
   python -m extensions.cross_modality.run sgmri \
     --assets /local/assets/sgmri.json --data /local/data \
     --reference /local/reference --results results/ext-sgmri-ixi \
     --datasets ixi --controls
   ```

The launcher reads only the repository's `dev` split. It cannot select
`paper`, resume an output directory, download checkpoints, or submit jobs.
Six ports reuse the shared full-JSON prompt/parser and support `--task qna`
with the repository's alternating option order. Use `--datasets ms --ms-flair`
to test FLAIR separately. The extension deliberately does not enable the
diagnosis-provided prior condition. NeuroVFM writes native report JSON and
rendered text to the details log, not a synthetic diagnostic classification.
Its optional control zeros the packed image values but retains their
geometry and modality; it is not the same control as a blank raw scan.

## Validation and interpretation

Before proposing a later paper round, check complete weight restoration,
execution on every dev dataset, parsing, scan/blank differences, and any
available author example. Record preparation failures, runtime failures and
parser failures separately from reference disagreements. Raw output, labels,
scan files, local asset configs and credentials do not belong in this public
repository. Existing pilot jobs are not validation of this port: GPU smoke
tests of the shared-runner path are still required.

In the earlier, separate development pilots:

- On four training-partition CT-RATE cases, NV and Photon each matched 8/8
  report-derived answers about pleural effusion and consolidation. RadSight
  answered absent eight times, matching 7/8 and missing the only positive
  answer. Some Photon reports contradicted their targeted answers. These labels
  were not independently adjudicated and are not the paper CT task.
- SGMRI and Hulu produced MRI responses with formatting or consistency
  problems. NeuroVFM generated three valid native MRI reports, but validity
  did not establish agreement with the study references. ClinFusion's first
  CT test missed the positive consolidation. These are development observations,
  not a ranking or evidence of clinical accuracy.

No paired-timepoint inference was performed. Separate results on brain MRI
and chest CT, including any later transfer runs, should not be pooled with
the frozen paper tables.
