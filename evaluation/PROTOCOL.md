# Protocol of the paper's runs

**Status: DRAFT until the tag `paper-freeze` exists (planned Mon 2026-11-23, 22:00).** Until then only technical
problems seen on the dev patients may change it. After the tag nothing here, in `common/`, in `models/` or in
`splits.json` changes before the runs end; a forced change is listed under Deviations with its date and reason.

## Question

How well do openly available 3D-capable MLLMs produce a structured diagnostic record from 3D medical images, and
how does that depend on the model family and on the prompting strategy?

## Data (`splits.json`, split `paper`, drawn 2026-11-22)

| dataset | patients | truth | seen before the runs? |
|---|---|---|---|
| IXI | 120, random | normal | no: dev and pilot patients excluded |
| MOLAB | all 75, first scan by date | tumor, primary origin | yes: 30 in the pilot, 1 in dev, all 75 in the March runs |
| open_ms_data | all 30, T1; FLAIR as a separate condition | MS, subtype from its `patient_info.csv` | yes: 29 in the pilot, 1 in dev |
| CT-RATE | 30 without and 30 with a positive label | normal / abnormal | 30 of 60 in the pilot |

MOLAB and MS are whole cohorts, so they could not be kept unseen. The paper says so.

## Models

Eleven MLLMs and HLIP as a non-LLM reference (`evaluation/stats.py::MODELS`). No model is added after the freeze.

Validation gate, decided on dev only, before the freeze: (1) the model loads its weights completely, (2) it runs
without errors on every dev dataset, (3) its answers parse on at least one dataset, (4) its output is not identical
for a scan and a blank volume in every field, (5) where the adapter can run the authors' example (RadFM), that
example gives a sensible, image-dependent answer. A model that fails stays in the paper under failure modes as
"could not be validated". Diagnostic accuracy on dev is never a criterion.

## Conditions

1. Native mode of every model on the four datasets, and on MS FLAIR.
2. Q&A for the six full-prompt models (`--task qna`).
3. Prior knowledge for the six full-prompt models on MOLAB and MS (`--task prior`).
4. One blank-volume control per model, condition and dataset.

The prompt is the same for every dataset. The only dataset-specific texts are the prior sentences (condition 3),
the Q&A follow-up questions for origin and MS subtype (separate calls that cannot influence the diagnosis
answer) and the chest question of CT-RATE, which is scored separately.

## Rules fixed in advance

- Unit: the patient (one scan each). An error, a null or an out-of-vocabulary answer is wrong and is never dropped.
- Q&A: the value is the first ask's; which option order is first alternates with a hash of the patient id.
- Several JSON objects for one scan: per field a strict majority, otherwise null.
- HLIP: mean probability per diagnosis group, renormalised.
- Primary outcome: balanced accuracy of `diagnosis_name` over IXI, MOLAB and MS T1.
- Interval: 95% percentile bootstrap, stratified by dataset, 10,000 resamples, seed 0.
- Comparisons: paired sign-flip permutation, 10,000 resamples, seed 0. Two families, each Holm-corrected:
  all 66 model pairs; the six Q&A vs. full-prompt pairs.
- Secondary: recall per dataset, accuracy, macro and weighted F1, confusion matrices; CT-RATE sensitivity,
  specificity and balanced accuracy; origin and subtype next to the majority class; descriptive fields as sanity
  checks; unparseable, abstained, top-answer share, order stability, blank answers.
- Not scored: `diagnosis_detailed` (no truth in these datasets), severity, and confidences (calibration is out of
  scope).
- Baselines: a constant answer (33.3%), and acquisition features only (`tools/site_baseline.py --split paper`).
- A run that is incomplete on a dataset is not scored there.

## What the results cannot show

- Each diagnosis is its own dataset, so a high score can be dataset recognition. No claim of disease detection in
  general (MR-RATE is future work).
- MS FLAIR has no negative FLAIR scans: it is a sensitivity only, outside the primary outcome.
- The 2D and 3D families differ in more than the encoder (training domain, interface, slice count): their
  comparison is descriptive.

## Deviations

None so far.
