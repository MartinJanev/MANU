# GoodBaiBai88/M3D-LaMed-Phi-3-4B

## Setup

Sources read (2026-11-22): the Quick start and requirements.txt of github.com/BAAI-DCAI/M3D (main), and
config.json and modeling_m3d_lamed.py of the checkpoint.

- transformers 4.43.3, the version of the 2 Oct draft (Phi-3 needs >= 4.40; the repo's 4.39.1 has no Phi-3).
- monai 1.3.2 instead of 1.3.0 (the pin med2e3 already uses in this image on Python 3.12).
- Greedy decoding instead of the Quick start's sampling. Confidences stay null.

## Dev run

Not run yet.

## Eval run

Not run yet.

## Pitfalls

- The model is CT-trained (M3D-Data): compare brain MRI with CT-RATE before blaming the setup.
