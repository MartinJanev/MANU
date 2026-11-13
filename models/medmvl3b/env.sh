SIF=containers/ngc2412.sif
SIF_SOURCE=docker://nvcr.io/nvidia/pytorch:24.12-py3
# The checkpoint, plus the repos its loader reads by name at load time (the job is offline): Llama-3.2-3B-Instruct
# (gated; config, tokenizer and weights, which the checkpoint's then replace), M3D-CLIP (config, code, weights) and
# bert-base-uncased, which M3D-CLIP's __init__ loads with BertModel.from_pretrained.
HF_REPOS="shiym2000/MedM-VL-CT-3B-en meta-llama/Llama-3.2-3B-Instruct GoodBaiBai88/M3D-CLIP bert-base-uncased"
# Not HEAD: the last commit before the 2025-10-10 config refactor, whose config format the checkpoint has (NOTES.md).
GIT_REPOS="https://github.com/MSIIP/MedM-VL@4a84399db005a7ec3f2e8556518e696b5d2732c8"
