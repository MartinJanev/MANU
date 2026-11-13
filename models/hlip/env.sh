SIF=containers/ngc2412.sif
SIF_SOURCE=docker://nvcr.io/nvidia/pytorch:24.12-py3
# Checkpoint: pytorch_model.bin of zch0414/clip-vit_large-scan_study-dualdinotxt1568 (the README's Pub-Brain-5 model);
# BiomedBERT supplies the text tower's config and the tokenizer.
HF_REPOS="zch0414/clip-vit_large-scan_study-dualdinotxt1568 microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
GIT_REPOS="https://github.com/Zch0414/hlip@691297f7fcbedfeecea5387cd8e3a79057a82fbe"
