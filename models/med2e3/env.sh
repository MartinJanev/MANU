SIF=containers/ngc2412.sif
SIF_SOURCE=docker://nvcr.io/nvidia/pytorch:24.12-py3
# The checkpoint, plus the repos its loader reads by name at load time (the job is offline): Phi-3 config, config
# code and tokenizer; SigLIP config and image processor; M3D-CLIP config, code, weights and the tokenizer of the
# TG-IS question; bert-base-uncased, which M3D-CLIP's __init__ loads with BertModel.from_pretrained.
HF_REPOS="shiym2000/Med-2E3-M3D microsoft/Phi-3-mini-128k-instruct google/siglip-large-patch16-256 GoodBaiBai88/M3D-CLIP bert-base-uncased"
GIT_REPOS="https://github.com/MSIIP/Med-2E3@0b4c6c457b2437cae686c6f49310cf2666c2f77f"
