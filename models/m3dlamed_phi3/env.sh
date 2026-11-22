SIF=containers/ngc2412.sif
SIF_SOURCE=docker://nvcr.io/nvidia/pytorch:24.12-py3
HF_REPOS="GoodBaiBai88/M3D-LaMed-Phi-3-4B"
# 16 GB duplicate of the safetensors shards
HF_IGNORE="model.bin"
