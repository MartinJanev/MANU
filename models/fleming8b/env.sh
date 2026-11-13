SIF=containers/ngc2412.sif
SIF_SOURCE=docker://nvcr.io/nvidia/pytorch:24.12-py3
# The card loads model code and weights from this repo alone (trust_remote_code), so no GIT_REPOS.
HF_REPOS="IQuestLab/Fleming-VL-8B"
