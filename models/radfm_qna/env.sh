# The image of the full RadFM run (built from containers/radfm.Dockerfile, already on the cluster). Only if it is
# missing does setup_env.sh pull SIF_SOURCE, the Dockerfile's base; requirements.txt then adds the Dockerfile's pins.
SIF=containers/radfm.sif
SIF_SOURCE=docker://nvcr.io/nvidia/pytorch:23.05-py3
# Nothing to download: the RadFM clone with pytorch_model.bin is in third_party/RadFM/Quick_demo, and Med-KEBERT
# (loaded by RadFM's embedding layer) is in hf_cache, both from the full run.
HF_REPOS=""
GIT_REPOS=""
