SIF=containers/ngc2412.sif
SIF_SOURCE=docker://nvcr.io/nvidia/pytorch:24.12-py3
# Code: Prima's repo (its end-to-end pipeline, unchanged). Weights are not on HF: after setup_env.sh, Prima's own
# tools/download_models_and_setup_test.py (gdown) puts primafullmodel107.pt and vqvae_model_step16799.pth under
# third_party/prima/test/trained_models/ (command in NOTES.md).
GIT_REPOS="https://github.com/MLNeurosurg/prima@604f1fc4dd7059c278d4d34bbbcaab3e404fbcd4"
