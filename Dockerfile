FROM nvcr.io/nvidia/pytorch:23.05-py3

RUN pip list --format=freeze | grep -E "^torch(vision|audio)?==" > /tmp/torch-pin.txt || true

# Верзиите се од requirements.txt на RadFM (transformers 4.28.1, како во трудот).
RUN pip install --no-cache-dir -c /tmp/torch-pin.txt \
        transformers==4.28.1 \
        tokenizers==0.13.3 \
        huggingface-hub==0.16.4 \
        sentencepiece==0.1.99 \
        einops==0.6.1 \
        einops-exts==0.0.4 \
        nibabel==5.1.0
