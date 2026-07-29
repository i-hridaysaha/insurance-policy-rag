# Hosted demo image, sized for a free HuggingFace Spaces "Docker" Space (2 vCPU / 16 GB).
# Generation runs on a free OpenAI-compatible API (set GEN_API_KEY as a Space secret), so no
# model weights for the LLM ship here -- only the ~420 MB Sentence-BERT retriever, baked in at
# build time so the first question does not pay a cold download.
FROM python:3.12-slim

# HuggingFace runs the container as a non-root user (uid 1000). Give it a writable home so the
# transformers / sentence-transformers cache lands somewhere it can actually write.
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model into the image layer.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-mpnet-base-v2')" \
    && chown -R user:user /home/user/.cache

COPY --chown=user:user . .
USER user

# Spaces routes public traffic to container port 7860.
EXPOSE 7860
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "7860"]
