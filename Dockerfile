# Container image for a Docker-based host (>=1 GB RAM). This is the FULL stack: it bakes in the
# ~420 MB Sentence-BERT retriever and runs embedding locally, so it does NOT need EMBED_API_KEY.
# Generation still uses a free OpenAI-compatible API (set GEN_API_KEY). The free live demo does NOT
# use this file -- it deploys on Render from render.yaml with the slim, torch-free requirements and
# offloaded embedding (see docs/DEPLOY-RENDER.md). Kept for anyone deploying to a Docker host.
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
