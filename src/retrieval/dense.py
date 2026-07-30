"""Dense retrieval: Sentence-BERT embeddings over a FAISS index.

Catches paraphrase. A policyholder asks "can I get treatment at home instead of being
admitted", the clause says "Home Health Care ... treatment at Home ... would require
In-patient Care at a Hospital". No useful term overlap, but the meaning matches, and this
is the retriever that finds it.

Vectors are L2-normalised and the index is inner-product (IndexFlatIP), so the score is
cosine similarity. Flat (exhaustive) rather than IVF/HNSW: the corpus is a few hundred
chunks, exhaustive search is sub-millisecond, and an approximate index would trade exact
recall for a speedup we do not need. Choosing the boring index here is the right call, and
it means recall@k measures the embedding, not the approximation error.
"""

from __future__ import annotations

import json
from pathlib import Path

# HOSTED-EMBEDDING MODE. When EMBED_API_KEY is set, the query is embedded by a remote endpoint and
# this process must NOT import sentence_transformers / torch at all -- that is the whole point on a
# 512 MB host. So the torch import is conditional on the mode, decided once at import time.
from src.config import EMBED_API_KEY, EMBED_API_URL, EMBED_TIMEOUT, EMBEDDING_MODEL

_HOSTED = bool(EMBED_API_KEY)

# IMPORT ORDER IS LOAD-BEARING ON macOS/ARM. DO NOT REORDER, AND DO NOT LET AN
# IMPORT SORTER (isort/ruff) REORDER IT -- hence the isort: off fence below.
#
# faiss-cpu and torch each vendor their own OpenMP runtime:
#     faiss/_swigfaiss.so  -> @loader_path/.dylibs/libomp.dylib
#     libtorch_cpu.dylib   -> @rpath/libomp.dylib
# Loading two OpenMP runtimes into one process is undefined behaviour. In practice,
# whichever libomp loads FIRST wins, and if that is faiss's copy, torch segfaults
# (SIGSEGV, exit 139) the moment it dispatches real multi-threaded work -- i.e. on the
# first batch encode of non-trivial text. It does NOT reproduce on short strings, because
# those never spin up torch's thread pool, which is what makes this look flaky rather than
# deterministic.
#
# Importing sentence_transformers (and therefore torch) first makes torch's libomp win,
# and torch tolerates faiss loading afterwards. Verified: faiss-first + real corpus
# segfaults; torch-first + real corpus succeeds. In hosted mode torch is never imported, so
# there is no second OpenMP runtime and the ordering is moot -- only faiss loads.
#
# isort: off
if not _HOSTED:
    from sentence_transformers import SentenceTransformer  # noqa: I001  (must precede faiss)
import faiss

# isort: on
import httpx
import numpy as np

# Exhaustive search over a few hundred vectors is sub-millisecond single-threaded. Pinning
# faiss to one thread removes it from the OpenMP contention entirely, at no measurable cost.
faiss.omp_set_num_threads(1)


class DenseRetriever:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self._model = None
        self.index: faiss.Index | None = None
        self.chunk_ids: list[str] = []

    @property
    def model(self):
        # Lazy: loading the transformer costs seconds, and the eval harness constructs
        # retrievers it does not always query. Never reached in hosted mode.
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _encode(self, texts: list[str]) -> np.ndarray:
        if _HOSTED:
            return self._encode_hosted(texts)
        vecs = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,  # cosine similarity via inner product
            show_progress_bar=False,
        )
        return vecs.astype(np.float32)

    def _encode_hosted(self, texts: list[str]) -> np.ndarray:
        """Embed via the remote all-mpnet endpoint, then reproduce the local post-processing.

        The local path passes normalize_embeddings=True; HF's feature-extraction pipeline returns
        the same mean-pooled sentence vectors but UN-normalised, so we L2-normalise here to land on
        byte-for-byte the same space the FAISS index was built in. wait_for_model rides out the
        cold-load 503 instead of failing the first request after the model has been evicted.
        """
        r = httpx.post(
            EMBED_API_URL,
            timeout=EMBED_TIMEOUT,
            headers={"Authorization": f"Bearer {EMBED_API_KEY}"},
            json={"inputs": texts, "options": {"wait_for_model": True}},
        )
        r.raise_for_status()
        vecs = np.asarray(r.json(), dtype=np.float32)
        if vecs.ndim == 1:  # a single input can come back as a bare vector
            vecs = vecs[None, :]
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.clip(norms, 1e-12, None)

    def build(self, chunks: list[dict]) -> None:
        self.chunk_ids = [c["chunk_id"] for c in chunks]
        vecs = self._encode([c["text"] for c in chunks])
        self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(vecs)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        if self.index is None:
            raise RuntimeError("index not built or loaded")
        q = self._encode([query])
        scores, idx = self.index.search(q, min(k, self.index.ntotal))
        return [
            (self.chunk_ids[i], float(s))
            for i, s in zip(idx[0], scores[0])
            if i != -1
        ]

    def save(self, index_path: Path, ids_path: Path) -> None:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        ids_path.write_text(json.dumps(self.chunk_ids))

    def load(self, index_path: Path, ids_path: Path) -> None:
        self.index = faiss.read_index(str(index_path))
        self.chunk_ids = json.loads(ids_path.read_text())
