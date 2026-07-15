"""Inspect what retrieval actually returns for a question.

    python -m scripts.query "is maternity covered in the first year"
    python -m scripts.query --compare "what is my room rent limit"

--compare shows dense-only, sparse-only and hybrid side by side, which is how you see
what the fusion is actually buying you on a given query.

This is a retrieval debugging tool. There is no generation layer yet, by design: the
answer layer is only worth building once you can see that the right clause is coming back.
"""

from __future__ import annotations

import argparse

from src.config import BM25_PATH, CHUNKS_PATH, FAISS_IDS_PATH, FAISS_PATH
from src.ingest.chunk import load_chunks
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.sparse import SparseRetriever


def load() -> tuple[HybridRetriever, dict]:
    chunks = load_chunks(CHUNKS_PATH)
    dense = DenseRetriever()
    dense.load(FAISS_PATH, FAISS_IDS_PATH)
    sparse = SparseRetriever()
    sparse.load(BM25_PATH)
    return HybridRetriever(chunks, dense, sparse), {c["chunk_id"]: c for c in chunks}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--compare", action="store_true", help="show dense vs sparse vs hybrid")
    ap.add_argument("--full", action="store_true", help="print full chunk text")
    args = ap.parse_args()

    retriever, by_id = load()

    if args.compare:
        dense_ids = retriever.retrieve_dense_only(args.question, args.k)
        sparse_ids = retriever.retrieve_sparse_only(args.question, args.k)
        hybrid = retriever.retrieve(args.question, args.k)

        print(f"\nQ: {args.question}\n")
        for label, ids in (
            ("DENSE  (Sentence-BERT + FAISS)", dense_ids),
            ("SPARSE (BM25)", sparse_ids),
            ("HYBRID (RRF)", [r.chunk_id for r in hybrid]),
        ):
            print(f"  {label}")
            if not ids:
                print("    (no results)")
            for i, cid in enumerate(ids, 1):
                c = by_id[cid]
                print(f"    {i}. [{c['section']:>8}] {c['section_title'][:56]}")
            print()
        return 0

    results = retriever.retrieve(args.question, args.k)
    print(f"\nQ: {args.question}\n")
    if not results:
        print("  no chunks retrieved")
        return 0

    for i, r in enumerate(results, 1):
        agree = "both" if r.dense_rank and r.sparse_rank else ("dense" if r.dense_rank else "sparse")
        print(f"  {i}. {r.citation()}")
        print(
            f"     rrf={r.fused_score:.4f}  found_by={agree}"
            f"  dense_rank={r.dense_rank}  sparse_rank={r.sparse_rank}"
        )
        body = r.text if args.full else r.text[:260].replace("\n", " ") + "..."
        print(f"     {body}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
