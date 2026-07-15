"""Verify retrieval against the ground-truth questions, before building anything on top.

    python -m scripts.verify_retrieval
    python -m scripts.verify_retrieval --k 5 --show-misses

This is a gate, not the evaluation layer. It answers one question: does the right clause
come back? If it does not, no amount of prompt engineering downstream will save the answer,
and measuring generation quality on top of broken retrieval would just be measuring noise.

A question counts as a hit when ANY of its gold sections appears in the top-k. Questions whose
gold_sections is empty (the deliberate refusals) are scored separately: for those, a "hit" is
the WRONG outcome to chase, so we report what score the top result got instead, which is the
signal a refusal threshold would key off.
"""

from __future__ import annotations

import argparse

import yaml

from src.config import EVAL_QA_PATH
from src.retrieval.hybrid import HybridRetriever
from scripts.query import load


def recall_at_k(retrieved: list[str], gold: list[str]) -> bool:
    return any(g in retrieved for g in gold)


def rr(retrieved: list[str], gold: list[str]) -> float:
    """Reciprocal rank of the first gold section."""
    for i, sec in enumerate(retrieved, 1):
        if sec in gold:
            return 1.0 / i
    return 0.0


def sections_of(retriever: HybridRetriever, ids: list[str]) -> list[str]:
    return [retriever.by_id[i]["section"] for i in ids]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--show-misses", action="store_true")
    args = ap.parse_args()

    qa = yaml.safe_load(EVAL_QA_PATH.read_text())
    retriever, _ = load()

    answerable = [p for p in qa["pairs"] if p["gold_sections"]]
    refusals = [p for p in qa["pairs"] if not p["gold_sections"]]

    rows = []
    for p in answerable:
        q, gold = p["question"], p["gold_sections"]
        hybrid = sections_of(retriever, [r.chunk_id for r in retriever.retrieve(q, args.k)])
        dense = sections_of(retriever, retriever.retrieve_dense_only(q, args.k))
        sparse = sections_of(retriever, retriever.retrieve_sparse_only(q, args.k))
        rows.append(
            {
                "id": p["id"],
                "difficulty": p["difficulty"],
                "question": q,
                "gold": gold,
                "hybrid": hybrid,
                "hit_hybrid": recall_at_k(hybrid, gold),
                "hit_dense": recall_at_k(dense, gold),
                "hit_sparse": recall_at_k(sparse, gold),
                "rr_hybrid": rr(hybrid, gold),
                "rr_dense": rr(dense, gold),
                "rr_sparse": rr(sparse, gold),
            }
        )

    n = len(rows)

    def pct(key: str) -> float:
        return sum(r[key] for r in rows) / n * 100

    def mrr(key: str) -> float:
        return sum(r[key] for r in rows) / n

    print(f"\n{'=' * 66}")
    print(f"RETRIEVAL VERIFICATION   k={args.k}   n={n} answerable questions")
    print("=" * 66)
    print(f"\n{'retriever':<22} {f'recall@{args.k}':>10} {'MRR':>8}")
    print("-" * 42)
    print(f"{'dense  (SBERT+FAISS)':<22} {pct('hit_dense'):>9.1f}% {mrr('rr_dense'):>8.3f}")
    print(f"{'sparse (BM25)':<22} {pct('hit_sparse'):>9.1f}% {mrr('rr_sparse'):>8.3f}")
    print(f"{'hybrid (RRF)':<22} {pct('hit_hybrid'):>9.1f}% {mrr('rr_hybrid'):>8.3f}")

    print(f"\n{'by difficulty':<22} {'n':>4} {f'recall@{args.k}':>10}")
    print("-" * 42)
    for d in ("easy", "medium", "hard"):
        sub = [r for r in rows if r["difficulty"] == d]
        if sub:
            hit = sum(r["hit_hybrid"] for r in sub) / len(sub) * 100
            print(f"{d:<22} {len(sub):>4} {hit:>9.1f}%")

    # Where does the hybrid earn its keep? Cases one retriever alone would have missed.
    only_dense = [r for r in rows if r["hit_dense"] and not r["hit_sparse"]]
    only_sparse = [r for r in rows if r["hit_sparse"] and not r["hit_dense"]]
    neither = [r for r in rows if not r["hit_dense"] and not r["hit_sparse"]]
    print(f"\n{'complementarity':<22}")
    print("-" * 42)
    print(f"{'found by dense only':<22} {len(only_dense):>4}   {[r['id'] for r in only_dense]}")
    print(f"{'found by sparse only':<22} {len(only_sparse):>4}   {[r['id'] for r in only_sparse]}")
    print(f"{'found by neither':<22} {len(neither):>4}   {[r['id'] for r in neither]}")

    misses = [r for r in rows if not r["hit_hybrid"]]
    if misses:
        print(f"\n{'MISSES (hybrid)':<22} {len(misses)}")
        print("-" * 66)
        for r in misses:
            print(f"  [{r['id']}] {r['question']}")
            print(f"      gold     : {r['gold']}")
            print(f"      retrieved: {r['hybrid']}")

    if args.show_misses and misses:
        return 0

    # Refusal questions: nothing to hit. Report the top fused score, which is the quantity a
    # refusal threshold would be set on once the generation layer exists.
    print(f"\n{'REFUSAL CASES':<22} (must NOT be answered)")
    print("-" * 66)
    for p in refusals:
        top = retriever.retrieve(p["question"], 1)
        if top:
            r = top[0]
            print(f"  [{p['id']}] top hit: {r.section:<10} dense_score={r.dense_score:.3f}")
        else:
            print(f"  [{p['id']}] no chunks retrieved")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
