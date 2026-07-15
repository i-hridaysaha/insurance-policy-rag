"""Hybrid retrieval: Reciprocal Rank Fusion over the dense and sparse retrievers.

WHY RRF AND NOT A WEIGHTED SCORE SUM
------------------------------------
Cosine similarity lives in [-1, 1] and clusters tightly (most chunks in this corpus score
0.3-0.6 against any query, because they are all insurance prose). BM25 is unbounded and its
scale shifts with query length and corpus statistics. Summing them, even with weights, means
committing to a normalisation that has to be re-tuned whenever the corpus changes, and
min-max normalising per query makes the top score always 1.0 regardless of whether the top
hit was any good.

RRF sidesteps this. It reads only the RANK each retriever assigned, never the score, so
there is nothing to normalise and nothing to re-tune:

    score(d) = sum over retrievers of  1 / (k + rank(d))

The k constant (60, from Cormack et al. 2009) damps the contribution of top ranks enough
that a document ranked #1 by one retriever and unranked by the other does not automatically
beat a document ranked #2 by both. That agreement-rewarding property is what we want: the
chunks both retrievers like are the chunks most likely to be right.

The honest tradeoff: RRF discards score magnitude, so it cannot tell "rank 1 with 0.95
similarity" from "rank 1 with 0.31 similarity". That matters for the refusal path (a query
about nothing in the document still produces a ranked list), so refusal is decided on the
raw retriever scores BEFORE fusion, not on the fused score. See `retrieve` below.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.config import CANDIDATE_K, DEFAULT_TOP_K, RRF_K
from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse import SparseRetriever


@dataclass
class Result:
    chunk_id: str
    text: str
    section: str
    section_title: str
    page_start: int
    page_end: int
    clause_type: str
    plan_scope: str
    fused_score: float
    dense_rank: int | None
    sparse_rank: int | None
    dense_score: float | None
    sparse_score: float | None

    def citation(self) -> str:
        page = (
            f"p.{self.page_start}"
            if self.page_start == self.page_end
            else f"p.{self.page_start}-{self.page_end}"
        )
        return f"Section {self.section} - {self.section_title}, {page}"


class HybridRetriever:
    def __init__(self, chunks: list[dict], dense: DenseRetriever, sparse: SparseRetriever):
        self.by_id = {c["chunk_id"]: c for c in chunks}
        self.dense = dense
        self.sparse = sparse

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        candidate_k: int = CANDIDATE_K,
    ) -> list[Result]:
        dense_hits = self.dense.search(query, candidate_k)
        sparse_hits = self.sparse.search(query, candidate_k)

        dense_rank = {cid: r for r, (cid, _) in enumerate(dense_hits, 1)}
        sparse_rank = {cid: r for r, (cid, _) in enumerate(sparse_hits, 1)}
        dense_score = dict(dense_hits)
        sparse_score = dict(sparse_hits)

        fused: dict[str, float] = {}
        for cid, r in dense_rank.items():
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + r)
        for cid, r in sparse_rank.items():
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + r)

        ordered = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        ordered = self._dedupe_by_section(ordered)[:top_k]

        results = []
        for cid, score in ordered:
            c = self.by_id[cid]
            results.append(
                Result(
                    chunk_id=cid,
                    text=c["text"],
                    section=c["section"],
                    section_title=c["section_title"],
                    page_start=c["page_start"],
                    page_end=c["page_end"],
                    clause_type=c["clause_type"],
                    plan_scope=c["plan_scope"],
                    fused_score=score,
                    dense_rank=dense_rank.get(cid),
                    sparse_rank=sparse_rank.get(cid),
                    dense_score=dense_score.get(cid),
                    sparse_score=sparse_score.get(cid),
                )
            )
        return results

    def _dedupe_by_section(self, ranked: list[tuple[str, float]]) -> list[tuple[str, float]]:
        """Keep only the best-scoring chunk per section.

        A long clause is split across several chunks, and those chunks are near-identical in
        both term profile and embedding, so they rank adjacently and consume several of the k
        slots between them. Section 4.4 was taking two of the top five on its own, crowding out
        the section that actually held the answer.

        Sections, not chunks, are the unit that matters downstream: a citation names a section,
        and the answer layer wants k DISTINCT clauses to reason over, not the same clause three
        times. Deduplicating here spends the k budget on diversity instead of redundancy.
        """
        seen: set[str] = set()
        out: list[tuple[str, float]] = []
        for cid, score in ranked:
            section = self.by_id[cid]["section"]
            if section in seen:
                continue
            seen.add(section)
            out.append((cid, score))
        return out

    def _dedupe_ids(self, hits: list[tuple[str, float]], top_k: int) -> list[str]:
        return [cid for cid, _ in self._dedupe_by_section(hits)[:top_k]]

    def retrieve_dense_only(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[str]:
        # Deduplicated the same way as the hybrid, so the ablation compares like with like.
        return self._dedupe_ids(self.dense.search(query, CANDIDATE_K), top_k)

    def retrieve_sparse_only(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[str]:
        return self._dedupe_ids(self.sparse.search(query, CANDIDATE_K), top_k)
