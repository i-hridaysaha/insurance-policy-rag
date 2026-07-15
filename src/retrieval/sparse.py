"""Sparse retrieval: BM25 over the same chunks.

This is not a baseline to be beaten. In insurance it is load-bearing, and it is the half
of the hybrid that saves the plan-specific questions.

Insurance queries hinge on exact terms that embeddings blur together. "Co-payment",
"deductible" and "sub-limit" are three different things with three different financial
consequences, and a bi-encoder will happily place them close together because they are all
"money you pay". Likewise "Optima Secure" vs "Optima Super Secure" vs "Optima Secure +"
differ by one token and mean different sums of money; cosine similarity between those three
strings is near 1.0. BM25 does not have that problem, because it matches tokens, not vibes.

Tokenisation keeps the domain's meaningful character patterns rather than stripping them:
  - "excl01" survives as one token (the IRDAI exclusion codes are literal query terms)
  - "24" and "36" survive (waiting periods ARE the query)
  - "pre-existing" -> "pre existing" so it matches both hyphenations in the source
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

import snowballstemmer
from rank_bm25 import BM25Okapi

from src.config import BM25_SIGNAL_GATE

# Words carrying no discriminative signal in a corpus where every chunk is about insurance.
# Deliberately short: aggressive stopword removal strips "not", which inverts the meaning of
# an exclusion clause, and this corpus is largely made of exclusion clauses.
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "of", "to", "in",
    "for", "on", "at", "by", "with", "as", "and", "or", "that", "this", "it",
    "i", "my", "me", "you", "your", "we", "our",
}

_STEMMER = snowballstemmer.stemmer("english")

# The document mixes British and American spellings of the same word ("hospitalization" 123
# times, "hospitalisation" twice). The stemmer does NOT unify those -- it maps them to "hospit"
# and "hospitalis" -- so they are normalised to one spelling first.
_ISE = re.compile(r"(is)(ation|ations|ing|ed|es|e)\b")


def tokenize(text: str) -> list[str]:
    """Lowercase, de-hyphenate, normalise spelling, stem.

    Stemming is the single biggest lever on BM25 here, and it was missing at first. Without it
    BM25 treats `cover`, `covers`, `covered` and `coverage` as four unrelated tokens -- 250
    occurrences fragmented across four types -- so a user asking "is physiotherapy covered?"
    scores zero term overlap against a clause reading "this Cover shall indemnify...". Same for
    claim/claims (196 occurrences split) and exclusion/exclusions. Adding the stemmer is a
    tokenisation fix, not a tuning knob; reaching for fusion weights to compensate for it would
    have been papering over the actual defect.

    Stemming does NOT blunt the exact-match property that makes BM25 worth having: the plan names
    stem to distinct tokens (optima / secur / super), so "Optima Secure" and "Optima Super Secure"
    stay distinguishable, which is the whole reason sparse retrieval is in this system.
    """
    text = text.lower().replace("-", " ")
    text = _ISE.sub(r"iz\2", text)  # hospitalisation -> hospitalization
    tokens = re.findall(r"[a-z0-9]+", text)
    return _STEMMER.stemWords([t for t in tokens if t not in STOPWORDS])


class SparseRetriever:
    def __init__(self) -> None:
        self.bm25: BM25Okapi | None = None
        self.chunk_ids: list[str] = []

    def build(self, chunks: list[dict]) -> None:
        self.chunk_ids = [c["chunk_id"] for c in chunks]
        corpus = [tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(corpus)

    def search(self, query: str, k: int, gate: float = BM25_SIGNAL_GATE) -> list[tuple[str, float]]:
        """Top-k BM25 hits, with weak matches gated out.

        `gate` drops any hit scoring below that fraction of this query's own top BM25 score.
        Without it BM25 returns a full ranked list for every query -- almost every chunk in an
        insurance corpus shares "policy" or "claim" with almost every question -- and that noisy
        tail carries real rank positions into the fusion, where RRF cannot tell it apart from
        signal. Abstaining is the correct behaviour when there is nothing to say. See
        BM25_SIGNAL_GATE in config for the measurement behind the value.
        """
        if self.bm25 is None:
            raise RuntimeError("index not built or loaded")
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self.chunk_ids, scores), key=lambda x: x[1], reverse=True)

        # A zero score means no shared term at all: not a weak result, the absence of one.
        hits = [(cid, float(s)) for cid, s in ranked[:k] if s > 0.0]
        if not hits:
            return []

        floor = gate * hits[0][1]
        return [(cid, s) for cid, s in hits if s >= floor]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({"bm25": self.bm25, "chunk_ids": self.chunk_ids}, f)

    def load(self, path: Path) -> None:
        with path.open("rb") as f:
            state = pickle.load(f)
        self.bm25 = state["bm25"]
        self.chunk_ids = state["chunk_ids"]
