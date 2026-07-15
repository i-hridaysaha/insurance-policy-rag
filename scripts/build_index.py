"""Build the retrieval corpus and both indexes from the source PDF.

    python -m scripts.build_index

Ingestion is REGION-BASED. The document is cut into page regions (see src/config.py) and each
is chunked on its own terms, because the clause parser cuts a page stream at headings and will
happily let the last heading in a region swallow everything downstream of it. Chunking the
whole document as one stream let Section 17 -- the final heading -- absorb pages 58-68 and emit
31 chunks of mangled Annexure A table text.

Prints a corpus report at the end. Read it: the numbers there are what tell you whether
ingestion did what it was supposed to.
"""

from __future__ import annotations

import sys
from collections import Counter

from src.config import (
    ANNEXURE_A_PAGES,
    ANNEXURE_B_PAGES,
    BM25_PATH,
    CHUNKS_PATH,
    EMBEDDING_MAX_TOKENS,
    FAISS_IDS_PATH,
    FAISS_PATH,
    NONPAYABLE_PAGES,
    PREMIUM_SECTIONS,
    PROSE_PAGES,
    RATE_CHART_PAGES,
    RAW_PDF,
)
from src.ingest import annexure_a
from src.ingest.chunk import build_chunks, build_flat_chunks, to_dicts, write_chunks
from src.ingest.extract import boilerplate_report, extract_pages
from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse import SparseRetriever


def main() -> int:
    if not RAW_PDF.exists():
        print(f"source PDF not found: {RAW_PDF}", file=sys.stderr)
        return 1

    print(f"reading {RAW_PDF.name}")
    report = boilerplate_report(RAW_PDF)
    print(
        f"  boilerplate stripped : {report['pct_removed']}% of chars "
        f"({len(report['furniture_lines'])} recurring lines, detected not hardcoded)"
    )

    # Chunk against the embedding model's real token window, not a char-count proxy. The proxy
    # leaks on numeric tables: "10,00,000" is one short string and many tokens. Loading the model
    # here rather than inside the chunker keeps src.ingest free of a torch dependency.
    dense = DenseRetriever()
    tokenizer = dense.model.tokenizer

    def fits(text: str) -> bool:
        return len(tokenizer.encode(text, add_special_tokens=True)) <= EMBEDDING_MAX_TOKENS

    # --- region: policy body (p.1-58) ---
    prose = to_dicts(build_chunks(extract_pages(RAW_PDF, keep=PROSE_PAGES), fits=fits))

    dropped_premium = [c for c in prose if c["section"] in PREMIUM_SECTIONS]
    prose = [c for c in prose if c["section"] not in PREMIUM_SECTIONS]

    # --- region: Annexure A (p.59-66), hand-structured, replaces the mangled table text ---
    matrix = annexure_a.as_chunks()

    # --- region: Annexure B (p.67-68) + non-payable lists (p.289-291) ---
    # Flat item tables with no clause headings, so build_chunks finds no blocks and returns
    # nothing. They need the flat path. These lists ARE answers -- Protect Benefit (4.3) pays for
    # exactly the Annexure B items, and "is a nebuliser kit covered" is a real claim-time question.
    annex_b = to_dicts(
        build_flat_chunks(
            extract_pages(RAW_PDF, keep=ANNEXURE_B_PAGES),
            section="annexure_b",
            title="Annexure B - Non-Medical Expenses covered under Protect Benefit (Section 4.3)",
            fits=fits,
        )
    )
    lists = to_dicts(
        build_flat_chunks(
            extract_pages(RAW_PDF, keep=NONPAYABLE_PAGES),
            section="non_payable_lists",
            title="Lists II, III and IV - Items not payable / subsumed into other charges",
            fits=fits,
        )
    )

    chunks = prose + matrix + annex_b + lists
    write_chunks(chunks, CHUNKS_PATH)

    print("\nregions")
    print(f"  policy body   p.{PROSE_PAGES.start}-{PROSE_PAGES.stop - 1:<3} -> {len(prose):>3} chunks")
    print(
        f"  Annexure A    p.{ANNEXURE_A_PAGES.start}-{ANNEXURE_A_PAGES.stop - 1:<3} -> {len(matrix):>3} chunks "
        f"(hand-structured plan matrix)"
    )
    print(f"  Annexure B    p.{ANNEXURE_B_PAGES.start}-{ANNEXURE_B_PAGES.stop - 1:<3} -> {len(annex_b):>3} chunks")
    print(f"  non-payable   p.{NONPAYABLE_PAGES.start}-{NONPAYABLE_PAGES.stop - 1} -> {len(lists):>3} chunks")
    print(
        f"  rate charts   p.{RATE_CHART_PAGES.start}-{RATE_CHART_PAGES.stop - 1} ->   0 chunks "
        f"(EXCLUDED: {len(RATE_CHART_PAGES)} pages of numeric tables)"
    )
    print(
        f"  premium calc  §{'/§'.join(sorted(PREMIUM_SECTIONS))}      ->   0 chunks "
        f"(EXCLUDED: {len(dropped_premium)} chunks of worked premium examples)"
    )

    print(f"\ncorpus: {len(chunks)} chunks")
    by_type = Counter(c["clause_type"] for c in chunks)
    print(f"  product-specific: {by_type['product_specific']}")
    print(f"  IRDAI-standard  : {by_type['irdai_standard']}")
    lens = sorted(len(c["text"]) for c in chunks)
    print(f"  chars: min {lens[0]}, median {lens[len(lens) // 2]}, max {lens[-1]}")

    # --- invariants. Both were real bugs, and both were silent. ---

    # 1. Duplicate ids. The id is the citation anchor AND the key of the id->chunk map, so a
    #    collision means retrieving one clause and citing another. Nine clauses once collided on
    #    "prose::10.1.i" because the roman numeral "i." nested in each exclusion read as a
    #    lettered clause -- including the sub-item of 10.1.r, maternity.
    ids = Counter(c["chunk_id"] for c in chunks)
    dupes = {k: n for k, n in ids.items() if n > 1}
    if dupes:
        print(f"\nFATAL: {len(dupes)} duplicate chunk ids: {dupes}", file=sys.stderr)
        return 1

    # 2. Chunks over the embedding window. SentenceTransformer truncates silently, so such a
    #    chunk is partly invisible to dense retrieval while looking correct everywhere else.
    over = [
        (c["chunk_id"], len(tokenizer.encode(c["text"], add_special_tokens=True)))
        for c in chunks
    ]
    over = [(cid, n) for cid, n in over if n > EMBEDDING_MAX_TOKENS]
    if over:
        print(
            f"\nFATAL: {len(over)} chunks exceed the {EMBEDDING_MAX_TOKENS}-token window and "
            f"would be silently truncated:",
            file=sys.stderr,
        )
        for cid, n in sorted(over, key=lambda x: -x[1])[:10]:
            print(f"  {cid}  {n} tokens", file=sys.stderr)
        return 1
    print(f"  all {len(chunks)} chunks fit the {EMBEDDING_MAX_TOKENS}-token embedding window")

    print("\nbuilding dense index (Sentence-BERT + FAISS)")
    dense.build(chunks)
    dense.save(FAISS_PATH, FAISS_IDS_PATH)
    print(f"  {dense.index.ntotal} vectors, dim {dense.index.d}")

    print("building sparse index (BM25)")
    sparse = SparseRetriever()
    sparse.build(chunks)
    sparse.save(BM25_PATH)
    print(f"  {len(sparse.chunk_ids)} documents")

    print("\nwrote data/processed/chunks.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
