"""Regression tests for the chunker.

Every test here corresponds to a bug that actually shipped into an index and was caught only
by inspecting the corpus. They are silent failures by nature: each one produced a plausible
chunk with a plausible citation and quietly wrong content, which is precisely the failure mode
a citation-bearing RAG system must not have.
"""

from __future__ import annotations

import pytest

from src.config import EMBEDDING_MAX_TOKENS, PROSE_PAGES, RAW_PDF
from src.ingest.chunk import build_chunks
from src.ingest.extract import extract_pages


@pytest.fixture(scope="module")
def chunks():
    return build_chunks(extract_pages(RAW_PDF, keep=PROSE_PAGES))


@pytest.fixture(scope="module")
def by_section(chunks):
    out: dict[str, list] = {}
    for c in chunks:
        out.setdefault(c.section, []).append(c)
    return out


def test_chunk_ids_are_unique(chunks):
    """The id is the citation anchor AND the key of the id->chunk map.

    A collision means retrieving one clause and citing another. Nine different clauses once
    collided on "prose::10.1.i", because the roman numeral "i." nested inside every exclusion
    was read as lettered clause "i" -- among them the sub-item of 10.1.r, maternity.
    """
    ids = [c.chunk_id for c in chunks]
    dupes = {i for i in ids if ids.count(i) > 1}
    assert not dupes, f"duplicate chunk ids: {dupes}"


def test_all_eighteen_standard_exclusions_are_present(by_section):
    """Section 10.1 runs a..r (Excl01..Excl18). Every one must survive as its own citable chunk."""
    expected = {f"10.1.{c}" for c in "abcdefghijklmnopqr"}
    missing = expected - set(by_section)
    assert not missing, f"missing standard exclusions: {sorted(missing)}"


def test_maternity_clause_is_intact(by_section):
    """10.1.r is the highest-consequence clause in the document.

    "Is maternity covered?" is a question people genuinely ask, and the answer is no. This clause
    was previously shredded: its first roman sub-item was hoisted out into a bogus "10.1.i".
    """
    text = " ".join(c.text for c in by_section["10.1.r"]).lower()
    assert "maternity" in text
    assert "excl18" in text
    assert "childbirth" in text
    assert "ectopic pregnancy" in text  # the sole exception, and dropping it inverts the answer


def test_no_boilerplate_leaks_into_chunks(chunks):
    """The registration footer is typeset in two different line wraps.

    A frequency threshold set between them stripped one and left the other, and the surviving
    layout was the one used on the prose pages. Product UIN codes ended up inside Section 3.1.1.
    """
    for c in chunks:
        assert "HDFHLI" not in c.text, f"UIN boilerplate leaked into {c.chunk_id}"
        assert "IRDAI Reg. No" not in c.text, f"registration footer leaked into {c.chunk_id}"
        assert "Leela Business Park" not in c.text, f"address footer leaked into {c.chunk_id}"


def test_cross_references_do_not_create_headings(by_section):
    """Clause 3.5 wraps a cross-reference so "3.1 (Hospitalization Expenses)..." starts a line.

    Read as a heading, that truncated 3.5 and 3.6 -- the pre- and post-hospitalization clauses --
    and misfiled their bodies under a duplicate "3.1".
    """
    pre = " ".join(c.text for c in by_section["3.5"])
    post = " ".join(c.text for c in by_section["3.6"])
    assert "60 days" in pre, "Section 3.5 lost its 60-day pre-hospitalization window"
    assert "180 days" in post, "Section 3.6 lost its 180-day post-hospitalization window"


def test_numbered_list_items_do_not_become_sections(by_section):
    """Section 4.12 contains "1. Modification of PED waiting period...".

    Matched as a heading, that (and nine like it) produced chunks all claiming to be "Section 1"
    with ten different titles. Section 1 is Eligibility and nothing else -- it may be SPLIT across
    several chunks by the token window, which is fine, but every one of them is Eligibility.
    """
    titles = {c.section_title.lower() for c in by_section["1"]}
    assert titles == {"eligibility"}, f"section 1 has conflicting titles: {titles}"

    # The list items that used to masquerade as sections live in 4.12; it must still be one clause.
    assert {c.section_title for c in by_section["4.12"]} == {"PED wait period modification"}


def test_exclusion_chunks_state_that_they_are_exclusions(by_section):
    """Some exclusions are a single short sentence.

    10.2.l is the whole of "Treatment taken on outpatient basis" -- text that never says "not
    covered" anywhere. The denial lives in the parent preamble, so each chunk restates it;
    otherwise neither retriever can connect "are my doctor visits covered?" to it.
    """
    text = by_section["10.2.l"][0].text.lower()
    assert "exclusion" in text
    assert "not" in text and "payment" in text


def test_no_chunk_exceeds_the_embedding_window():
    """SentenceTransformer truncates past its 384-token window silently.

    An over-window chunk looks correct in chunks.jsonl and in its citation while being partly
    invisible to dense retrieval. Section 13.2 was losing 48% of its body this way.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    tok = model.tokenizer

    def fits(t: str) -> bool:
        return len(tok.encode(t, add_special_tokens=True)) <= EMBEDDING_MAX_TOKENS

    built = build_chunks(extract_pages(RAW_PDF, keep=PROSE_PAGES), fits=fits)
    over = [c.chunk_id for c in built if not fits(c.text)]
    assert not over, f"chunks exceed the {EMBEDDING_MAX_TOKENS}-token window: {over}"
