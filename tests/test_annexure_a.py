"""Verify the hand-transcribed Annexure A against the source PDF.

src/ingest/annexure_a.py is typed by hand, because the PDF's text layer destroys the plan
columns. Hand-transcription buys correct plan->value bindings and introduces a new risk: a
typo here becomes a confidently wrong answer about money, with a citation attached, and
nothing downstream would ever catch it.

So every transcribed value is checked to actually occur on the source page it claims. This is
the test that makes the hand-transcription trustworthy rather than merely convenient.
"""

from __future__ import annotations

import re

import pytest
from pypdf import PdfReader

from src.config import ANNEXURE_A_PAGES, PLANS, RAW_PDF
from src.ingest import annexure_a


@pytest.fixture(scope="module")
def page_text() -> dict[int, str]:
    """Normalised text of each Annexure A page, whitespace collapsed."""
    reader = PdfReader(str(RAW_PDF))
    return {
        n: _normalise_digits(re.sub(r"\s+", " ", reader.pages[n - 1].extract_text() or "").lower())
        for n in ANNEXURE_A_PAGES
    }


def _normalise_digits(text: str) -> str:
    """Canonicalise a string so that only its VALUE is compared, not its typesetting.

    Two kinds of noise stand between the transcription and the source, and neither is an error:

    1. Digit grouping. Page 62 writes "500,000" (Western) and "4800" (none) in adjacent cells;
       the transcription uses Indian grouping ("5,00,000", "4,800"). Same numbers.

    2. Line wrapping THROUGH a token. Page 60 breaks the sum-insured list mid-number, as
       "5/10/15/20/25/5" / "0 Lakhs", which collapses to "5/10/15/20/25/5 0 Lakhs" -- a space
       inside a numeral. No contiguous substring of the real value can survive that, so all
       whitespace is removed before comparing.

    What remains after this is the number itself, which is the thing a typo would corrupt and
    the thing the test exists to protect.
    """
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    return re.sub(r"\s+", "", text)


def _searchable(value: str) -> list[str]:
    """The distinctive fragments of a cell value that must appear in the source.

    The PDF wraps cells mid-phrase, so the full string rarely survives contiguously. We check the
    parts that carry the meaning and that a typo would corrupt: the numbers, and the verdict.
    """
    v = _normalise_digits(value.lower())
    frags = re.findall(r"\d[\d./]*%?", v)

    for phrase in ("not covered", "at actuals", "unlimited times", "optional", "in india", "global"):
        if phrase in v:
            frags.append(phrase)
    return frags


def test_every_row_covers_all_eight_plans():
    for row in annexure_a.ROWS:
        assert set(row.values) == set(PLANS), (
            f"Section {row.section} ({row.benefit}) does not have a value for every plan. "
            f"Missing: {set(PLANS) - set(row.values)}"
        )


def test_pages_are_inside_annexure_a():
    for row in annexure_a.ROWS:
        assert row.page in ANNEXURE_A_PAGES, (
            f"Section {row.section} claims page {row.page}, outside Annexure A "
            f"({ANNEXURE_A_PAGES.start}-{ANNEXURE_A_PAGES.stop - 1})"
        )


@pytest.mark.parametrize("row", annexure_a.ROWS, ids=lambda r: f"{r.section}-{r.benefit[:24]}")
def test_transcribed_values_appear_on_their_source_page(row, page_text):
    """Every number and verdict in a transcribed cell must occur on the page it cites."""
    source = page_text[row.page]

    for plan, value in row.values.items():
        for frag in _searchable(value):
            assert frag in source, (
                f"Section {row.section} / {plan}: transcribed {value!r} but fragment {frag!r} "
                f"does not appear on source page {row.page}. Either the transcription is wrong "
                f"or the page reference is."
            )


def test_chunks_bind_every_plan_to_its_value():
    """The whole point: a chunk must state each plan's value next to that plan's name.

    This is the property the raw text layer loses and the reason this module exists. If it ever
    stops holding, plan-specific questions silently start returning another plan's answer.
    """
    for chunk in annexure_a.as_chunks():
        row = next(r for r in annexure_a.ROWS if r.section == chunk["section"])
        for plan in PLANS:
            assert f"- {plan}: {row.values[plan]}" in chunk["text"], (
                f"chunk {chunk['chunk_id']} does not bind {plan} to its value"
            )


def test_secure_benefit_distinguishes_the_two_that_matter():
    """A regression guard on the single most expensive confusion in the document.

    Optima Secure gives 100% of Base Sum Insured; Optima Super Secure gives 200%. The plan names
    differ by one token and the values differ by crores. Getting these backwards is the worst
    factual error this system could make.
    """
    row = next(r for r in annexure_a.ROWS if r.section == "4.5")
    assert row.values["Optima Secure"] == "Equal to 100% of Base Sum Insured"
    assert row.values["Optima Super Secure"] == "Equal to 200% of Base Sum Insured"
    assert row.values["Optima Suraksha"] == annexure_a.NOT_COVERED
