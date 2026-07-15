"""PDF text extraction with data-driven boilerplate removal.

The source PDF repeats an IRDAI registration block and a "Page N of 291" footer on
every one of its 291 pages. That furniture is 37% of the raw extracted characters.
Left in place it lands in every chunk, where it dominates BM25 term statistics and
pulls every embedding toward the same point in vector space.

Rather than hardcode the strings, we detect them: any line appearing on more than
BOILERPLATE_PAGE_FRACTION of pages is furniture. This generalises to other insurers'
documents, which is what makes the pipeline reusable beyond this one file.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from src.config import BOILERPLATE_PAGE_FRACTION


@dataclass
class Page:
    number: int  # 1-indexed, matches the PDF's own page numbering
    text: str


def _normalise(line: str) -> str:
    """Collapse whitespace so near-identical furniture lines compare equal."""
    return re.sub(r"\s+", " ", line).strip()


def find_boilerplate(pages: list[list[str]], threshold: float) -> set[str]:
    """Return normalised lines that recur on `threshold` or more of the pages.

    Page-number footers ("Page 12 of 291") are near-unique per page and so would
    survive a pure frequency filter. We fold their digits to a placeholder before
    counting, which makes them collapse into one high-frequency form.
    """
    counts: Counter[str] = Counter()
    for lines in pages:
        # A line repeated within one page still only counts once for that page.
        counts.update({_digit_fold(_normalise(ln)) for ln in lines if ln.strip()})

    cutoff = max(2, int(len(pages) * threshold))
    return {line for line, n in counts.items() if n >= cutoff}


def _digit_fold(line: str) -> str:
    return re.sub(r"\d+", "#", line)


def extract_pages(pdf_path: Path, keep: range | None = None) -> list[Page]:
    """Extract text per page, stripped of recurring header/footer furniture.

    `keep` selects a page region. Boilerplate frequency is always computed over the WHOLE
    document, then the region is selected -- otherwise a narrow region would have too few
    pages for the frequency threshold to identify the furniture at all.
    """
    reader = PdfReader(str(pdf_path))
    raw = [(i, (p.extract_text() or "").split("\n")) for i, p in enumerate(reader.pages, 1)]

    furniture = find_boilerplate([lines for _, lines in raw], BOILERPLATE_PAGE_FRACTION)

    pages: list[Page] = []
    for number, lines in raw:
        if keep is not None and number not in keep:
            continue
        kept = [
            ln.rstrip()
            for ln in lines
            if _digit_fold(_normalise(ln)) not in furniture and ln.strip()
        ]
        if kept:
            pages.append(Page(number=number, text="\n".join(kept)))
    return pages


def boilerplate_report(pdf_path: Path) -> dict:
    """Diagnostics for what the stripper removed. Used by tests and the ingest CLI."""
    reader = PdfReader(str(pdf_path))
    raw = [(i, (p.extract_text() or "").split("\n")) for i, p in enumerate(reader.pages, 1)]

    furniture = find_boilerplate([lines for _, lines in raw], BOILERPLATE_PAGE_FRACTION)
    total = sum(len(ln) for _, lines in raw for ln in lines)
    removed = sum(
        len(ln)
        for _, lines in raw
        for ln in lines
        if _digit_fold(_normalise(ln)) in furniture
    )
    return {
        "pages_considered": len(raw),
        "furniture_lines": sorted(furniture),
        "chars_total": total,
        "chars_removed": removed,
        "pct_removed": round(removed / total * 100, 1) if total else 0.0,
    }
