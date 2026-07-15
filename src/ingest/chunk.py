"""Clause-boundary chunking.

Fixed-length windows are wrong for legal text. A 512-token window cutting through the
middle of Section 10.1.b leaves half the waiting-period rule in one chunk and half in the
next, and neither chunk answers the question. Worse, a chunk that contains "shall be
excluded until the expiry of 24 months" but not the sentence establishing WHAT is excluded
is actively dangerous: it retrieves well and reads authoritatively while being unusable.

So chunks follow the document's own clause numbering. One chunk per leaf clause. The
document numbers its sections (3.1, 3.1.1, 10.1.a ...) and cites them internally, which
means clause numbers double as citation anchors — the same string that identifies a chunk
is the string a user needs to look it up in the PDF.

Two special cases:

1. Exclusions (10.1, 10.2). Each lettered sub-clause is independently citable and carries
   its own IRDAI code (Excl01..Excl18). "Is maternity covered" is answered by 10.1.r alone,
   so 10.1.r is its own chunk. Splitting here is what makes exclusion citations precise.

2. Over-long clauses. Where a single clause exceeds MAX_CHUNK_CHARS it is split at
   sub-clause boundaries (a., b., i., ii.), never mid-sentence, and every part keeps the
   parent's section number and heading so no fragment loses its context.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from src.config import MAX_CHUNK_CHARS, MIN_CHUNK_CHARS
from src.ingest.extract import Page

# A heading is a dotted clause number followed by a title:  "3.", "3.1", "3.1.1.", "10.2."
# Requiring a dot (trailing or internal) separates real headings from the numbered list rows
# in the critical-illness and consumables tables ("1 Cancer of specified severity",
# "35 Oxygen Cylinder"), which have no dot.
#
# That alone is NOT enough. Numbered lists INSIDE a clause also carry dots -- Section 4.12
# contains "1. Modification of PED waiting period from 36 months...". Matching those as
# top-level headings produced ten different chunks all claiming to be "Section 1", whose IDs
# then collided and silently overwrote each other. So heading acceptance is additionally
# gated on sequence (see _parse_blocks): the document numbers its sections 1..17 in order,
# and a "1." appearing after we have already passed section 12 is a list item, not a section.
HEADING = re.compile(r"^\s*(\d{1,2}(?:\.\d{1,2})+\.?|\d{1,2}\.)\s+(\S.{2,110})$")

# Lettered sub-clause inside the exclusions sections: "a. Pre-Existing Diseases - Code - Excl01"
#
# Single-letter roman numerals are the trap here. Section 10.1 runs a..r, and "i." is BOTH a
# legitimate exclusion letter (10.1.i, Excl09 Hazardous Sports) AND the first item of the roman
# sub-list nested inside every other exclusion. Matching naively made nine different clauses all
# claim the id "10.1.i" -- including the sub-item of 10.1.r, the maternity clause.
#
# The fix is sequence, not pattern: a real lettered clause is always the SUCCESSOR of the last
# one accepted (a->b->...->h->i->j->...->r), whereas nested roman lists restart at "i." every
# time. See _split_exclusions.
LETTERED = re.compile(r"^\s*([a-z])\.\s+(\S.*)$")

# IRDAI standard exclusion code, e.g. "Excl01". Its presence marks regulator-mandated wording.
EXCL_CODE = re.compile(r"Excl(\d{2})", re.IGNORECASE)

# Sections whose lettered sub-clauses are independently citable.
EXCLUSION_SECTIONS = {"10.1", "10.2"}

# Sections that reproduce IRDAI-standardised wording rather than product-specific terms.
# Section 12 is literally titled "Standard General Terms and Clauses"; 10.1 is the
# standard exclusions with their Excl codes. Tagging these lets the answer layer attribute
# correctly ("this is a regulator-standard clause, not specific to this product") without
# changing whether they are indexed. They are indexed: people genuinely ask about them.
IRDAI_STANDARD_PREFIXES = ("10.1", "12.")


@dataclass
class Chunk:
    chunk_id: str
    section: str
    section_title: str
    text: str
    page_start: int
    page_end: int
    clause_type: str  # product_specific | irdai_standard
    source: str  # prose | annexure_a
    plan_scope: str  # all | plan_specific


@dataclass
class _Block:
    """A heading and the raw lines beneath it, before splitting."""

    section: str
    title: str
    lines: list[str]
    page_start: int
    page_end: int


def _classify(section: str) -> str:
    return (
        "irdai_standard"
        if any(section.startswith(p) for p in IRDAI_STANDARD_PREFIXES)
        else "product_specific"
    )


def _is_heading(section: str, title: str, top: int, seen: set[str]) -> bool:
    """Decide whether a dotted-number line is a real clause heading or something else.

    Three things masquerade as headings in the extracted text, and each needs its own guard.

    1. Numbered list items inside a clause. Section 4.12 contains "1. Modification of PED
       waiting period from 36 months...". Guard: the document numbers its sections 1..17
       strictly in order, so a top-level "N." is real only if it advances the count, and a
       sub-heading "N.M" is real only if N is the section we are currently inside.

    2. Wrapped cross-references. Clause 3.5 ends a line with "...under Section" and begins the
       next with "3.1 (Hospitalization Expenses). Such expenses shall be...". That reads as a
       heading for section 3.1 with a parenthetical title, and it silently truncated 3.5 and
       3.6 -- the pre- and post-hospitalization clauses -- misfiling their bodies under a
       duplicate "3.1". Guards: section numbers are unique in a legal document, so a number we
       have already issued is a reference, not a heading; and a real title never opens with a
       parenthesis or a lowercase word.

    3. Table rows, which are mostly digits.
    """
    if not title or title[0] == "(" or title[0].islower():
        return False

    digits = sum(c.isdigit() for c in title)
    if digits > len(title) * 0.4:
        return False

    if section in seen:
        return False

    parts = section.split(".")
    major = int(parts[0])

    if len(parts) == 1:
        return major == top + 1
    return major == top


def _parse_blocks(pages: list[Page]) -> list[_Block]:
    """Walk the page stream and cut it at every clause heading."""
    blocks: list[_Block] = []
    current: _Block | None = None
    top = 0  # highest top-level section number accepted so far
    seen: set[str] = set()  # section numbers already issued; a repeat is a cross-reference

    for page in pages:
        for line in page.text.split("\n"):
            m = HEADING.match(line)
            if m:
                section = m.group(1).rstrip(".")
                title = m.group(2).strip()

                if not _is_heading(section, title, top, seen):
                    # Not a heading: body text (a numbered list item, a wrapped cross-reference,
                    # a table row). Keep it as part of the clause we are inside.
                    if current:
                        current.lines.append(line)
                        current.page_end = page.number
                    continue

                seen.add(section)
                if "." not in section:
                    top = int(section)

                if current:
                    blocks.append(current)
                current = _Block(
                    section=section,
                    title=title,
                    lines=[],
                    page_start=page.number,
                    page_end=page.number,
                )
            elif current:
                current.lines.append(line)
                current.page_end = page.number

    if current:
        blocks.append(current)
    return blocks


def _split_exclusions(block: _Block, fits: Fits) -> list[Chunk]:
    """Split 10.1 / 10.2 into one chunk per lettered sub-clause.

    Each carries its own Excl code and answers a distinct question, so each gets its own
    citation anchor. The parent heading is prepended to every part, because "Maternity:
    Code Excl18" is only meaningful once you know it sits under "Exclusions".
    """
    chunks: list[Chunk] = []
    preamble: list[str] = []
    letter: str | None = None
    buf: list[str] = []
    page_start = block.page_start
    seen_pages = block.page_start

    def flush(ltr: str | None, lines: list[str], p_start: int, p_end: int) -> None:
        body = "\n".join(lines).strip()
        if not body or ltr is None:
            return
        section = f"{block.section}.{ltr}"
        code = EXCL_CODE.search(body)
        header = f"Section {section} - {block.title}"
        if code:
            header += f" (Code Excl{code.group(1)})"

        # Give every exclusion an explicit statement of what an exclusion IS.
        #
        # Some of these clauses are tiny. Section 10.2.l is the whole of "Treatment taken on
        # outpatient basis." -- 77 characters including the header. Standing alone, that text
        # never says "not covered" anywhere, so a user asking "are my regular doctor visits
        # covered?" gets no match from either retriever: no shared terms for BM25, and an
        # embedding that encodes a bare noun phrase rather than a denial.
        #
        # The denial lives in the parent's preamble, which every sub-clause inherits but none
        # repeats. Restating it per chunk is what a lawyer reading the section already does in
        # their head, and it costs ~25 tokens.
        header += (
            "\nThis is an EXCLUSION. The Company shall NOT make payment for, and is NOT liable "
            "for, any claim caused by, arising from or attributable to the following:"
        )

        # Exclusion clauses can be long -- 10.1.b carries the full specified-disease and
        # surgical-procedure tables -- so they need the same window-safe splitting as prose.
        #
        # Size the text that is ACTUALLY EMBEDDED, which is header + body, not body alone. The
        # header is ~15 tokens and sizing without it put seven chunks a few tokens over the
        # window -- close enough to look like a rounding error, but truncation is truncation.
        for i, part in enumerate(_split_long(body, _with_header(header, fits))):
            suffix = f"#{i + 1}" if i else ""
            cont = " (continued)" if i else ""
            chunks.append(
                Chunk(
                    chunk_id=f"prose::{section}{suffix}",
                    section=section,
                    section_title=block.title,
                    text=f"{header}{cont}\n{part}",
                    page_start=p_start,
                    page_end=p_end,
                    clause_type=_classify(block.section),
                    source="prose",
                    plan_scope="all",
                )
            )

    for line in block.lines:
        m = LETTERED.match(line)
        # Accept a lettered clause only if it is the SUCCESSOR of the last one accepted.
        # Nested roman sub-lists restart at "i." inside every clause, so without this the
        # "i." of 10.1.r (maternity) is indistinguishable from clause 10.1.i (hazardous
        # sports). Sequence tells them apart; the regex alone cannot.
        expected = "a" if letter is None else chr(ord(letter) + 1)
        if m and m.group(1) == expected:
            flush(letter, buf, page_start, seen_pages)
            letter = m.group(1)
            buf = [line.strip()]
            page_start = seen_pages
        elif letter is None:
            preamble.append(line)
        else:
            buf.append(line)
    flush(letter, buf, page_start, seen_pages)

    # The preamble ("All the Waiting Periods and exclusions listed below shall be applicable
    # individually for each Insured Person...") is a real rule and must survive as its own chunk.
    body = "\n".join(preamble).strip()
    if len(body) >= MIN_CHUNK_CHARS:
        chunks.insert(
            0,
            Chunk(
                chunk_id=f"prose::{block.section}",
                section=block.section,
                section_title=block.title,
                text=f"Section {block.section} - {block.title}\n{body}",
                page_start=block.page_start,
                page_end=block.page_end,
                clause_type=_classify(block.section),
                source="prose",
                plan_scope="all",
            ),
        )
    return chunks


# Boundaries we are willing to split an over-long clause on, best first. The cascade matters:
# splitting at a sub-clause boundary preserves a complete legal unit, splitting at a sentence
# does not, so we only fall back when the better boundary does not exist in the text.
_SUBCLAUSE = re.compile(r"^\s*(?:[a-z]\.|[ivx]{1,4}\.|\d{1,2}\.)\s+\S")
_SENTENCE = re.compile(r"(?<=[.;:])\s+")

# A size predicate: does this text fit in one chunk?
#
# Chars are only ever a proxy for what actually constrains us, which is the embedding model's
# 384-TOKEN window. The proxy leaks badly on the numeric tables: "10,00,000" is one short
# string but many tokens, so Section 2.1 and Section 17 (the premium and claim-payout
# illustrations) blew the token window while sitting comfortably under any char limit.
#
# So callers may pass a real tokenizer-backed predicate. build_index.py does exactly that,
# which is why the chunker is correct rather than approximately correct. The char default
# keeps src.ingest importable and testable without pulling in torch.
Fits = Callable[[str], bool]


def _char_fits(limit: int = MAX_CHUNK_CHARS) -> Fits:
    return lambda t: len(t) <= limit


def _with_header(header: str, fits: Fits) -> Fits:
    """Budget for the prefix that gets prepended to every chunk before embedding.

    Budgets against the WORST-CASE prefix -- header plus the " (continued)" marker that every
    part after the first carries -- rather than the bare header. Budgeting for the bare header
    left the continued parts a few tokens over the window, which is the same silent-truncation
    bug in miniature: small enough to look like rounding, still a clause with its tail cut off.
    """
    worst_case = f"{header} (continued)"
    return lambda body: fits(f"{worst_case}\n{body}")


def _pack(units: list[str], joiner: str, fits: Fits) -> list[str]:
    """Greedily pack units into parts, each of which satisfies `fits`."""
    parts: list[str] = []
    buf: list[str] = []
    for u in units:
        candidate = joiner.join([*buf, u])
        if buf and not fits(candidate):
            parts.append(joiner.join(buf))
            buf = [u]
        else:
            buf.append(u)
    if buf:
        parts.append(joiner.join(buf))
    return [p.strip() for p in parts if p.strip()]


def _hard_wrap(text: str, fits: Fits) -> list[str]:
    """Last resort: bisect until every piece fits. Only reached by a single unbreakable run."""
    if fits(text) or len(text) <= 1:
        return [text]
    mid = len(text) // 2
    return _hard_wrap(text[:mid], fits) + _hard_wrap(text[mid:], fits)


def _split_long(body: str, fits: Fits | None = None) -> list[str]:
    """Split an over-long clause, preferring the most semantically intact boundary.

    This used to split only at sub-clause markers, and silently gave up when a clause had none,
    leaving chunks of up to 4,245 chars. That was not cosmetic. SentenceTransformer TRUNCATES
    past its 384-token window without warning, so the tail of every oversized clause was
    invisible to dense retrieval while still present in the chunk text. That is the worst
    failure mode available: the citation looks right, a human reading chunks.jsonl sees the
    content, and the retriever simply never encoded it. Section 13.2 (Utilization of Sum
    Insured) was losing 48% of its body this way.

    BM25 has no such window and was reading the full text throughout, which is a quiet argument
    for keeping the hybrid honest: the two halves fail differently.
    """
    fits = fits or _char_fits()
    if fits(body):
        return [body]

    # 1. sub-clause boundaries (a., i., 1.) -- keeps whole legal units intact
    groups: list[list[str]] = [[]]
    for line in body.split("\n"):
        if _SUBCLAUSE.match(line) and groups[-1]:
            groups.append([])
        groups[-1].append(line)
    units = ["\n".join(g) for g in groups if "\n".join(g).strip()]
    parts = _pack(units, "\n", fits) if len(units) > 1 else [body]

    # 2. still too long (one enormous sub-clause, or a clause with no markers at all -- the
    #    51-row critical illness table in 4.8, the premium illustration in 17): try sentences.
    out: list[str] = []
    for part in parts:
        if fits(part):
            out.append(part)
            continue
        packed = _pack(_SENTENCE.split(part.replace("\n", " ")), " ", fits)
        for p in packed:
            out.extend([p] if fits(p) else _hard_wrap(p, fits))
    return out


def build_chunks(pages: list[Page], fits: Fits | None = None) -> list[Chunk]:
    """Cut pages into clause chunks.

    `fits` decides whether a piece of text is small enough to stand as one chunk. Pass a
    tokenizer-backed predicate (build_index.py does) so splitting respects the embedding
    model's real token window rather than a char-count proxy that leaks on numeric tables.
    """
    fits = fits or _char_fits()
    blocks = _parse_blocks(pages)
    chunks: list[Chunk] = []

    for block in blocks:
        if block.section in EXCLUSION_SECTIONS:
            chunks.extend(_split_exclusions(block, fits))
            continue

        body = "\n".join(block.lines).strip()
        if len(body) < MIN_CHUNK_CHARS:
            # Too short to stand alone. Almost always a parent heading whose only content is
            # its children (e.g. "3. Base Coverage"). Keep it if it carries a real rule,
            # drop it if it is a bare signpost.
            if not body:
                continue

        header = f"Section {block.section} - {block.title}"
        parts = _split_long(body, _with_header(header, fits))

        for i, part in enumerate(parts):
            suffix = f"#{i + 1}" if len(parts) > 1 else ""
            cont = " (continued)" if i > 0 else ""
            chunks.append(
                Chunk(
                    chunk_id=f"prose::{block.section}{suffix}",
                    section=block.section,
                    section_title=block.title,
                    text=f"{header}{cont}\n{part}",
                    page_start=block.page_start,
                    page_end=block.page_end,
                    clause_type=_classify(block.section),
                    source="prose",
                    plan_scope="all",
                )
            )

    return chunks


def build_flat_chunks(
    pages: list[Page],
    section: str,
    title: str,
    fits: Fits | None = None,
) -> list[Chunk]:
    """Chunk a region that has no clause structure at all.

    Annexure B (the non-medical expenses list) and Lists II/III/IV (non-payable items) are flat
    two-column item tables: "1 Baby Food 35 Oxygen Cylinder". There are no numbered headings, so
    build_chunks finds no blocks and returns nothing -- which is exactly what happened, silently,
    and left Protect Benefit questions with no item list to retrieve against.

    These regions carry real answers ("is a nebuliser kit covered?"), and what makes them
    findable is BM25 matching the item name as a literal token. So they are packed into
    window-sized chunks with the region title on each, and nothing more clever is attempted.
    """
    fits = fits or _char_fits()
    if not pages:
        return []

    header = f"{title} (Section {section})"
    body = "\n".join(p.text for p in pages)
    page_start, page_end = pages[0].number, pages[-1].number

    return [
        Chunk(
            chunk_id=f"prose::{section}{f'#{i + 1}' if i else ''}",
            section=section,
            section_title=title,
            text=f"{header}\n{part}",
            page_start=page_start,
            page_end=page_end,
            clause_type="product_specific",
            source="prose",
            plan_scope="all",
        )
        for i, part in enumerate(_split_long(body, _with_header(header, fits)))
    ]


def write_chunks(chunks: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")


def load_chunks(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def to_dicts(chunks: list[Chunk]) -> list[dict]:
    return [asdict(c) for c in chunks]
