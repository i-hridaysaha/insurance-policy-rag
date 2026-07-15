"""Annexure A - Schedule of Benefits, hand-structured.

WHY THIS FILE EXISTS
--------------------
Annexure A (source pages 60-66) is the authoritative plan-comparison matrix: eight plan
columns against ~20 benefit rows. It is the only place in the document that says what a
given benefit is worth under a given plan.

It is also the one table that naive PDF text extraction destroys. The columns collapse
into a vertical stream, so the raw text layer reads:

    4.5 Secure Benefit Not Covered Equal to 100% of
    Base sum insured
    Equal to 200% of Base
    sum insured
    ...

Every value survives, but its association with a plan does not. Chunk that and a
retriever will happily return "Secure Benefit: Not Covered" for a question about Optima
Super Secure, where the true answer is 200% of Base Sum Insured. That is a confidently
wrong answer about money, which is the exact failure this whole system exists to prevent.

So the matrix is transcribed by hand from the source pages and emitted as one chunk per
benefit row, with each plan's value written out explicitly. Retrieval then cannot lose
the plan->value binding, because the binding is in the chunk text itself.

The transcription is asserted against the PDF's own text in tests/test_annexure_a.py:
every value below must appear on its stated source page. That guards against a typo here
silently becoming a wrong answer to a user.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.config import PLANS

NOT_COVERED = "Not Covered"


@dataclass(frozen=True)
class BenefitRow:
    section: str
    benefit: str
    page: int
    # plan name -> value, verbatim from the Annexure A cell
    values: dict[str, str]
    footnote: str | None = None


# Column order in the source table, for reference:
#   Optima Suraksha | Optima Secure | Optima Super Secure | Optima Secure Global
#   | Optima Secure Global Plus | Optima Select | Optima Lite | Optima Secure +

ROWS: list[BenefitRow] = [
    BenefitRow(
        section="2",
        benefit="Base Sum Insured options",
        page=60,
        values={
            "Optima Suraksha": "5/10/15/20/25/50 Lakhs",
            "Optima Secure": "5/10/15/20/25/50/100/200 Lakhs",
            "Optima Super Secure": "10/15/20/25/50/100/200 Lakhs",
            "Optima Secure Global": "100/200 Lakhs",
            "Optima Secure Global Plus": "25/50/75/100/200 Lakhs",
            "Optima Select": "5/7.5/10/15/20/25 Lakhs",
            "Optima Lite": "5/7.5 Lakhs",
            "Optima Secure +": "5/10/15/20/25/50/100/200 Lakhs",
        },
    ),
    BenefitRow(
        section="geography",
        benefit="Geography of cover",
        page=60,
        values={
            "Optima Suraksha": "India only",
            "Optima Secure": "India only",
            "Optima Super Secure": "India only",
            "Optima Secure Global": "Worldwide including India",
            "Optima Secure Global Plus": "Worldwide including India",
            "Optima Select": "India only",
            "Optima Lite": "India only",
            "Optima Secure +": "India only",
        },
    ),
    BenefitRow(
        section="3.1.a",
        benefit="Room Rent",
        page=60,
        values={
            "Optima Suraksha": "At Actuals",
            "Optima Secure": "At Actuals",
            "Optima Super Secure": "At Actuals",
            "Optima Secure Global": "At Actuals",
            "Optima Secure Global Plus": "At Actuals",
            "Optima Select": "Upto Single Private room",
            "Optima Lite": "Upto 1% of base sum insured per day",
            "Optima Secure +": "At Actuals",
        },
        footnote=(
            "If admitted to a room above the eligible category, a proportionate deduction "
            "applies to room rent and to all associated medical expenses (Section 3.1.1 Note iii)."
        ),
    ),
    BenefitRow(
        section="3.1.b",
        benefit="ICU / ICCU charges",
        page=60,
        values={
            "Optima Suraksha": "At Actuals",
            "Optima Secure": "At Actuals",
            "Optima Super Secure": "At Actuals",
            "Optima Secure Global": "At Actuals",
            "Optima Secure Global Plus": "At Actuals",
            "Optima Select": "At Actuals",
            "Optima Lite": "Upto 2% of base sum insured per day",
            "Optima Secure +": "At Actuals",
        },
    ),
    BenefitRow(
        section="3.5",
        benefit="Pre-Hospitalization Expenses",
        page=61,
        values={
            "Optima Suraksha": "60 days",
            "Optima Secure": "60 days",
            "Optima Super Secure": "60 days",
            "Optima Secure Global": "60 days (India only)",
            "Optima Secure Global Plus": "60 days",
            "Optima Select": "60 days",
            "Optima Lite": "30 days",
            "Optima Secure +": "60 days",
        },
    ),
    BenefitRow(
        section="3.6",
        benefit="Post-Hospitalization Expenses",
        page=61,
        values={
            "Optima Suraksha": "180 days",
            "Optima Secure": "180 days",
            "Optima Super Secure": "180 days",
            "Optima Secure Global": "180 days (India only)",
            "Optima Secure Global Plus": "180 days",
            "Optima Select": "180 days",
            "Optima Lite": "60 days",
            "Optima Secure +": "180 days",
        },
    ),
    BenefitRow(
        section="3.8",
        benefit="Cumulative Bonus",
        page=62,
        values={
            "Optima Suraksha": "10% of Base Sum Insured per policy year, maximum upto 100%, irrespective of claims",
            "Optima Secure": NOT_COVERED,
            "Optima Super Secure": NOT_COVERED,
            "Optima Secure Global": NOT_COVERED,
            "Optima Secure Global Plus": NOT_COVERED,
            "Optima Select": "25% of Base Sum Insured per policy year, maximum upto 100%, irrespective of claims",
            "Optima Lite": "10% of Base Sum Insured per policy year, maximum upto 100%, irrespective of claims",
            "Optima Secure +": NOT_COVERED,
        },
    ),
    BenefitRow(
        section="4.1",
        benefit="Emergency Air Ambulance",
        page=62,
        values={
            "Optima Suraksha": "Covered up to Rs 5,00,000",
            "Optima Secure": "Covered up to Rs 5,00,000",
            "Optima Super Secure": "Covered up to Rs 5,00,000",
            "Optima Secure Global": "Covered up to Rs 5,00,000",
            "Optima Secure Global Plus": "Covered up to Rs 5,00,000",
            "Optima Select": NOT_COVERED,
            "Optima Lite": "Covered up to Rs 5,00,000",
            "Optima Secure +": "Covered up to Rs 5,00,000",
        },
    ),
    BenefitRow(
        section="4.2",
        benefit="Daily Cash for choosing Shared Accommodation",
        page=62,
        values={
            "Optima Suraksha": "Rs 800 per day, maximum upto Rs 4,800",
            "Optima Secure": "Rs 800 per day, maximum upto Rs 4,800",
            "Optima Super Secure": "Rs 1,000 per day, maximum upto Rs 6,000",
            "Optima Secure Global": "Rs 800 per day, maximum upto Rs 4,800 (India only)",
            "Optima Secure Global Plus": "Rs 800 per day, maximum upto Rs 4,800 (India only)",
            "Optima Select": NOT_COVERED,
            "Optima Lite": "Rs 800 per day, maximum upto Rs 4,800",
            "Optima Secure +": "Rs 800 per day, maximum upto Rs 4,800",
        },
    ),
    BenefitRow(
        section="4.3",
        benefit="Protect Benefit (non-medical expenses / consumables)",
        page=62,
        values={
            "Optima Suraksha": NOT_COVERED,
            "Optima Secure": "Default: Covered upto sum insured (optional to remove)",
            "Optima Super Secure": "Default: Covered upto sum insured (optional to remove)",
            "Optima Secure Global": "Covered upto sum insured",
            "Optima Secure Global Plus": "Covered upto sum insured",
            "Optima Select": "Optional",
            "Optima Lite": "Optional",
            "Optima Secure +": "Default: Covered upto sum insured (optional to remove)",
        },
    ),
    BenefitRow(
        section="4.4",
        benefit="Plus Benefit",
        page=63,
        values={
            "Optima Suraksha": NOT_COVERED,
            "Optima Secure": "Bonus of 50% of Base Sum Insured per year, maximum upto 100%",
            "Optima Super Secure": "Bonus of 50% of Base Sum Insured per year, maximum upto 100%",
            "Optima Secure Global": "Bonus of 50% of Base Sum Insured per year, maximum upto 100%",
            "Optima Secure Global Plus": "Bonus of 50% of Base Sum Insured per year, maximum upto 100%",
            "Optima Select": "Optional (Bonus of 50% of Base Sum Insured per year, maximum upto 100%)",
            "Optima Lite": "Optional (Bonus of 50% of Base Sum Insured per year, maximum upto 100%)",
            "Optima Secure +": NOT_COVERED,
        },
    ),
    BenefitRow(
        section="4.5",
        benefit="Secure Benefit",
        page=63,
        values={
            "Optima Suraksha": NOT_COVERED,
            "Optima Secure": "Equal to 100% of Base Sum Insured",
            "Optima Super Secure": "Equal to 200% of Base Sum Insured",
            "Optima Secure Global": "Equal to 100% of Base Sum Insured (India only)",
            "Optima Secure Global Plus": "Equal to 100% of Base Sum Insured (India only)",
            "Optima Select": NOT_COVERED,
            "Optima Lite": NOT_COVERED,
            "Optima Secure +": "Equal to 100% of Base Sum Insured",
        },
    ),
    BenefitRow(
        section="4.6",
        benefit="Automatic Restore Benefit",
        page=63,
        values={
            "Optima Suraksha": "Equal to 100% of Base Sum Insured",
            "Optima Secure": "Equal to 100% of Base Sum Insured",
            "Optima Super Secure": "Equal to 100% of Base Sum Insured",
            "Optima Secure Global": "Equal to 100% of Base Sum Insured (India only)",
            "Optima Secure Global Plus": "Equal to 100% of Base Sum Insured (India only)",
            "Optima Select": "Unlimited times",
            "Optima Lite": "Unlimited times",
            "Optima Secure +": "Unlimited times",
        },
    ),
    BenefitRow(
        section="4.7",
        benefit="Aggregate Deductible (optional)",
        page=63,
        values={
            "Optima Suraksha": "10K/25K/50K/1L/2L/3L/5L/10L/20L/25L",
            "Optima Secure": "10K/25K/50K/1L/2L/3L/5L/10L/20L/25L",
            "Optima Super Secure": "10K/25K/50K/1L/2L/3L/5L/10L/20L/25L",
            "Optima Secure Global": "10K/25K/50K/1L/2L/3L/5L/10L/20L/25L (India only)",
            "Optima Secure Global Plus": "10K/25K/50K/1L/2L/3L/5L/10L/20L/25L (India only)",
            "Optima Select": "10K/25K/50K/1L/2L/3L/5L/10L",
            "Optima Lite": "10K/25K/50K",
            "Optima Secure +": "10K/25K/50K/1L/2L/3L/5L/10L/20L/25L",
        },
        footnote=(
            "5L/10L deductible requires Base Sum Insured >= 25 Lakhs. 20L/25L deductible requires "
            "Base Sum Insured >= 50 Lakhs. Aggregate Deductible applies only to claims arising in "
            "India; a per-claim deductible of Rs 10,000 applies to claims outside India in Global plans."
        ),
    ),
    BenefitRow(
        section="4.8",
        benefit="E-Opinion for Critical Illness",
        page=63,
        values={
            "Optima Suraksha": "In India",
            "Optima Secure": "In India",
            "Optima Super Secure": "Global",
            "Optima Secure Global": "Global",
            "Optima Secure Global Plus": "Global",
            "Optima Select": NOT_COVERED,
            "Optima Lite": "In India",
            "Optima Secure +": "In India",
        },
    ),
    BenefitRow(
        section="4.9",
        benefit="Global Health Cover (Emergency Treatments Only)",
        page=64,
        values={
            "Optima Suraksha": NOT_COVERED,
            "Optima Secure": NOT_COVERED,
            "Optima Super Secure": NOT_COVERED,
            "Optima Secure Global": "Covered (Outside India only)",
            "Optima Secure Global Plus": NOT_COVERED,
            "Optima Select": NOT_COVERED,
            "Optima Lite": NOT_COVERED,
            "Optima Secure +": NOT_COVERED,
        },
    ),
    BenefitRow(
        section="4.10",
        benefit="Global Health Cover (Emergency & Planned Treatments)",
        page=64,
        values={
            "Optima Suraksha": NOT_COVERED,
            "Optima Secure": NOT_COVERED,
            "Optima Super Secure": NOT_COVERED,
            "Optima Secure Global": NOT_COVERED,
            "Optima Secure Global Plus": "Covered (Outside India only)",
            "Optima Select": NOT_COVERED,
            "Optima Lite": NOT_COVERED,
            "Optima Secure +": NOT_COVERED,
        },
    ),
    BenefitRow(
        section="4.11",
        benefit="Overseas Travel Secure (optional)",
        page=64,
        values={
            "Optima Suraksha": NOT_COVERED,
            "Optima Secure": NOT_COVERED,
            "Optima Super Secure": NOT_COVERED,
            "Optima Secure Global": "Covered upto sum insured (Outside India only)",
            "Optima Secure Global Plus": "Covered upto sum insured (Outside India only)",
            "Optima Select": NOT_COVERED,
            "Optima Lite": NOT_COVERED,
            "Optima Secure +": NOT_COVERED,
        },
    ),
    BenefitRow(
        section="4.13",
        benefit="Modification of Room Rent (optional)",
        page=64,
        values={
            "Optima Suraksha": NOT_COVERED,
            "Optima Secure": "Upto Single Private room",
            "Optima Super Secure": "Upto Single Private room",
            "Optima Secure Global": NOT_COVERED,
            "Optima Secure Global Plus": NOT_COVERED,
            "Optima Select": "At Actuals OR Shared room",
            "Optima Lite": NOT_COVERED,
            "Optima Secure +": "Upto Single Private room",
        },
    ),
    BenefitRow(
        section="4.17",
        benefit="Infinite Benefit",
        page=64,
        values={
            "Optima Suraksha": NOT_COVERED,
            "Optima Secure": NOT_COVERED,
            "Optima Super Secure": "Optional (Bonus of 100% of Base Sum Insured post every policy year)",
            "Optima Secure Global": NOT_COVERED,
            "Optima Secure Global Plus": NOT_COVERED,
            "Optima Select": NOT_COVERED,
            "Optima Lite": NOT_COVERED,
            "Optima Secure +": "Bonus of 100% of Base Sum Insured post every policy year",
        },
    ),
    BenefitRow(
        section="5",
        benefit="Preventive Health Check-up",
        page=65,
        values={p: "Inbuilt cover" for p in PLANS} | {"Optima Select": "Optional cover"},
        footnote=(
            "India only. Limits by Base Sum Insured, individual policy: 5 & 7.5 Lakhs -> Rs 1,500; "
            "10 Lakhs -> Rs 2,000; 15 Lakhs -> Rs 4,000; 20, 25, 50 & 75 Lakhs -> Rs 5,000; "
            "100 & 200 Lakhs -> Rs 8,000. Family floater policy (cumulative for all insured persons): "
            "5 & 7.5 Lakhs -> Rs 2,500; 10 Lakhs -> Rs 5,000; 15 Lakhs -> Rs 8,000; "
            "20, 25, 50 & 75 Lakhs -> Rs 10,000; 100 & 200 Lakhs -> Rs 15,000. "
            "Not available if an Aggregate Deductible of Rs 5 Lakhs or more is in force."
        ),
    ),
]


def as_chunks() -> list[dict]:
    """Emit one chunk per benefit row, plan->value binding preserved in the text.

    Each chunk is self-contained prose so that both the embedder and BM25 see the plan
    names as literal tokens adjacent to their values. A query naming a plan therefore has
    a lexical hook, which is exactly what BM25 is good at.
    """
    chunks: list[dict] = []
    for row in ROWS:
        lines = [
            f"Annexure A - Schedule of Benefits. {row.benefit} (Section {row.section}). "
            f"This benefit differs by plan. Values for each of the eight plans:"
        ]
        for plan in PLANS:
            lines.append(f"- {plan}: {row.values[plan]}")
        if row.footnote:
            lines.append(f"Note: {row.footnote}")

        chunks.append(
            {
                "chunk_id": f"annexure_a::{row.section}",
                "section": row.section,
                "section_title": f"Annexure A - {row.benefit}",
                "text": "\n".join(lines),
                "page_start": row.page,
                "page_end": row.page,
                "clause_type": "product_specific",
                "source": "annexure_a",
                "plan_scope": "plan_specific",
            }
        )
    return chunks
