"""Tests for the generation layer's deterministic guards.

The model is faked here on purpose. These cover the checks that must hold NO MATTER what the
model returns, so they must not depend on what the model returns. They are also provider-agnostic:
the same guard has to protect a 8B local model and a frontier model alike, and it matters more for
the small one. Anything needing a live model belongs in the eval, not the test suite.
"""

from __future__ import annotations

import pytest

from src.generation.answer import Answerer
from src.generation.prompt import ANSWER_SCHEMA
from src.retrieval.hybrid import Result


class _FakeBackend:
    """Stands in for any LLM. Returns whatever payload the test hands it."""

    def __init__(self, payload: dict):
        self.name = "fake/model"
        self.payload = payload
        self.seen: dict = {}
        self.calls: list[str] = []

    _USAGE = {"input_tokens": 100, "output_tokens": 50,
              "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}

    def complete(self, system: str, user: str, schema: dict):
        self.calls.append("complete")
        self.seen = {"system": system, "user": user, "schema": schema}
        return self.payload, self._USAGE

    def complete_text(self, system: str, user: str):
        self.calls.append("complete_text")
        self.seen = {"system": system, "user": user, "schema": None}
        return self.payload.get("answer", ""), self._USAGE


class _FakeRetriever:
    """Always returns two clauses: sections 4.5 and 10.1.r."""

    def retrieve(self, question, top_k=5):
        return [_result("4.5", "Secure Benefit"), _result("10.1.r", "Standard Exclusions")]


def _result(section: str, title: str) -> Result:
    return Result(
        chunk_id=f"prose::{section}",
        text=f"Section {section} - {title}\nbody text",
        section=section,
        section_title=title,
        page_start=1,
        page_end=1,
        clause_type="product_specific",
        plan_scope="all",
        fused_score=0.03,
        dense_rank=1,
        sparse_rank=1,
        dense_score=0.6,
        sparse_score=9.0,
    )


def _payload(**over) -> dict:
    return {
        "status": "answered",
        "answer": "No, maternity is excluded.",
        "cited_sections": ["10.1.r"],
        "plan_dependent": False,
    } | over


def _ask(payload: dict):
    backend = _FakeBackend(payload)
    return Answerer(_FakeRetriever(), backend).ask("q"), backend


def test_citation_that_was_retrieved_verifies():
    a, _ = _ask(_payload())
    assert a.citations_verified
    assert a.hallucinated_citations == []
    assert a.answered


def test_fabricated_citation_is_caught():
    """The failure a reader cannot catch for themselves.

    A citation is precisely the thing someone trusts without checking. An answer citing
    "Section 3.7" that it was never shown is fluent, authoritative, wrong, and indistinguishable
    from a correct one. This check does not depend on the model cooperating -- which is the whole
    point, because a small local model cooperates less than a frontier one.
    """
    a, _ = _ask(_payload(cited_sections=["3.7", "10.1.r"]))
    assert not a.citations_verified
    assert a.hallucinated_citations == ["3.7"]  # 10.1.r was real; 3.7 was invented


def test_refusal_is_not_an_answer():
    a, _ = _ask(_payload(status="not_in_document", cited_sections=[]))
    assert not a.answered
    assert a.citations_verified  # citing nothing is not a hallucination


def test_missing_field_fails_loudly():
    """A dropped `status` that silently defaulted to "answered" would turn a refusal into an
    assertion about someone's medical cover. Raise instead."""
    bad = _payload()
    del bad["status"]
    with pytest.raises(ValueError, match="omitted required fields"):
        _ask(bad)


def test_model_receives_the_schema_and_the_clauses():
    _, backend = _ask(_payload())
    assert backend.seen["schema"] is ANSWER_SCHEMA
    # The section label must reach the model, or the citation it returns is uncheckable.
    assert "[Section 10.1.r]" in backend.seen["user"]
    # The grounding instruction must reach the model.
    assert "only from the clauses" in backend.seen["system"].lower() or "answer from those clauses" in backend.seen["system"].lower()


def test_default_is_one_stage():
    """One-stage is the default because it MEASURED better (100% vs 62% citation verification).

    Guarding it so the decision cannot silently regress: a future refactor that flips the default
    back to two-stage would reintroduce the sub-clause-citation hallucinations two-stage produced.
    """
    from src.generation.answer import Answerer

    backend = _FakeBackend(_payload())
    ans = Answerer(_FakeRetriever(), backend)
    ans.ask("q")
    # One-stage calls complete() once with the answer schema; it never calls complete_text().
    assert backend.calls == ["complete"], backend.calls


@pytest.mark.parametrize("status", ["answered", "not_in_document", "out_of_scope"])
def test_all_three_statuses_round_trip(status):
    a, _ = _ask(_payload(status=status, cited_sections=[]))
    assert a.status == status
    assert a.answered == (status == "answered")


def test_non_string_citation_does_not_break_the_check():
    """A model may emit 5 instead of "5". Coerce, then verify -- never skip the check."""
    a, _ = _ask(_payload(cited_sections=[10.1]))
    assert a.hallucinated_citations == ["10.1"]  # coerced, and correctly flagged as not retrieved
