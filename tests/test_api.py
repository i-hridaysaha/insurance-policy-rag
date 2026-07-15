"""API contract tests. The model is faked, so these run in milliseconds and need no Ollama.

They pin the one thing the HTTP layer must not get wrong: the refusal status and the
citation-verification flag have to survive the trip out to the client intact. A UI branches on
those, and a wrapper that dropped or defaulted them would turn a refusal into an apparent answer.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient

from src.api import app as app_module
from src.retrieval.hybrid import Result


@dataclass
class _FakeAnswer:
    status: str
    answer: str
    cited_sections: list
    plan_dependent: bool
    hallucinated_citations: list
    backend: str = "fake/model"
    retrieved: list = None

    @property
    def citations_verified(self) -> bool:
        return not self.hallucinated_citations


class _FakeAnswerer:
    def __init__(self, answer: _FakeAnswer):
        self._answer = answer
        self.last_top_k = None

    def ask(self, question, top_k=5):
        self.last_top_k = top_k
        return self._answer


def _clause(section="10.1.r"):
    return Result(
        chunk_id=f"prose::{section}", text="body", section=section,
        section_title="Standard Exclusions", page_start=31, page_end=32,
        clause_type="product_specific", plan_scope="all",
        fused_score=0.03, dense_rank=1, sparse_rank=1, dense_score=0.6, sparse_score=9.0,
    )


@pytest.fixture
def client_with(monkeypatch):
    """A TestClient whose answerer is a fake we control. Lifespan still runs, so we override the
    state it populates rather than letting it load the real index."""

    def _make(answer: _FakeAnswer):
        # Neuter the real startup load; inject our fake into the module state the endpoint reads.
        monkeypatch.setattr(app_module, "load", lambda: (None, None))
        client = TestClient(app_module.app)
        app_module._state["answerer"] = _FakeAnswerer(answer)
        app_module._state["model"] = "fake/model"
        return client

    return _make


def test_answered_question_returns_answer_and_verified_citation(client_with):
    ans = _FakeAnswer(
        status="answered", answer="No, maternity is excluded.",
        cited_sections=["10.1.r"], plan_dependent=False,
        hallucinated_citations=[], retrieved=[_clause()],
    )
    r = client_with(ans).post("/ask", json={"question": "is maternity covered?"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "answered"
    assert body["citations_verified"] is True
    assert body["cited_sections"] == ["10.1.r"]
    assert body["retrieved"][0]["section"] == "10.1.r"
    assert body["retrieved"][0]["pages"] == "p.31-32"


def test_refusal_status_survives_to_the_client(client_with):
    """The endpoint must NOT flatten a refusal into a 200-with-answer that looks answered.

    'not_in_document' is a successful HTTP response carrying a refusal, and the status field is how
    a UI tells the difference. Dropping it would present an "I don't know" as if it were an answer.
    """
    ans = _FakeAnswer(
        status="not_in_document", answer="The policy does not address this.",
        cited_sections=[], plan_dependent=False,
        hallucinated_citations=[], retrieved=[],
    )
    body = client_with(ans).post("/ask", json={"question": "claim settlement ratio?"}).json()
    assert body["status"] == "not_in_document"
    assert body["cited_sections"] == []


def test_unverified_citation_is_flagged_to_the_client(client_with):
    """If the model cited a clause it was never shown, the client must be told, in a field it can
    act on -- not left to notice a discrepancy in the prose."""
    ans = _FakeAnswer(
        status="answered", answer="Room rent is capped (4.13.i).",
        cited_sections=["4.13.i"], plan_dependent=True,
        hallucinated_citations=["4.13.i"], retrieved=[_clause("3.1.a")],
    )
    body = client_with(ans).post("/ask", json={"question": "room rent?"}).json()
    assert body["citations_verified"] is False
    assert body["hallucinated_citations"] == ["4.13.i"]
    assert body["plan_dependent"] is True


def test_empty_question_is_rejected(client_with):
    ans = _FakeAnswer("answered", "x", [], False, [])
    r = client_with(ans).post("/ask", json={"question": ""})
    assert r.status_code == 422  # pydantic min_length


def test_top_k_is_passed_through(client_with):
    ans = _FakeAnswer("answered", "x", [], False, [], retrieved=[])
    client = client_with(ans)
    client.post("/ask", json={"question": "q", "top_k": 8})
    assert app_module._state["answerer"].last_top_k == 8


def test_health_reports_ready_once_loaded(client_with):
    client = client_with(_FakeAnswer("answered", "x", [], False, []))
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model"] == "fake/model"
