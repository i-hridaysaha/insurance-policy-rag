"""FastAPI backend: one /ask endpoint over the existing Answerer.

This is a thin HTTP wrapper. Every hard part -- retrieval, grounding, refusal, citation
verification -- already lives in src/ and is tested. The API's only jobs are to load the index and
model ONCE at startup, and to hand the response back in a shape a frontend can render, with the two
things that make this system trustworthy surfaced explicitly rather than buried:

  status               answered | not_in_document | out_of_scope  -- refusal is a first-class field
  citations_verified   whether every cited clause was actually shown to the model

A caller must be able to see, without reading the prose, that the system refused, or that a
citation could not be verified. Those are the states a careless UI would hide and a policyholder
would be harmed by.

Run:  uvicorn src.api.app:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import DEFAULT_TOP_K, OLLAMA_MODELS
from src.generation.answer import Answerer
from src.generation.backends import OllamaBackend
from scripts.query import load

# Populated at startup. The FAISS index and the Sentence-BERT model take a few seconds to load, so
# they are loaded once here, not per request.
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    retriever, _ = load()
    _state["answerer"] = Answerer(retriever, OllamaBackend(OLLAMA_MODELS[0]))
    _state["model"] = OLLAMA_MODELS[0]
    yield
    _state.clear()


app = FastAPI(
    title="Insurance Policy Q&A",
    description="Ask a health insurance policy a question; get an answer grounded in its clauses, "
    "with verified citations, or an honest refusal.",
    version="1.0.0",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, examples=["Is maternity covered in the first year?"])
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=20)


class Citation(BaseModel):
    section: str
    title: str
    pages: str


class AskResponse(BaseModel):
    question: str
    # The refusal states are first-class, not an error path. A UI must be able to branch on this.
    status: str = Field(description="answered | not_in_document | out_of_scope")
    answer: str
    plan_dependent: bool = Field(description="the answer differs across the eight plan variants")

    cited_sections: list[str]
    # The deterministic backstop, surfaced. False means a cited clause was never shown to the model
    # and the answer must not be trusted -- see hallucinated_citations for which.
    citations_verified: bool
    hallucinated_citations: list[str]

    retrieved: list[Citation] = Field(description="the clauses passed to the model, for auditing")
    model: str


@app.get("/health")
def health() -> dict:
    ready = "answerer" in _state
    return {"status": "ok" if ready else "loading", "model": _state.get("model")}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    answerer: Answerer | None = _state.get("answerer")
    if answerer is None:
        # 503: the model is still loading, or startup failed. Retryable.
        raise HTTPException(status_code=503, detail="model not loaded yet")

    # A regular `def` endpoint, not `async def`: local inference blocks for ~30-40s, and FastAPI
    # runs a sync endpoint in a worker thread so that blocking does not stall the event loop.
    a = answerer.ask(req.question, top_k=req.top_k)

    return AskResponse(
        question=req.question,
        status=a.status,
        answer=a.answer,
        plan_dependent=a.plan_dependent,
        cited_sections=a.cited_sections,
        citations_verified=a.citations_verified,
        hallucinated_citations=a.hallucinated_citations,
        retrieved=[
            Citation(
                section=r.section,
                title=r.section_title,
                pages=(
                    f"p.{r.page_start}"
                    if r.page_start == r.page_end
                    else f"p.{r.page_start}-{r.page_end}"
                ),
            )
            for r in a.retrieved
        ],
        model=a.backend,
    )
