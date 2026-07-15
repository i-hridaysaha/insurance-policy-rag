"""Generation: retrieved clauses in, grounded answer with verified citations out.

Provider-agnostic. The backend (local Ollama, or the Anthropic API) supplies the model;
everything that constitutes the system's actual promise -- the prompt, the JSON contract, the
refusal path, the citation check -- lives here and does not change when the model does.

WHY THERE IS NO RETRIEVAL-SCORE REFUSAL GATE
--------------------------------------------
The plan was to refuse cheaply, before spending a model call: if no retriever scored a chunk
above some threshold, the question has no answer here. Measured against the eval set, that does
not work, and it is not close.

    top dense score, 31 answerable questions  : 0.341 - 0.647  (median 0.546)
    top dense score, 3 unanswerable questions : 0.479 - 0.640

They overlap almost entirely. A threshold that rejects all three unanswerable questions also
rejects 29 of the 31 answerable ones. BM25 separates no better.

The cause is structural, not a tuning failure. "What is this insurer's claim settlement ratio?"
IS an insurance question, phrased in the document's own vocabulary, and it embeds next to real
insurance clauses. Retrieval similarity measures TOPICAL RELATEDNESS. Answerability is a
different property, and no threshold on the first can recover the second.

So refusal is a semantic judgement and it belongs to the model. The prompt treats it as a
first-class outcome, and `citations_verified` is the deterministic backstop underneath it:
whatever the model claims, a citation it was never shown cannot pass.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.config import DEFAULT_TOP_K
from src.generation.backends import LLMBackend
from src.generation.extract import EXTRACT_SCHEMA, EXTRACT_SYSTEM, build_extract_message
from src.generation.prompt import (
    ANSWER_SCHEMA,
    SYSTEM,
    SYSTEM_SINGLE_STAGE,
    build_user_message,
)
from src.retrieval.hybrid import HybridRetriever, Result

_CITE_PREFIX = re.compile(r"^\s*(section|clause|sec\.?|§)\s*", re.IGNORECASE)


def normalize_citation(raw: object) -> str:
    """Reduce a cited section to the bare clause id, e.g. "Section 10.1.r." -> "10.1.r".

    The context labels each clause "[Section 10.1.r]", so models naturally echo the word back and
    cite "Section 10.1.r". Comparing that against the bare id "10.1.r" made the hallucination
    check fire on a PERFECTLY CORRECT citation.

    That false positive is worse than no check at all. The warning it raises is the loudest signal
    this system emits -- "do not trust this answer" -- and an alarm that goes off on correct
    answers is an alarm people learn to ignore, which is exactly when the real one gets missed.

    Normalisation is deliberately narrow: a formatting prefix and trailing punctuation, nothing
    more. It must not be so eager that an invented section slips through, because catching those
    is the entire job.
    """
    s = str(raw).strip()
    s = _CITE_PREFIX.sub("", s)
    return s.strip().rstrip(".").strip().lower()


@dataclass
class Answer:
    status: str  # answered | not_in_document | out_of_scope
    answer: str
    cited_sections: list[str]
    plan_dependent: bool

    backend: str = ""
    retrieved: list[Result] = field(default_factory=list)

    # Deterministic post-checks. The model cannot influence these.
    hallucinated_citations: list[str] = field(default_factory=list)
    usage: dict = field(default_factory=dict)

    @property
    def answered(self) -> bool:
        return self.status == "answered"

    @property
    def citations_verified(self) -> bool:
        """Every cited section was actually present in the retrieved context.

        This is the one guarantee that does not depend on the model behaving. An answer can be
        fluent, authoritative, and cite "Section 3.7" without ever having been shown Section 3.7
        -- and a citation is precisely what a reader trusts without checking. It would look
        identical to a correct answer. So the citation is checked against what was actually put
        in the context window, not against what sounds right.

        This matters more with a small local model than with a frontier one, which is exactly
        why it is a hard check and not a prompt instruction.
        """
        return not self.hallucinated_citations


class Answerer:
    """Single constrained call by default. The two-stage alternative measured WORSE.

    THE STORY THIS DEFAULT ENCODES
    ------------------------------
    A single call hands the model the JSON schema and lets it answer and self-label in one shot.
    The alternative (`two_stage=True`) reasons in free text first, then extracts the structure in a
    second constrained call.

    I built the two-stage path on a hypothesis: that constraining decoding to a JSON grammar was
    degrading the model's reading of its own context. The evidence was three anecdotes -- room
    rent, maternity, benefit order -- where a constrained call did worse than free text.

    Then I measured it properly, against a FAIR one-stage baseline (same instructions plus the
    field guidance it needs), across all 34 questions. The hypothesis did not survive:

        metric                   1-stage    2-stage
        citations verified         100.0%     61.8%
        cited the correct clause    90.0%     76.7%
        seconds per question         38.5       59.7

    One-stage hallucinated ZERO citations across all 34 questions. Two-stage hallucinated on 13,
    because free-text prose reaches for fine-grained sub-clause references ("as per 4.6.iv.b",
    "10.1.q.i") that the system indexes at section granularity and cannot verify -- and the
    extractor faithfully transcribes them. The single constrained call forces a discrete, section-
    level commitment the model is far more careful with. And the room-rent anecdote that started
    the whole thing turned out to be a PROMPT bug (the baseline was missing its field guidance),
    not a property of constrained decoding: with the fair prompt, one-stage answers it correctly.

    So the default is the simpler path, because the measurement said so. two_stage is kept, honest,
    behind `scripts/evaluate.py --stages`, so the comparison reproduces. There is one nuance the
    deterministic metrics miss: on the single question whose answering clause was not retrieved,
    two-stage said "not specified" (honest) where one-stage confabulated an order from adjacent
    clauses. That is a retrieval miss, n=1, and it does not outweigh a 100% vs 62% citation gap.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        backend: LLMBackend,
        two_stage: bool = False,
    ):
        self.retriever = retriever
        self.backend = backend
        self.two_stage = two_stage

    def ask(self, question: str, top_k: int = DEFAULT_TOP_K) -> Answer:
        results = self.retriever.retrieve(question, top_k)
        user = build_user_message(question, results)

        if self.two_stage:
            # Stage 1: READ AND REASON. Unconstrained -- no grammar competing for the model's
            # capacity while it works out what the clauses actually say.
            text, u1 = self.backend.complete_text(system=SYSTEM, user=user)

            # Stage 2: EXTRACT. Constrained, but now it is shallow labelling of a short text that
            # already exists. There is no reasoning left to degrade.
            meta, u2 = self.backend.complete(
                system=EXTRACT_SYSTEM,
                user=build_extract_message(question, text),
                schema=EXTRACT_SCHEMA,
            )
            data = {**meta, "answer": text}
            usage = _merge_usage(u1, u2)
            required = EXTRACT_SCHEMA["required"]
        else:
            # The baseline gets SYSTEM plus the field guidance it needs to fill the schema. Handing
            # it the prose-only SYSTEM would handicap it on a prompt difference and let me claim a
            # win that constrained decoding had not actually earned.
            data, usage = self.backend.complete(
                system=SYSTEM_SINGLE_STAGE, user=user, schema=ANSWER_SCHEMA
            )
            required = ANSWER_SCHEMA["required"]

        # A model can still drop a field. Fail loudly rather than defaulting -- a missing `status`
        # that quietly became "answered" would turn a refusal into an assertion about someone's
        # medical cover.
        missing = set(required) - set(data)
        if missing:
            raise ValueError(f"{self.backend.name} omitted required fields: {sorted(missing)}")

        retrieved = {normalize_citation(r.section) for r in results}
        cited = [normalize_citation(c) for c in data["cited_sections"]]

        return Answer(
            status=data["status"],
            answer=data["answer"],
            cited_sections=cited,
            plan_dependent=bool(data["plan_dependent"]),
            backend=self.backend.name,
            retrieved=results,
            hallucinated_citations=[c for c in cited if c not in retrieved],
            usage=usage,
        )


def _merge_usage(*us: dict) -> dict:
    keys = ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
    return {k: sum(u.get(k, 0) for u in us) for k in keys}
