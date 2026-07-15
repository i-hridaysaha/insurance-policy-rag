"""Ask the policy a question.

    python -m scripts.ask "is maternity covered in the first year"
    python -m scripts.ask "what is my room rent limit" --show-context
"""

from __future__ import annotations

import argparse

from src.config import ANTHROPIC_MODEL, DEFAULT_TOP_K, OLLAMA_MODELS
from src.generation.answer import Answerer
from src.generation.backends import AnthropicBackend, OllamaBackend
from scripts.query import load

STATUS_LABEL = {
    "answered": "ANSWERED",
    "not_in_document": "NOT IN THIS POLICY",
    "out_of_scope": "OUT OF SCOPE FOR THIS SYSTEM",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("-k", type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--show-context", action="store_true", help="print the clauses passed to the model")
    ap.add_argument("--model", default=OLLAMA_MODELS[0], help="ollama model tag")
    ap.add_argument("--anthropic", action="store_true", help="use the frontier baseline (needs credit)")
    ap.add_argument("--two-stage", action="store_true",
                    help="reason-then-extract; measured WORSE on citations, kept for comparison")
    args = ap.parse_args()

    retriever, _ = load()
    backend = AnthropicBackend(ANTHROPIC_MODEL) if args.anthropic else OllamaBackend(args.model)
    a = Answerer(retriever, backend, two_stage=args.two_stage).ask(args.question, top_k=args.k)

    print(f"\nQ: {args.question}\n")

    if args.show_context:
        print("  retrieved:")
        for r in a.retrieved:
            print(f"    - {r.citation()}")
        print()

    print(f"  [{STATUS_LABEL.get(a.status, a.status)}]")
    if a.plan_dependent:
        print("  [ANSWER DEPENDS ON YOUR PLAN]")
    print()

    for line in a.answer.split("\n"):
        print(f"  {line}")

    if a.cited_sections:
        print(f"\n  Sources: {', '.join('Section ' + c for c in a.cited_sections)}")

    # A fabricated citation is the one failure a reader cannot catch on their own, because a
    # citation is exactly what they would trust without checking. Say it loudly.
    if not a.citations_verified:
        print(
            f"\n  WARNING: the model cited {a.hallucinated_citations}, which it was never shown. "
            "Do not trust this answer."
        )

    u = a.usage
    print(f"\n  [{a.backend}]  {u['input_tokens']} in / {u['output_tokens']} out tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
