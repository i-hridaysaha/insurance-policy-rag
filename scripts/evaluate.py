"""Evaluate the full pipeline against the 34 ground-truth questions.

    python -m scripts.evaluate                          # default local model
    python -m scripts.evaluate --model llama3.1:8b      # a different local model
    python -m scripts.evaluate --compare                # every local model, side by side
    python -m scripts.evaluate --backend anthropic      # frontier baseline (needs API credit)

WHAT IS MEASURED, AND WHY THERE IS NO LLM JUDGE BY DEFAULT
----------------------------------------------------------
Every headline metric here is DETERMINISTIC. Not one of them needs a model to grade it:

  RETRIEVAL       was the gold clause in the top-k                      (set membership)
  CITATION        was every cited section actually shown to the model   (set membership)
  CITED GOLD      did it cite the clause that actually answers it       (set membership)
  REFUSAL         did it decline the 3 unanswerable questions -- AND    (status check)
                  NOT decline the other 31
  PLAN AWARENESS  did it flag the plan-dependent questions              (flag check)

That is deliberate. The obvious move is an LLM-as-judge for answer correctness, but the only
models available here are the ones under test, and a 14B model grading its own answers is not
evidence. So the numbers that get quoted are the ones that cannot be flattered, and the generated
answers are written to eval/results.json in full so the answer text can be read and checked by a
human against the hand-written reference. At n=34 that is entirely tractable, and it is more
honest than a self-graded score.

REFUSAL IS SCORED IN BOTH DIRECTIONS. A system that refuses everything gets 3/3 on the negatives
and is worthless. False refusals are reported next to missed ones.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter

import yaml

from src.config import ANTHROPIC_MODEL, DEFAULT_TOP_K, EVAL_QA_PATH, OLLAMA_MODELS
from src.generation.answer import Answerer
from src.generation.backends import AnthropicBackend, OllamaBackend
from scripts.query import load


def run_one(answerer: Answerer, pairs: list[dict], k: int) -> list[dict]:
    rows = []
    for i, p in enumerate(pairs, 1):
        t0 = time.perf_counter()
        a = answerer.ask(p["question"], top_k=k)
        elapsed = time.perf_counter() - t0

        retrieved = [r.section for r in a.retrieved]
        expects_refusal = not p["gold_sections"]

        rows.append(
            {
                "id": p["id"],
                "question": p["question"],
                "difficulty": p["difficulty"],
                "expects_refusal": expects_refusal,
                "gold_sections": p["gold_sections"],
                "reference": p["answer"].strip(),
                # what the system produced
                "backend": a.backend,
                "status": a.status,
                "answer": a.answer,
                "cited_sections": a.cited_sections,
                "plan_dependent": a.plan_dependent,
                "retrieved_sections": retrieved,
                "usage": a.usage,
                "seconds": round(elapsed, 1),
                # deterministic checks
                "retrieval_hit": any(g in retrieved for g in p["gold_sections"]),
                "citations_verified": a.citations_verified,
                "hallucinated_citations": a.hallucinated_citations,
                "cited_gold": any(c in p["gold_sections"] for c in a.cited_sections),
                "refused": not a.answered,
            }
        )
        flag = "" if a.citations_verified else "  !! FABRICATED CITATION"
        print(f"  [{i:>2}/{len(pairs)}] {p['id']:<8} {a.status:<16} {elapsed:>5.1f}s{flag}")
    return rows


def report(rows: list[dict], label: str) -> dict:
    answerable = [r for r in rows if not r["expects_refusal"]]
    negatives = [r for r in rows if r["expects_refusal"]]
    n = len(rows)

    def pct(c: int, t: int) -> str:
        return f"{c}/{t} ({c / t * 100:5.1f}%)" if t else "n/a"

    hits = sum(r["retrieval_hit"] for r in answerable)
    halluc = [r for r in rows if not r["citations_verified"]]
    answered = [r for r in answerable if not r["refused"]]
    cited_gold = sum(r["cited_gold"] for r in answered)
    correct_refusals = sum(r["refused"] for r in negatives)
    false_refusals = [r for r in answerable if r["refused"]]
    plan_qs = [r for r in answerable if "annexure_a" in r["gold_sections"]]
    flagged = sum(r["plan_dependent"] for r in plan_qs)

    print(f"\n{'=' * 72}")
    print(f"{label}   n={n}  ({len(answerable)} answerable, {len(negatives)} unanswerable)")
    print("=" * 72)

    print(f"\n  retrieval: gold clause in top-k          {pct(hits, len(answerable))}")
    print(f"\n  CITATIONS")
    print(f"    verified (cited what it was shown)     {pct(n - len(halluc), n)}")
    for r in halluc:
        print(f"        [{r['id']}] fabricated {r['hallucinated_citations']}")
    print(f"    cited the CORRECT clause               {pct(cited_gold, len(answered))}"
          f"   (of {len(answered)} answers given)")

    print(f"\n  REFUSAL  (both directions -- refusing everything is not a win)")
    print(f"    correctly refused the unanswerable     {pct(correct_refusals, len(negatives))}")
    print(f"    WRONGLY refused an answerable one      {pct(len(false_refusals), len(answerable))}")
    for r in false_refusals:
        print(f"        [{r['id']}] {r['question'][:56]}")

    print(f"\n  PLAN AWARENESS")
    print(f"    flagged plan_dependent correctly       {pct(flagged, len(plan_qs))}")

    tin = sum(r["usage"]["input_tokens"] for r in rows)
    tout = sum(r["usage"]["output_tokens"] for r in rows)
    secs = sum(r["seconds"] for r in rows)
    print(f"\n  {tin:,} in / {tout:,} out tokens, {secs:.0f}s total ({secs / n:.1f}s per question)")

    cr = sum(r["usage"]["cache_read_input_tokens"] for r in rows)
    cw = sum(r["usage"]["cache_creation_input_tokens"] for r in rows)
    if rows[0]["backend"].startswith("anthropic") and cr == 0 and cw == 0:
        print("  NOTE: prompt caching did nothing. The system prompt is below Opus 4.8's")
        print("        4096-token minimum cacheable prefix, so the breakpoint is inert.")

    return {
        "label": label,
        "retrieval": hits / len(answerable) if answerable else 0,
        "citations_verified": (n - len(halluc)) / n,
        "cited_gold": cited_gold / len(answered) if answered else 0,
        "refusal_correct": correct_refusals / len(negatives) if negatives else 0,
        "false_refusals": len(false_refusals) / len(answerable) if answerable else 0,
        "plan_flagged": flagged / len(plan_qs) if plan_qs else 0,
        "seconds_per_q": secs / n,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["ollama", "anthropic"], default="ollama")
    ap.add_argument("--model", help="ollama model tag; defaults to the first in OLLAMA_MODELS")
    ap.add_argument("--compare", action="store_true", help="run every local model side by side")
    ap.add_argument("--stages", action="store_true",
                    help="compare one-stage (constrained) against two-stage (reason, then extract)")
    ap.add_argument("--limit", type=int, help="only the first N questions (smoke test)")
    ap.add_argument("-k", type=int, default=DEFAULT_TOP_K)
    args = ap.parse_args()

    qa = yaml.safe_load(EVAL_QA_PATH.read_text())
    pairs = qa["pairs"][: args.limit] if args.limit else qa["pairs"]
    retriever, _ = load()

    if args.compare:
        backends = [OllamaBackend(m) for m in OLLAMA_MODELS]
    elif args.backend == "anthropic":
        backends = [AnthropicBackend(ANTHROPIC_MODEL)]
    else:
        backends = [OllamaBackend(args.model or OLLAMA_MODELS[0])]

    # (backend, two_stage) configurations to run
    if args.stages:
        configs = [(b, ts) for b in backends for ts in (False, True)]
    else:
        configs = [(b, True) for b in backends]

    summaries, all_rows = [], {}
    for b, two_stage in configs:
        label = f"{b.name} [{'2-stage' if two_stage else '1-stage'}]"
        print(f"\n>>> {label}   ({len(pairs)} questions, top_k={args.k})")
        rows = run_one(Answerer(retriever, b, two_stage=two_stage), pairs, args.k)
        summaries.append(report(rows, label))
        all_rows[label] = rows

    if len(summaries) > 1:
        print(f"\n{'=' * 72}\nSIDE BY SIDE\n{'=' * 72}")
        hdr = f"{'metric':<30}" + "".join(f"{s['label'][-22:]:>24}" for s in summaries)
        print(hdr)
        print("-" * len(hdr))
        for key, name in [
            ("citations_verified", "citations verified"),
            ("cited_gold", "cited correct clause"),
            ("refusal_correct", "refused unanswerable"),
            ("false_refusals", "FALSE refusals (lower=better)"),
            ("plan_flagged", "plan-dependent flagged"),
            ("seconds_per_q", "seconds / question"),
        ]:
            vals = "".join(
                f"{s[key]:>23.1f} " if key == "seconds_per_q" else f"{s[key] * 100:>22.1f}% "
                for s in summaries
            )
            print(f"{name:<30}{vals}")

    with open("eval/results.json", "w") as f:
        json.dump(all_rows, f, indent=2)
    print("\nwrote eval/results.json  (full answer text, for reading against the references)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
