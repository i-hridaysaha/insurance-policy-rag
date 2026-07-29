"""LLM backends. The generator is provider-agnostic; only these classes know who is answering.

The whole retrieval layer, the prompt, the JSON schema and the citation guard are identical
across backends. That is the point: swapping the model must not change what the system promises,
only how well it keeps the promise. It also means the eval measures the MODEL, not the plumbing.

Three backends:

  OllamaBackend        local, free, no API key. The default. Runs on the developer's machine, so
                       anyone can clone this repo and run the whole thing end to end.
  OpenAICompatBackend  any OpenAI-compatible chat endpoint with strict json_schema structured
                       output. Exists for the HOSTED demo: a free host cannot run Ollama, so the
                       deployed instance points this at a free API (Groq's OpenAI-compatible
                       endpoint). Free to run, no local model, same schema-constrained contract.
  AnthropicBackend     frontier model, paid. Kept so the local results can be benchmarked against a
                       frontier baseline when that is worth the spend.
"""

from __future__ import annotations

import json
from typing import Protocol

import httpx

from src.config import (
    GEN_EFFORT,
    GEN_MAX_TOKENS,
    OLLAMA_CTX,
    OLLAMA_TIMEOUT,
    OLLAMA_URL,
    OPENAI_COMPAT_TIMEOUT,
)


class LLMBackend(Protocol):
    name: str

    def complete(self, system: str, user: str, schema: dict) -> tuple[dict, dict]:
        """Return (parsed JSON obeying `schema`, usage dict). Grammar-constrained."""
        ...

    def complete_text(self, system: str, user: str) -> tuple[str, dict]:
        """Return (free text, usage dict). UNCONSTRAINED -- no grammar.

        Used only by the two-stage generation path, which is NOT the default: measured across the
        eval set it produced more unverifiable citations than a single constrained call. See the
        long comment on Answerer in src/generation/answer.py for the numbers.
        """
        ...


class OllamaBackend:
    """Local inference via Ollama, with schema-constrained JSON output."""

    def __init__(self, model: str, url: str = OLLAMA_URL, ctx: int = OLLAMA_CTX):
        self.model = model
        self.name = f"ollama/{model}"
        self.url = url
        self.ctx = ctx

    def complete_text(self, system: str, user: str) -> tuple[str, dict]:
        """Unconstrained generation: the model gets its full capacity for reading and reasoning."""
        data = self._chat(system, user, schema=None)
        return data["message"]["content"], self._usage(data)

    def complete(self, system: str, user: str, schema: dict) -> tuple[dict, dict]:
        data = self._chat(system, user, schema=schema)
        content = data["message"]["content"]
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"{self.name} returned invalid JSON despite schema constraint: {content[:300]}"
            ) from e
        return parsed, self._usage(data)

    def _chat(self, system: str, user: str, schema: dict | None) -> dict:
        r = httpx.post(
            f"{self.url}/api/chat",
            timeout=OLLAMA_TIMEOUT,
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                # When a schema is given, Ollama constrains decoding to it -- the same contract the
                # Anthropic backend passes to output_config.format. When it is None, the model
                # generates freely. Stage 1 runs free; stage 2 runs constrained.
                **({"format": schema} if schema else {}),
                "options": {
                    # Extraction from a fixed context should be deterministic. There is exactly
                    # one right answer in the clauses; sampling can only move away from it.
                    "temperature": 0,
                    # num_ctx MUST be set explicitly. Ollama's default context is small, and when
                    # the prompt overflows it Ollama SILENTLY DROPS the overflow -- no error, no
                    # warning. The system prompt plus five retrieved clauses runs ~2.5k tokens, so
                    # the default would quietly truncate the very clauses the answer must be
                    # grounded in, and the model would then "hallucinate" content that we had in
                    # fact torn out of its context ourselves. _warn_if_truncated() below catches it.
                    "num_ctx": self.ctx,
                    "num_predict": GEN_MAX_TOKENS,
                },
            },
        )
        r.raise_for_status()
        data = r.json()
        self._warn_if_truncated(data.get("prompt_eval_count", 0))
        return data

    def _usage(self, data: dict) -> dict:
        return {
            "input_tokens": data.get("prompt_eval_count", 0),
            "output_tokens": data.get("eval_count", 0),
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }

    def _warn_if_truncated(self, prompt_tokens: int) -> None:
        """Ollama truncates silently. If the prompt filled the window, we may have lost clauses."""
        if prompt_tokens and prompt_tokens >= self.ctx:
            raise RuntimeError(
                f"{self.name}: prompt used {prompt_tokens} tokens against a {self.ctx}-token "
                f"context window. Ollama silently drops the overflow, so the retrieved clauses "
                f"may have been truncated out of the model's view. Raise OLLAMA_CTX."
            )


class OpenAICompatBackend:
    """Any OpenAI-compatible /chat/completions endpoint, with strict json_schema output.

    This is what the hosted demo runs on. A free host (HuggingFace Spaces) cannot run Ollama, so
    the deployed instance points this at a free OpenAI-compatible API -- Groq by default. The
    prompt, the schema, the citation guard and the refusal logic are byte-for-byte identical to the
    local path; only who generates the tokens changes.

    Structured output uses response_format={"type": "json_schema", ..., "strict": true}, the
    OpenAI/Groq analogue of Ollama's `format` and Anthropic's output_config.format. ANSWER_SCHEMA
    already satisfies strict mode's rules (every property required, additionalProperties false), so
    the same schema object is passed through unchanged.

    The model is NOT the one the eval measured. The README numbers are for the local models
    (qwen2.5:14b, llama3.1:8b); this backend serves a different model for a zero-cost public demo,
    and the demo labels it as such rather than borrowing the local eval's credibility.
    """

    def __init__(self, model: str, base_url: str, api_key: str):
        self.model = model
        self.name = f"openai-compat/{model}"
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key

    def complete_text(self, system: str, user: str) -> tuple[str, dict]:
        """Unconstrained generation, for the (non-default) two-stage path."""
        data = self._chat(system, user, schema=None)
        return data["choices"][0]["message"]["content"], self._usage(data)

    def complete(self, system: str, user: str, schema: dict) -> tuple[dict, dict]:
        data = self._chat(system, user, schema=schema)
        content = data["choices"][0]["message"]["content"]
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"{self.name} returned invalid JSON despite schema constraint: {content[:300]}"
            ) from e
        return parsed, self._usage(data)

    def _chat(self, system: str, user: str, schema: dict | None) -> dict:
        body: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            # Extraction from a fixed context is deterministic: one right answer in the clauses,
            # sampling can only move away from it. Matches the local backend's temperature=0.
            "temperature": 0,
            "max_tokens": GEN_MAX_TOKENS,
        }
        if schema is not None:
            # strict:true makes the endpoint constrain decoding to the schema rather than merely
            # asking for JSON. name is required by the response_format contract.
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "answer", "schema": schema, "strict": True},
            }
        r = httpx.post(
            f"{self.base_url}/chat/completions",
            timeout=OPENAI_COMPAT_TIMEOUT,
            headers={"Authorization": f"Bearer {self._api_key}"},
            json=body,
        )
        r.raise_for_status()
        return r.json()

    def _usage(self, data: dict) -> dict:
        u = data.get("usage", {}) or {}
        return {
            "input_tokens": u.get("prompt_tokens", 0),
            "output_tokens": u.get("completion_tokens", 0),
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }


class AnthropicBackend:
    """Frontier model via the Anthropic API. Requires credit; not the default."""

    def __init__(self, model: str, client=None, effort: str = GEN_EFFORT):
        import anthropic

        self.model = model
        self.name = f"anthropic/{model}"
        self.effort = effort
        self.client = client or anthropic.Anthropic()

    def complete(self, system: str, user: str, schema: dict) -> tuple[dict, dict]:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=GEN_MAX_TOKENS,
            # Adaptive is the only thinking mode on Opus 4.8. budget_tokens is removed (400), and
            # OMITTING this parameter runs with no thinking at all -- it is not adaptive by default.
            thinking={"type": "adaptive"},
            output_config={
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": schema},
            },
            system=system,
            messages=[{"role": "user", "content": user}],
            # No temperature / top_p / top_k: all three are removed on Opus 4.8 and return a 400.
        )

        # Safety classifiers can decline. Check stop_reason BEFORE touching content -- on a
        # refusal the content list can be empty and indexing it raises.
        if response.stop_reason == "refusal":
            return (
                {
                    "status": "not_in_document",
                    "answer": "The request was declined by the model's safety system.",
                    "cited_sections": [],
                    "plan_dependent": False,
                },
                _anthropic_usage(response),
            )

        # Adaptive thinking puts thinking blocks first, so select by type rather than content[0].
        text = next((b.text for b in response.content if b.type == "text"), "")
        return json.loads(text), _anthropic_usage(response)


def _anthropic_usage(response) -> dict:
    u = response.usage
    return {
        "input_tokens": u.input_tokens,
        "output_tokens": u.output_tokens,
        "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
        "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
    }
