"""Example Anthropic baseline runner for `fieldbench run`.

    pip install "fieldbench[anthropic]"
    ANTHROPIC_API_KEY=... FIELDBENCH_MODEL=claude-sonnet-4-20250514 \\
      fieldbench run --corpus ./corpus --out ./preds/sonnet \\
        --runner examples.anthropic_runner:make_windowed_runner

Same integration surface as any provider: give fieldbench.run.LLMRunner a
`complete(prompt) -> str` callable.
"""

from __future__ import annotations

import os

from fieldbench.run import LLMRunner


def _complete_fn():
    from anthropic import Anthropic  # imported lazily so core stays dep-light

    client = Anthropic()
    model = os.environ.get("FIELDBENCH_MODEL", "claude-sonnet-4-20250514")

    def complete(prompt: str) -> str:
        resp = client.messages.create(
            model=model,
            max_tokens=4096,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
        return "".join(parts) or "{}"

    return complete


def make_runner() -> LLMRunner:
    """Naive whole-document baseline (single call)."""
    return LLMRunner(_complete_fn())


def make_windowed_runner() -> LLMRunner:
    """Windowed baseline for long documents (see openai_runner for notes)."""
    max_chars = int(os.environ.get("FIELDBENCH_MAX_DOC_CHARS", "120000"))
    return LLMRunner(_complete_fn(), max_doc_chars=max_chars)
