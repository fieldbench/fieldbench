"""Example OpenAI baseline runner for `fieldbench run`.

Keeps the model SDK out of fieldbench core. Install the extra and run:

    pip install "fieldbench[openai]"
    OPENAI_API_KEY=... FIELDBENCH_MODEL=gpt-4o-mini \\
      fieldbench run --corpus ./corpus --out ./preds/gpt-4o-mini \\
        --runner examples.openai_runner:make_runner
    fieldbench score --corpus ./corpus --results ./preds/gpt-4o-mini --mode markdown

Any provider works the same way: give fieldbench.run.LLMRunner a
`complete(prompt) -> str` callable. That's the whole integration surface.
"""

from __future__ import annotations

import os

from fieldbench.run import LLMRunner


def _complete_fn():
    from openai import OpenAI  # imported lazily so core stays dep-light

    client = OpenAI()
    model = os.environ.get("FIELDBENCH_MODEL", "gpt-4o-mini")

    def complete(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content or "{}"

    return complete


def make_runner() -> LLMRunner:
    """Naive whole-document baseline (single call; fails on docs over the window)."""
    return LLMRunner(_complete_fn())


def make_windowed_runner() -> LLMRunner:
    """Windowed baseline: long docs are split, extracted per window, and merged —
    a fair long-document baseline. Window size via FIELDBENCH_MAX_DOC_CHARS
    (default 300k chars ≈ 75k tokens, safely under a 128k context)."""
    max_chars = int(os.environ.get("FIELDBENCH_MAX_DOC_CHARS", "300000"))
    return LLMRunner(_complete_fn(), max_doc_chars=max_chars)
