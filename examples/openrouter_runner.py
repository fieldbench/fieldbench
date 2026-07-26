"""OpenRouter baseline runner for `fieldbench run` — one key reaches many providers
(Llama, Qwen, Gemini, DeepSeek, ...). OpenAI-compatible endpoint.

    OPENROUTER_API_KEY=... FIELDBENCH_MODEL=meta-llama/llama-3.1-70b-instruct \\
      fieldbench run --corpus ./corpus --out ./preds/llama-3.1-70b \\
        --runner examples.openrouter_runner:make_windowed_runner --mode markdown

No response_format is forced: many open models reject JSON mode, so we rely on the
schema-driven prompt plus fieldbench's robust JSON extraction. temperature=0.
"""

from __future__ import annotations

import os

from fieldbench.run import LLMRunner


def _complete_fn():
    from openai import OpenAI  # OpenRouter is OpenAI-compatible

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    model = os.environ["FIELDBENCH_MODEL"]

    def complete(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return (resp.choices[0].message.content or "{}") if resp.choices else "{}"

    return complete


def make_runner() -> LLMRunner:
    return LLMRunner(_complete_fn())


def make_windowed_runner() -> LLMRunner:
    max_chars = int(os.environ.get("FIELDBENCH_MAX_DOC_CHARS", "120000"))
    return LLMRunner(_complete_fn(), max_doc_chars=max_chars)
