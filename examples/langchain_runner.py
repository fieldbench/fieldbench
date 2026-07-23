"""Example LangChain baseline runner for `fieldbench run`.

A genuine LangChain baseline: a pydantic model built from the schema, extracted
via `ChatOpenAI().with_structured_output(...)` in function-calling mode (so the
loose types of varied corpus schemas don't hit strict-json-schema rejection).
Whole-document, windowed for long docs. Behind the `langchain` extra.

    pip install "fieldbench[langchain]"
    OPENAI_API_KEY=... FIELDBENCH_MODEL=gpt-4o-mini \\
      fieldbench run --corpus ./corpus --out ./preds/langchain \\
        --runner examples.langchain_runner:make_runner
"""

from __future__ import annotations

import os
from typing import Any, List, Optional

from pydantic import BaseModel, Field, create_model

from fieldbench.run import _schema_field_names, merge_field_values, window_text

_PROMPT = (
    "Extract the requested fields from the document. Use null when a field is "
    "not present; do not guess.\n\nDocument:\n{doc}"
)


def _pydantic_model(schema: dict):
    fields = {}
    for name, spec in (schema.get("fields") or {}).items():
        spec = spec if isinstance(spec, dict) else {}
        t = str(spec.get("type", "")).lower()
        if t in ("array", "list"):
            typ = Optional[List[Any]]
        elif t in ("object", "dict"):
            typ = Optional[dict]
        elif t in ("number", "integer", "float", "currency", "int"):
            typ = Optional[float]
        else:
            typ = Optional[str]
        desc = str(spec.get("description", "")) or None
        fields[name] = (typ, Field(default=None, description=desc))
    return create_model("Extraction", **fields) if fields else None


class LangChainRunner:
    def __init__(self, model: str, max_doc_chars: int = 120000):
        from langchain_openai import ChatOpenAI

        self._chat = ChatOpenAI(model=model, temperature=0)
        self._max = max_doc_chars

    def _once(self, doc_text: str, schema: dict) -> dict:
        model = _pydantic_model(schema)
        if model is None:
            return {}
        structured = self._chat.with_structured_output(model, method="function_calling")
        obj = structured.invoke(_PROMPT.format(doc=doc_text))
        if isinstance(obj, BaseModel):
            return dict(obj.model_dump())
        return dict(obj) if isinstance(obj, dict) else {}

    def extract(self, doc_text: str, schema: dict, stem: str) -> dict:
        allowed = set(_schema_field_names(schema))
        if self._max and len(doc_text) > self._max:
            preds = [self._once(w, schema) for w in window_text(doc_text, self._max)]
            names = allowed or {k for p in preds for k in p}
            return {n: merge_field_values([p.get(n) for p in preds]) for n in names}
        return self._once(doc_text, schema)


def make_runner() -> LangChainRunner:
    return LangChainRunner(os.environ.get("FIELDBENCH_MODEL", "gpt-4o-mini"))
