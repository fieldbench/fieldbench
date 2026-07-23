"""Example LlamaIndex baseline runner for `fieldbench run`.

A *genuine* LlamaIndex baseline — it uses the library's structured extraction
(`llm.structured_predict` over a pydantic model built from the schema), not raw
function-calling relabeled. Whole-document (windowed for long docs), no RAG.

    pip install "fieldbench[llamaindex]"
    OPENAI_API_KEY=... FIELDBENCH_MODEL=gpt-4o-mini \\
      fieldbench run --corpus ./corpus --out ./preds/llamaindex \\
        --runner examples.llamaindex_runner:make_runner
"""

from __future__ import annotations

import os
from typing import Any, List, Optional

from pydantic import Field, create_model

from fieldbench.run import _schema_field_names, merge_field_values, window_text


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
        # Carry the schema's field description into the function-call schema so the
        # model gets the same guidance the prompt baseline does (fair comparison).
        desc = str(spec.get("description", "")) or None
        fields[name] = (typ, Field(default=None, description=desc))
    return create_model("Extraction", **fields) if fields else None


class LlamaIndexRunner:
    def __init__(self, model: str, max_doc_chars: int = 120000):
        from llama_index.core.types import PydanticProgramMode
        from llama_index.llms.openai import OpenAI

        # FUNCTION (tool-calling) mode, not response_format: it accepts the loose
        # types (open dicts, untyped arrays) that varied corpus schemas produce,
        # where OpenAI's strict json_schema rejects them.
        self._llm = OpenAI(model=model, temperature=0, pydantic_program_mode=PydanticProgramMode.FUNCTION)
        self._max = max_doc_chars

    def _once(self, doc_text: str, schema: dict) -> dict:
        from llama_index.core.prompts import PromptTemplate

        model = _pydantic_model(schema)
        if model is None:
            return {}
        tmpl = PromptTemplate(
            "Extract the requested fields from the document. Use null when a field "
            "is not present; do not guess.\n\nDocument:\n{doc}"
        )
        obj = self._llm.structured_predict(model, tmpl, doc=doc_text)
        return dict(obj.model_dump())

    def extract(self, doc_text: str, schema: dict, stem: str) -> dict:
        allowed = set(_schema_field_names(schema))
        if self._max and len(doc_text) > self._max:
            preds = [self._once(w, schema) for w in window_text(doc_text, self._max)]
            names = allowed or {k for p in preds for k in p}
            return {n: merge_field_values([p.get(n) for p in preds]) for n in names}
        return self._once(doc_text, schema)


def make_runner() -> LlamaIndexRunner:
    return LlamaIndexRunner(os.environ.get("FIELDBENCH_MODEL", "gpt-4o-mini"))
