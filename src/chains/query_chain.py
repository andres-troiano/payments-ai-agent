from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

try:
    from langchain.cache import InMemoryCache
    from langchain.globals import set_llm_cache
    # LCEL output parser
    try:
        from langchain_core.output_parsers import StrOutputParser
    except Exception:  # older versions fallback
        StrOutputParser = None  # type: ignore
    from .prompts import sql_prompt_template
    from .llm_factory import create_chat_llm
except Exception:  # graceful fallback if LangChain not installed
    InMemoryCache = None  # type: ignore
    set_llm_cache = lambda *_args, **_kwargs: None  # type: ignore
    from .prompts import sql_prompt_template  # type: ignore
    def create_chat_llm(*_args, **_kwargs):  # type: ignore
        return None


DEFAULT_DATA_PATH = str(Path(__file__).resolve().parents[2] / "data" / "payments.csv")


logger = logging.getLogger(__name__)


def _safe_guard_code(code: str) -> None:
    forbidden = [
        r"__",
        r"import\s",
        r"open\(",
        r"exec\(",
        r"eval\(",
        r"os\.",
        r"sys\.",
        r"subprocess",
        r"socket",
        r"requests",
    ]
    for pat in forbidden:
        if re.search(pat, code):
            raise ValueError("Generated code failed safety check.")


def _execute_pandas_code(df: pd.DataFrame, code: str) -> pd.DataFrame:
    _safe_guard_code(code)
    # Restricted globals; allow only minimal builtins
    safe_globals: Dict[str, Any] = {"__builtins__": {}}
    safe_locals: Dict[str, Any] = {"df": df, "pd": pd, "np": np}
    exec(code, safe_globals, safe_locals)
    if "result" not in safe_locals:
        # Try last expression by evaluating code; prefer explicit 'result'
        raise ValueError("Generated code did not assign 'result'.")
    result = safe_locals["result"]
    if isinstance(result, (pd.Series, list, tuple, np.ndarray)):
        result = pd.DataFrame(result)
    if not isinstance(result, pd.DataFrame):
        # Best effort to coerce dict-like
        try:
            result = pd.DataFrame(result)
        except Exception as exc:
            raise ValueError("Result is not a DataFrame-compatible object.") from exc
    return result


def _extract_code(text: str) -> str:
    """Extract executable Python code from LLM output.

    - Prefer content inside triple backticks
    - Strip optional language tag after backticks
    - Fallback: take from the first line containing 'result =' or 'df'/'pd'
    """
    if not isinstance(text, str):
        return str(text)

    # Triple backtick fenced block
    if "```" in text:
        parts = text.split("```")
        # find first non-empty fenced segment
        for seg in parts:
            s = seg.strip()
            if not s:
                continue
            # remove optional language hint like 'python\n'
            lines = s.splitlines()
            if lines and lines[0].strip().lower().startswith("python"):
                lines = lines[1:]
            code = "\n".join(lines).strip()
            if code:
                return code

    # Fallback: find the first occurrence of a likely code start
    markers = ["result =", "df[", "df.", "pd."]
    lower = text
    starts = [lower.find(m) for m in markers if lower.find(m) != -1]
    if starts:
        start = min(starts)
        return text[start:].strip()

    return text.strip()


def _fallback_translate(question: str) -> str:
    q = question.lower()
    if "highest" in q and ("revenue" in q or "amount" in q) and "merchant" in q:
        return (
            "result = (df.groupby('merchant')['amount'].sum()"
            ".sort_values(ascending=False).reset_index().head(10))"
        )
    if "average" in q and "country" in q:
        return "result = df.groupby('country')['amount'].mean().reset_index().sort_values('amount', ascending=False)"
    if "count" in q and "transactions" in q and "last" in q and "week" in q:
        return (
            "cutoff = df['timestamp'].max() - pd.to_timedelta(7, 'D')\n"
            "result = df[df['timestamp'] >= cutoff].shape[0:1]"
        )
    # default: return a safe noop head
    return "result = df.head(20)"


@dataclass
class QueryResult:
    answer: str
    table: pd.DataFrame

    def to_dict(self) -> Dict[str, Any]:
        return {"answer": self.answer, "table": self.table.to_dict(orient="split")}


class QueryChain:
    def __init__(self, data_path: Optional[str] = None, provider: Optional[str] = None, model: Optional[str] = None) -> None:
        self.data_path = data_path or DEFAULT_DATA_PATH
        self.df = pd.read_csv(self.data_path, parse_dates=["timestamp"])

        # Setup cache
        try:
            set_llm_cache(InMemoryCache())  # type: ignore
        except Exception:
            pass

        self.llm_chain = None  # backward name retained for metrics
        self.llm_pipeline = None
        # Track effective provider/model for logging/metrics
        self.llm_provider = (provider or os.getenv("LLM_PROVIDER") or "openai").strip().lower()
        self.llm_model = (model or os.getenv("LLM_MODEL") or "auto").strip()
        try:
            llm = create_chat_llm(provider=provider, model=model, temperature=0.1)
            if llm and sql_prompt_template:
                if StrOutputParser is not None:
                    self.llm_pipeline = sql_prompt_template | llm | StrOutputParser()
                else:
                    # Minimal fallback: just the llm callable
                    self.llm_pipeline = sql_prompt_template | llm
                self.llm_chain = True  # truthy for metrics
                logger.info("QueryChain initialized with LLM provider=%s model=%s", self.llm_provider, self.llm_model)
        except Exception:
            self.llm_chain = None
            self.llm_pipeline = None
            logger.info("QueryChain initialized without LLM; using heuristic fallback")

    def _generate_code(self, question: str) -> str:
        if self.llm_chain is None:
            logger.info("QueryChain using heuristic fallback for question: %s", question)
            return _fallback_translate(question)
        try:
            logger.info("QueryChain using LLM for question: %s", question)
            if self.llm_pipeline is not None:
                raw = self.llm_pipeline.invoke({"question": question})
            else:
                # Shouldn't happen, but guard
                raw = _fallback_translate(question)
            return _extract_code(raw)
        except Exception as e:
            logger.warning("QueryChain LLM error; falling back. Error: %s", e, exc_info=True)
            return _fallback_translate(question)

    def run(self, question: str) -> QueryResult:
        code = self._generate_code(question)
        try:
            table = _execute_pandas_code(self.df, code)
        except Exception as e:
            # One corrective retry with stricter instruction
            logger.info("Retrying with stricter instruction due to exec error: %s", e)
            if self.llm_pipeline is not None:
                try:
                    strict_q = question + "\nReturn ONLY executable Python code that assigns a DataFrame to variable 'result'. No explanations."
                    raw = self.llm_pipeline.invoke({"question": strict_q})
                    code2 = _extract_code(raw)
                    table = _execute_pandas_code(self.df, code2)
                except Exception as e2:
                    logger.warning("Second attempt failed; falling back. Error: %s", e2, exc_info=True)
                    # final fallback
                    code3 = _fallback_translate(question)
                    table = _execute_pandas_code(self.df, code3)
            else:
                # no LLM, just fallback
                code3 = _fallback_translate(question)
                table = _execute_pandas_code(self.df, code3)
        # Simple textual answer: use top row(s)
        answer = f"Computed result with {len(table)} rows. Showing top rows."
        return QueryResult(answer=answer, table=table)


def run(question: str) -> Dict[str, Any]:
    qc = QueryChain()
    res = qc.run(question)
    return res.to_dict()
