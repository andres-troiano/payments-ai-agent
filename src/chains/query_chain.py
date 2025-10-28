from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

try:
    from langchain.cache import InMemoryCache
    from langchain.globals import set_llm_cache
    from langchain.chains import LLMChain
    from langchain_openai import ChatOpenAI
    from .prompts import sql_prompt_template
except Exception:  # graceful fallback if LangChain/OpenAI not installed
    InMemoryCache = None  # type: ignore
    set_llm_cache = lambda *_args, **_kwargs: None  # type: ignore
    LLMChain = None  # type: ignore
    ChatOpenAI = None  # type: ignore
    from .prompts import sql_prompt_template  # type: ignore


DEFAULT_DATA_PATH = str(Path(__file__).resolve().parents[2] / "data" / "payments.csv")


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
    def __init__(self, data_path: Optional[str] = None, model: str = "gpt-4o-mini") -> None:
        self.data_path = data_path or DEFAULT_DATA_PATH
        self.df = pd.read_csv(self.data_path, parse_dates=["timestamp"])

        # Setup cache
        try:
            set_llm_cache(InMemoryCache())  # type: ignore
        except Exception:
            pass

        self.llm_chain: Optional[LLMChain] = None
        api_key = os.getenv("OPENAI_API_KEY")
        if LLMChain and ChatOpenAI and api_key:
            llm = ChatOpenAI(model=model, temperature=0.1)
            self.llm_chain = LLMChain(llm=llm, prompt=sql_prompt_template)

    def _generate_code(self, question: str) -> str:
        if self.llm_chain is None:
            return _fallback_translate(question)
        try:
            return self.llm_chain.run({"question": question})  # type: ignore
        except Exception:
            return _fallback_translate(question)

    def run(self, question: str) -> QueryResult:
        code = self._generate_code(question)
        table = _execute_pandas_code(self.df, code)
        # Simple textual answer: use top row(s)
        answer = f"Computed result with {len(table)} rows. Showing top rows."
        return QueryResult(answer=answer, table=table)


def run(question: str) -> Dict[str, Any]:
    qc = QueryChain()
    res = qc.run(question)
    return res.to_dict()


