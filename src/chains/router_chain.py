from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, Tuple

from .query_chain import QueryChain
from .summary_chain import SummaryChain


KEYWORDS_POLICY = {"policy", "refund", "guideline", "privacy", "procedure"}
KEYWORDS_FRAUD = {"fraud", "anomal", "suspicious", "chargeback"}


def route(question: str) -> str:
    q = question.lower()
    if any(k in q for k in KEYWORDS_POLICY):
        return "policy"
    if any(k in q for k in KEYWORDS_FRAUD):
        return "fraud"
    return "data"


def ask(question: str, provider: str | None = None, model: str | None = None) -> Dict[str, Any]:
    started = time.time()
    r = route(question)
    if r == "data":
        qc = QueryChain(provider=provider, model=model)
        res = qc.run(question)
        sc = SummaryChain(provider=provider, model=model)
        summary = sc.run(question, res.table)
        duration = time.time() - started
        llm_used = bool(qc.llm_chain)
        metrics = {"latency_ms": int(duration * 1000), "llm_used": llm_used}
        # attach provider/model hints when available
        if llm_used:
            metrics.update({
                "llm_provider": getattr(qc, "llm_provider", None),
                "llm_model": getattr(qc, "llm_model", None),
            })
        # one-line console hint
        print(f"[router] llm_used={llm_used} provider={metrics.get('llm_provider','-')} model={metrics.get('llm_model','-')}")
        return {
            "route": r,
            "answer": summary,
            "table": res.table.to_dict(orient="records"),
            "metrics": metrics,
        }
    # Placeholders for future stages
    duration = time.time() - started
    return {"route": r, "answer": "Not implemented yet.", "table": [], "metrics": {"latency_ms": int(duration * 1000)}}


async def ask_async(question: str, provider: str | None = None, model: str | None = None) -> Dict[str, Any]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, ask, question, provider, model)
