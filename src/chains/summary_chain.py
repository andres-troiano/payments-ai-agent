from __future__ import annotations

import os
from typing import Any, Dict, Optional
import logging

import pandas as pd

try:
    from langchain_community.cache import InMemoryCache
    from langchain.globals import set_llm_cache
    try:
        from langchain_core.output_parsers import StrOutputParser
    except Exception:
        StrOutputParser = None  # type: ignore
    from .prompts import summary_prompt_template
    from .llm_factory import create_chat_llm
except Exception:
    InMemoryCache = None  # type: ignore
    set_llm_cache = lambda *_args, **_kwargs: None  # type: ignore
    from .prompts import summary_prompt_template  # type: ignore
    def create_chat_llm(*_args, **_kwargs):  # type: ignore
        return None


logger = logging.getLogger(__name__)


class SummaryChain:
    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None) -> None:
        try:
            set_llm_cache(InMemoryCache())  # type: ignore
        except Exception:
            pass

        self.llm_chain = None  # backward compat name
        self.llm_pipeline = None
        self.llm_provider = (provider or os.getenv("LLM_PROVIDER") or "openai").strip().lower()
        self.llm_model = (model or os.getenv("LLM_MODEL") or "auto").strip()
        try:
            llm = create_chat_llm(provider=provider, model=model, temperature=0.2)
            if llm and summary_prompt_template:
                if StrOutputParser is not None:
                    self.llm_pipeline = summary_prompt_template | llm | StrOutputParser()
                else:
                    self.llm_pipeline = summary_prompt_template | llm
                self.llm_chain = True
                logger.info("SummaryChain initialized with LLM provider=%s model=%s", self.llm_provider, self.llm_model)
        except Exception:
            self.llm_chain = None
            self.llm_pipeline = None
            logger.info("SummaryChain initialized without LLM; using heuristic fallback")

    def run(self, question: str, result: pd.DataFrame) -> str:
        if self.llm_chain is None:
            # Fallback concise summary
            logger.info("SummaryChain using heuristic fallback for question: %s", question)
            preview = result.head(5)
            return f"Summary for '{question}': {len(result)} rows. Top rows:\n{preview.to_markdown(index=False)}"
        try:
            logger.info("SummaryChain using LLM for question: %s", question)
            if self.llm_pipeline is not None:
                return self.llm_pipeline.invoke({"question": question, "result": result.head(20).to_csv(index=False)})
            return f"Summary for '{question}': {len(result)} rows. Top rows:\n{result.head(5).to_markdown(index=False)}"
        except Exception as e:
            logger.warning("SummaryChain LLM error; falling back. Error: %s", e, exc_info=True)
            preview = result.head(5)
            return f"Summary for '{question}': {len(result)} rows. Top rows:\n{preview.to_markdown(index=False)}"


def run(question: str, result: pd.DataFrame) -> str:
    return SummaryChain().run(question, result)
