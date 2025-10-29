from __future__ import annotations

import os
from typing import Any, Dict, Optional

import pandas as pd

try:
    from langchain.cache import InMemoryCache
    from langchain.globals import set_llm_cache
    from langchain.chains import LLMChain
    from .prompts import summary_prompt_template
    from .llm_factory import create_chat_llm
except Exception:
    InMemoryCache = None  # type: ignore
    set_llm_cache = lambda *_args, **_kwargs: None  # type: ignore
    LLMChain = None  # type: ignore
    from .prompts import summary_prompt_template  # type: ignore
    def create_chat_llm(*_args, **_kwargs):  # type: ignore
        return None


class SummaryChain:
    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None) -> None:
        try:
            set_llm_cache(InMemoryCache())  # type: ignore
        except Exception:
            pass

        self.llm_chain: Optional[LLMChain] = None
        try:
            llm = create_chat_llm(provider=provider, model=model, temperature=0.2)
            if LLMChain and llm:
                self.llm_chain = LLMChain(llm=llm, prompt=summary_prompt_template)
        except Exception:
            self.llm_chain = None

    def run(self, question: str, result: pd.DataFrame) -> str:
        if self.llm_chain is None:
            # Fallback concise summary
            preview = result.head(5)
            return f"Summary for '{question}': {len(result)} rows. Top rows:\n{preview.to_markdown(index=False)}"
        try:
            return self.llm_chain.run({"question": question, "result": result.head(20).to_csv(index=False)})  # type: ignore
        except Exception:
            preview = result.head(5)
            return f"Summary for '{question}': {len(result)} rows. Top rows:\n{preview.to_markdown(index=False)}"


def run(question: str, result: pd.DataFrame) -> str:
    return SummaryChain().run(question, result)
