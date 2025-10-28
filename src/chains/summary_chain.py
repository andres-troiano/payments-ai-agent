from __future__ import annotations

import os
from typing import Any, Dict, Optional

import pandas as pd

try:
    from langchain.cache import InMemoryCache
    from langchain.globals import set_llm_cache
    from langchain.chains import LLMChain
    from langchain_openai import ChatOpenAI
    from .prompts import summary_prompt_template
except Exception:
    InMemoryCache = None  # type: ignore
    set_llm_cache = lambda *_args, **_kwargs: None  # type: ignore
    LLMChain = None  # type: ignore
    ChatOpenAI = None  # type: ignore
    from .prompts import summary_prompt_template  # type: ignore


class SummaryChain:
    def __init__(self, model: str = "gpt-4o-mini") -> None:
        try:
            set_llm_cache(InMemoryCache())  # type: ignore
        except Exception:
            pass

        self.llm_chain: Optional[LLMChain] = None
        api_key = os.getenv("OPENAI_API_KEY")
        if LLMChain and ChatOpenAI and api_key:
            llm = ChatOpenAI(model=model, temperature=0.2)
            self.llm_chain = LLMChain(llm=llm, prompt=summary_prompt_template)

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


