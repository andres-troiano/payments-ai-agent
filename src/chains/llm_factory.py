from __future__ import annotations

import os
from typing import Optional

# Base type hint only; actual classes imported lazily
try:
    from langchain.chains import LLMChain  # noqa: F401
except Exception:
    pass


def create_chat_llm(provider: Optional[str] = None, model: Optional[str] = None, temperature: float = 0.1):
    """Create a LangChain chat model based on provider/model env or args.

    Selection order:
      - Explicit args (provider/model) if provided
      - Environment variables: LLM_PROVIDER, LLM_MODEL
      - Defaults: provider=openai, model=gpt-4o-mini

    Supported providers: openai, groq, gemini, cohere
    Env keys used:
      - OPENAI_API_KEY
      - GROQ_API_KEY
      - GOOGLE_API_KEY (Gemini)
      - COHERE_API_KEY
    """

    provider_name = (provider or os.getenv("LLM_PROVIDER") or "openai").strip().lower()
    model_name = (model or os.getenv("LLM_MODEL") or "gpt-4o-mini").strip()

    if provider_name == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model_name, temperature=temperature)

    if provider_name == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=model_name, temperature=temperature)

    if provider_name in ("gemini", "google", "google-genai"):
        from langchain_google_genai import ChatGoogleGenerativeAI

        # Common Gemini model example: "gemini-1.5-pro"
        return ChatGoogleGenerativeAI(model=model_name, temperature=temperature)

    if provider_name == "cohere":
        from langchain_cohere import ChatCohere

        return ChatCohere(model=model_name, temperature=temperature)

    raise ValueError(f"Unsupported LLM provider: {provider_name}")
