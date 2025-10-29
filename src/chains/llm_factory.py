from __future__ import annotations

import os
from typing import Optional
import logging

# Base type hint only; actual classes imported lazily
try:
    from langchain.chains import LLMChain  # noqa: F401
except Exception:
    pass

logger = logging.getLogger(__name__)

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
    env_model = os.getenv("LLM_MODEL")
    if model:
        model_name = model.strip()
    elif env_model:
        model_name = env_model.strip()
    else:
        # Default model per provider
        default_map = {
            "openai": "gpt-4o-mini",
            "groq": "llama-3.3-70b-versatile",
            "gemini": "gemini-1.5-flash-latest",
            "google": "gemini-1.5-flash-latest",
            "google-genai": "gemini-1.5-flash-latest",
            "cohere": "command-r-plus",
        }
        model_name = default_map.get(provider_name, "gpt-4o-mini")

    if provider_name == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model_name, temperature=temperature)

    if provider_name == "groq":
        from langchain_groq import ChatGroq

        # Groq's OpenAI-compatible endpoint may require max_tokens; set a safe default.
        max_tokens_env = os.getenv("LLM_MAX_TOKENS")
        try:
            max_tokens = int(max_tokens_env) if max_tokens_env else 256
        except Exception:
            max_tokens = 256
        return ChatGroq(model=model_name, temperature=temperature, max_tokens=max_tokens)

    if provider_name in ("gemini", "google", "google-genai"):
        from langchain_google_genai import ChatGoogleGenerativeAI

        # Common Gemini model example: "gemini-1.5-pro"
        return ChatGoogleGenerativeAI(model=model_name, temperature=temperature)

    if provider_name == "cohere":
        from langchain_cohere import ChatCohere

        return ChatCohere(model=model_name, temperature=temperature)

    raise ValueError(f"Unsupported LLM provider: {provider_name}")
