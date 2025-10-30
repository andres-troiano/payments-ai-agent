from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

# Ensure project root is on sys.path so `src.*` imports work with Streamlit
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from src.chains.router_chain import ask
from src.ui.components import display_answer, render_chart, sanitize_text


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("ui")


def _init_session() -> None:
    if "messages" not in st.session_state:
        st.session_state["messages"] = []  # list[dict(role, content, table)]
    if "last_metrics" not in st.session_state:
        st.session_state["last_metrics"] = {}


def _sidebar() -> Dict[str, Any]:
    st.sidebar.header("Controls")
    options = ["auto", "openai", "groq", "gemini", "cohere"]
    env_provider = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    default_index = options.index(env_provider) if env_provider in options else 0
    provider = st.sidebar.selectbox("LLM Provider", options=options, index=default_index)
    model = st.sidebar.text_input("Model (optional)", value=os.getenv("LLM_MODEL", ""))
    cache_enabled = st.sidebar.checkbox("Enable Cache", value=False)

    st.sidebar.markdown("### Examples")
    examples = [
        "Which merchants had the highest total revenue last month?",
        "Which country has the highest average transaction amount?",
        "What was the total payment volume last week?",
    ]
    chosen = None
    for i, ex in enumerate(examples, start=1):
        if st.sidebar.button(f"Example {i}"):
            chosen = ex
    st.sidebar.divider()
    metrics_ph = st.sidebar.empty()
    # Do not render metrics here to avoid duplicate groups; we'll render after we have data
    return {"provider": provider, "model": model, "example": chosen, "cache": cache_enabled, "metrics_ph": metrics_ph}


def _render_metrics(metrics: Dict[str, Any], placeholder: Optional[st.delta_generator.DeltaGenerator] = None) -> None:
    target = placeholder if placeholder is not None else st.sidebar
    target.markdown(
        f"""
        ⏱ Latency: {metrics.get('latency_ms','-')} ms  
        🤖 LLM used: {metrics.get('llm_used','-')}  
        🔌 Provider: {metrics.get('llm_provider','-')}  
        🧠 Model: {metrics.get('llm_model','-')}
        """
    )


def _handle_query(query: str, provider: str, model: str) -> None:
    kwargs: Dict[str, Optional[str]] = {}
    if provider and provider != "auto":
        kwargs["provider"] = provider
    if model:
        kwargs["model"] = model
    with st.spinner("Thinking..."):
        response = ask(query, **kwargs)
    st.session_state["last_metrics"] = response.get("metrics", {})
    # Immediately update sidebar metrics for the first question
    _render_metrics(st.session_state["last_metrics"], placeholder=st.session_state.get("metrics_ph") or None)
    display_answer(
        answer=response.get("answer", ""),
        route=response.get("route", "-"),
        latency_ms=response.get("metrics", {}).get("latency_ms"),
    )
    table = response.get("table")
    if table:
        try:
            df = pd.DataFrame(table)
        except Exception:
            df = pd.DataFrame.from_records(table)
        render_chart(df)
        st.dataframe(df, use_container_width=True)
    st.session_state["messages"].append({"role": "user", "content": query})
    st.session_state["messages"].append({"role": "assistant", "content": response.get("answer", ""), "table": table})


def main() -> None:
    st.set_page_config(page_title="Chat with Your Payments Data", layout="wide")
    _init_session()

    st.title("💳 Chat with Your Payments Data")
    st.markdown("Ask questions about transactions, detect anomalies, or search internal policies.")

    controls = _sidebar()
    # store metrics placeholder for later updates
    st.session_state["metrics_ph"] = controls.get("metrics_ph")
    # If metrics already exist (e.g., navigating tabs), render them once
    if st.session_state.get("last_metrics"):
        _render_metrics(st.session_state["last_metrics"], placeholder=st.session_state["metrics_ph"]) 

    tab1, tab2, tab3, tab4 = st.tabs(["💬 Analytics", "🕵️ Fraud Explorer", "📚 Policy Search", "📊 Evaluation Dashboard"])

    with tab1:
        st.subheader("Chat")
        # Show history
        for msg in st.session_state["messages"]:
            with st.chat_message(msg.get("role", "assistant")):
                st.markdown(sanitize_text(msg.get("content", "")))
        # Input
        preset = controls.get("example")
        if preset:
            st.session_state["preset_query"] = preset
        query = st.chat_input("Ask a question about your payments data…")
        if not query and st.session_state.get("preset_query"):
            query = st.session_state.pop("preset_query")
        if query:
            _handle_query(query, provider=controls["provider"], model=controls["model"])

    with tab2:
        st.subheader("Fraud Explorer (coming in Stage 7)")
        st.info("This tab will display anomaly tables and severity once the fraud chain is implemented.")

    with tab3:
        st.subheader("Policy Search (coming in Stage 7)")
        st.info("This tab will show RAG answers with citations once the policy chain is implemented.")

    with tab4:
        st.subheader("Evaluation Dashboard")
        m = st.session_state.get("last_metrics", {})
        st.metric("Latency (ms)", m.get("latency_ms", "-"))
        st.metric("LLM Used", str(m.get("llm_used", "-")))
        st.metric("Provider", m.get("llm_provider", "-"))
        st.metric("Model", m.get("llm_model", "-"))


if __name__ == "__main__":
    main()
