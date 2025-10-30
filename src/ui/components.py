from __future__ import annotations

from typing import Any, Dict, Optional
import re

import pandas as pd
import streamlit as st


def sanitize_text(text: str) -> str:
    safe = text
    safe = safe.replace("$", "\\$").replace("_", "\\_").replace("*", "\\*")
    safe = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", safe)
    safe = re.sub(r"\s+", " ", safe).strip()
    return safe


def display_answer(answer: str, route: str, latency_ms: Optional[int]) -> None:
    latency_str = f"{(latency_ms or 0)/1000:.2f}s"
    st.markdown(f"**Route:** {route} ⏱ {latency_str}")
    st.markdown(sanitize_text(answer))


def _detect_result_type(df: pd.DataFrame) -> str:
    cols = set(df.columns.astype(str))
    if "timestamp" in cols and "amount" in cols:
        return "time_series"
    # simple groupby-style: two columns, one likely categorical
    if df.shape[1] == 2:
        return "bar"
    return "table"


def render_chart(df: pd.DataFrame, kind: Optional[str] = None) -> None:
    t = kind or _detect_result_type(df)
    if t == "time_series" and {"timestamp", "amount"}.issubset(df.columns):
        s = df.sort_values("timestamp")
        st.line_chart(s.set_index("timestamp")["amount"])
    elif t == "bar" and df.shape[1] == 2:
        # assume first is category, second is value
        cat_col = df.columns[0]
        val_col = df.columns[1]
        st.bar_chart(df.set_index(cat_col)[val_col])
    else:
        st.dataframe(df, use_container_width=True)
