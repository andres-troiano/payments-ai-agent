## Synthetic Payments Dataset (Stage 1) + Stage 2 Reasoning

This repository contains Stage 1 of "Chat with Your Payments Data": a synthetic transactions generator designed to power downstream LLM analytics and RAG demos.

### Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt

# Generate sample data (Stage 1)
python src/data/generate_synthetic.py --n_users 1000 --n_tx 10000 --out data/payments.csv
```

### Dataset Schema (v2)

Columns produced by `src/data/generate_synthetic.py`:

- `transaction_id` (int): unique identifier
- `user_id` (int): unique customer ID
- `segment` (str): one of `consumer`, `SMB`, `merchant`
- `country` (str): one of `US`, `MX`, `AR`, `BR`, `CL`
- `merchant` (str): e.g., Amazon, Uber, Spotify
- `category` (str): e.g., `e-commerce`, `mobility`, `subscriptions`, `electronics`, ...
- `amount` (float): transaction amount (log-normalish, with country and seasonality multipliers)
- `timestamp` (datetime): UTC timestamp with diurnal/seasonal patterns
- `device_type` (str): `mobile` | `desktop` | `tablet`
- `is_refunded` (bool): Refunds are request-based; if requested and amount < $50 → auto-approve; if ≥ $50 → small manual approval chance.
- `is_fraudulent` (bool): Rare, probabilistic, and conditioned on triggers: amount > $500, local hour ∈ [2–5], or high-velocity user.

### Realistic Patterns Encoded

- User heterogeneity via per-user spend factors and activity weights
- Merchant popularity differences across categories
- Seasonality: weekend uplift and Nov/Dec peak
- Country multipliers (US higher than MX/AR)
- Refunds: request-based, auto-approve < $50 (policy-aligned)
- Fraud: probabilistic, triggered by high amount, late local hours (2–5), or high velocity

### CLI Options

```bash
python src/data/generate_synthetic.py \
  --n_users 1000 \
  --n_tx 10000 \
  --seed 42 \
  --seasonality/--no-seasonality \
  --fraud_rate 0.005 \
  --categories "e_commerce,transport" \
  --start_date 2023-01-01 \
  --out data/payments.csv \
  --out_parquet data/payments.parquet
```

### Example Summary Output

```
Rows: 10000
Unique users: ~900-1000
Amount median: $30–50 (heavy-tailed)
Top merchants (by tx count): Amazon, Uber, Walmart, ...
```

### Stage 2 – LLM Reasoning & Router (LangChain)

Files under `src/chains/`:

- `prompts.py`: reusable few-shot templates
- `query_chain.py`: translates questions → safe Pandas code on `df` and executes
- `summary_chain.py`: turns tabular results into concise text
- `router_chain.py`: routes queries; `ask()`/`ask_async()` orchestrate data flow

LLM provider configuration (optional):

```bash
# Choose provider/model (OpenAI default if unset)
export LLM_PROVIDER=openai   # openai | groq | gemini | cohere
export LLM_MODEL=gpt-4o-mini # e.g., gpt-4o, llama-3.1-70b-versatile, gemini-1.5-pro, command-r-plus

# Provider API keys (set only those you use)
export OPENAI_API_KEY=...
export GROQ_API_KEY=...
export GOOGLE_API_KEY=...
export COHERE_API_KEY=...

# Tip: copy .env_example to .env and load in apps/notebooks with python-dotenv
```

Usage examples:

```python
# Routed Q&A over payments.csv (explicit provider/model)
from src.chains.router_chain import ask
resp = ask(
    "Which merchants had the highest total revenue last month?",
    provider="groq", model="llama-3.1-70b-versatile",
)
print(resp["route"])      # "data"
print(resp["answer"])     # concise summary
print(resp["table"][:3])  # preview rows

# Direct query chain with Gemini
from src.chains.query_chain import QueryChain
qc = QueryChain(provider="gemini", model="gemini-1.5-pro")
res = qc.run("Average transaction amount by country")
```

### Next Steps

- Stage 3: RAG over policy corpus with citations
- Stage 4: Streamlit UI with tabs and evaluation dashboard
