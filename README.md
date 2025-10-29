## Chat with Your Payments Data — Stages 1 & 2

This repository contains Stage 1 (Synthetic Data Generator) and Stage 2 (LLM Reasoning & Router) of the project *Chat with Your Payments Data*: a realistic payments dataset plus LangChain pipelines to answer natural-language questions over it (and prepare for future RAG/policy and fraud chains).

### What’s Implemented

* **Stage 1 – Synthetic Payments Dataset**

  * Modular generator with realistic refunds/fraud logic, seasonality, country multipliers.
* **Stage 2 – LLM Reasoning & Router**

  * `QueryChain` → translates questions into safe Pandas code and executes it
  * `SummaryChain` → turns tables into concise explanations
  * `RouterChain` → routes queries (`data` | `policy` | `fraud`) and orchestrates chains
  * Provider-agnostic config: OpenAI, Groq, Gemini, Cohere

### Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt

# Generate sample data (Stage 1)
python src/data/generate_synthetic.py --n_users 1000 --n_tx 10000 --out data/payments.csv
```

### Dataset Schema

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
Unique users: 934
Amount median: $21.66 | mean: $46.59
Top merchants (by tx count):
  - Amazon: 851
  - Uber: 728
  - Walmart: 686
  - Netflix: 588
  - MercadoLibre: 574
```

### Stage 2 — LLM Reasoning & Router (LangChain)

Files under `src/chains/`:

- `prompts.py`: reusable few-shot templates
- `query_chain.py`: translates questions → safe Pandas code on `df` and executes
- `summary_chain.py`: turns tabular results into concise text
- `router_chain.py`: routes queries; `ask()`/`ask_async()` orchestrate data flow

LLM provider configuration (optional):

```bash
# Choose provider/model (OpenAI default if unset)
export LLM_PROVIDER=openai   # openai | groq | gemini | cohere
export LLM_MODEL=gpt-4o-mini # e.g., gpt-4o, llama-3.3-70b-versatile, gemini-1.5-pro, command-r-plus

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
    provider="groq", model="llama-3.3-70b-versatile",
)
print(resp["route"])      # "data"
print(resp["answer"])     # concise summary
print(resp["table"][:3])  # preview rows

# Direct query chain with Gemini
from src.chains.query_chain import QueryChain
qc = QueryChain(provider="gemini", model="gemini-1.5-pro")
res = qc.run("Average transaction amount by country")
```

#### Example runs

##### 1. Routed Q&A over Payments Data

```python
from src.chains.router_chain import ask

# Example: natural-language analytics query
q1 = "Which merchants had the highest total revenue last month?"
resp1 = ask(q1)

print("Route:", resp1["route"])
print("Answer:", resp1["answer"])
```

*Output:*

*Route:* data
*Answer:* BestBuy led last month with $1,719.16 in revenue,
followed by Apple at $1,600.81, and Delta at $1,307.85.

### Roadmap (Next Stages)

* **RAG / Policy Chain**: cite refund/fraud policies
* **Fraud Chain**: anomaly tables + thresholds
* **UI (Streamlit)**: chat, charts, evaluation dashboard
* **Observability**: caching, latency metrics, tokens
* **Docker + CI**: one-command run and tests
