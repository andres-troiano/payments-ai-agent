## Synthetic Payments Dataset (Stage 1)

This repository contains Stage 1 of "Chat with Your Payments Data": a synthetic transactions generator designed to power downstream LLM analytics and RAG demos.

### Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
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
- `is_refunded` (bool): True if amount < $50 (policy-aligned)
- `is_fraudulent` (bool): True if amount > $500 or hour ∈ [2–5] or high-velocity user

### Realistic Patterns Encoded

- User heterogeneity via per-user spend factors and activity weights
- Merchant popularity differences across categories
- Seasonality: weekend uplift and Nov/Dec peak
- Country multipliers (US higher than MX/AR)
- Refunds under $50; Fraud if > $500, 2–5 AM, or high-velocity users

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

### Next Steps

- Stage 2: LLM chains for structured queries and summaries
- Stage 3: RAG over policy corpus with citations
- Stage 4: Streamlit UI with tabs and evaluation dashboard
