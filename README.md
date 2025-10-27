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

### Dataset Schema

Columns produced by `src/data/generate_synthetic.py`:

- `transaction_id` (string): unique identifier per transaction
- `user_id` (int): unique customer ID
- `merchant` (string): merchant name (e.g., Amazon, Uber, Walmart)
- `category` (string): semantic grouping (e_commerce, transport, subscription, ...)
- `amount` (float): transaction amount in USD-like units (heavy-tailed)
- `timestamp` (datetime): UTC timestamp with diurnal/seasonal patterns
- `is_fraudulent` (bool): anomaly label based on amount and late-night hours
- `device_type` (string): one of mobile/desktop/tablet
- `country` (string): one of US, MX, AR, BR, CL

### Realistic Patterns Encoded

- User heterogeneity via per-user spend factors and activity weights
- Merchant popularity differences across categories
- Seasonality: weekend uplift and Nov/Dec peak
- Fraud anomalies: higher probability for large late-night transactions

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
