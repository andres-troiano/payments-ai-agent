"""
Synthetic payments dataset generator (Stage 1).

Generates a realistic transactions table suitable for analytics and RAG demos.

Usage (CLI):
    python src/data/generate_synthetic.py --n_users 1000 --n_tx 20000 --out data/payments.csv
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import click
import numpy as np
import pandas as pd

try:
    # When executed as a module (python -m src.data.generate_synthetic)
    from .utils import (
        COUNTRIES,
        COUNTRY_WEIGHTS,
        DEVICE_TYPES,
        DEVICE_WEIGHTS,
        OutputPaths,
        category_amount_scale,
        compute_fraud_probability,
        generate_timestamps,
        generate_user_activity_weights,
        generate_user_spend_factors,
        sample_merchants,
        seasonality_multiplier,
        set_random_seed,
    )
except Exception:  # pragma: no cover - fallback for script execution
    import os
    import sys

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
    if SRC_DIR not in sys.path:
        sys.path.insert(0, SRC_DIR)
    from utils import (  # type: ignore  # noqa: E402
        COUNTRIES,
        COUNTRY_WEIGHTS,
        DEVICE_TYPES,
        DEVICE_WEIGHTS,
        OutputPaths,
        category_amount_scale,
        compute_fraud_probability,
        generate_timestamps,
        generate_user_activity_weights,
        generate_user_spend_factors,
        sample_merchants,
        seasonality_multiplier,
        set_random_seed,
    )


def _ensure_output_paths(out: Optional[str], parquet: Optional[str]) -> OutputPaths:
    csv_path = None
    parquet_path = None
    if out:
        csv_path = str(Path(out))
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    if parquet:
        parquet_path = str(Path(parquet))
        Path(parquet_path).parent.mkdir(parents=True, exist_ok=True)
    return OutputPaths(csv_path=csv_path, parquet_path=parquet_path)


def _generate_transaction_ids(n_tx: int, rng: np.random.Generator) -> np.ndarray:
    prefix = "txn_"
    unique = rng.integers(10**9, 10**12, size=n_tx, dtype=np.int64)
    return np.array([f"{prefix}{u}" for u in unique], dtype=object)


def _choose_users(n_tx: int, n_users: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Return user_ids array (int) and user_index array (int index into per-user factors)."""

    user_ids = np.arange(1, n_users + 1, dtype=int)
    user_activity_weights = generate_user_activity_weights(n_users, rng)
    idx = rng.choice(n_users, size=n_tx, replace=True, p=user_activity_weights)
    return user_ids[idx], idx


def _sample_amounts(
    base_scales: np.ndarray,
    user_factors: np.ndarray,
    user_idx: np.ndarray,
    seasonal_mult: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample transaction amounts using lognormal base scaled by category, user, season."""

    # Lognormal parameters
    mu = np.log(np.maximum(base_scales, 1.0)) - 0.5
    sigma = 0.7
    raw = rng.lognormal(mean=mu, sigma=sigma)
    amounts = raw * user_factors[user_idx] * seasonal_mult
    # Round to cents and minimum of $0.5
    amounts = np.maximum(np.round(amounts, 2), 0.5)
    return amounts


def _print_summary(df: pd.DataFrame) -> None:
    total = len(df)
    unique_users = df["user_id"].nunique()
    median_amt = float(df["amount"].median())
    mean_amt = float(df["amount"].mean())
    top_merchants = (
        df.groupby("merchant")["transaction_id"].count().sort_values(ascending=False).head(5)
    )
    click.echo(f"Rows: {total}")
    click.echo(f"Unique users: {unique_users}")
    click.echo(f"Amount median: ${median_amt:,.2f} | mean: ${mean_amt:,.2f}")
    click.echo("Top merchants (by tx count):")
    for name, cnt in top_merchants.items():
        click.echo(f"  - {name}: {int(cnt)}")


def generate_synthetic_payments(
    n_users: int = 1000,
    n_tx: int = 10000,
    seed: int = 42,
    seasonality: bool = True,
    fraud_rate: float = 0.005,
    categories: Optional[List[str]] = None,
    start_date: str = "2023-01-01",
) -> pd.DataFrame:
    """Generate a synthetic payments transactions DataFrame.

    Columns: transaction_id, user_id, merchant, category, amount, timestamp,
    is_fraudulent, device_type, country
    """

    rng = set_random_seed(seed)

    # Choose users for each transaction based on activity weights
    user_ids, user_idx = _choose_users(n_tx, n_users, rng)

    # Per-user spend scale
    user_spend_factors = generate_user_spend_factors(n_users=n_users, rng=rng)

    # Merchant and category sampling
    merchants, cats = sample_merchants(size=n_tx, rng=rng, allowed_categories=categories)

    # Timestamps with hour-of-day bias
    timestamps = generate_timestamps(n=n_tx, rng=rng, start_date=start_date)
    seasonal = seasonality_multiplier(timestamps) if seasonality else np.ones(n_tx)

    # Amounts
    base_scales = category_amount_scale(cats)
    amounts = _sample_amounts(
        base_scales=base_scales,
        user_factors=user_spend_factors,
        user_idx=user_idx,
        seasonal_mult=seasonal,
        rng=rng,
    )

    # Device and country
    device = np.asarray(
        rng.choice(DEVICE_TYPES, size=n_tx, p=np.asarray(DEVICE_WEIGHTS) / np.sum(DEVICE_WEIGHTS))
    )
    country = np.asarray(
        rng.choice(COUNTRIES, size=n_tx, p=np.asarray(COUNTRY_WEIGHTS) / np.sum(COUNTRY_WEIGHTS))
    )

    # Fraud probability and labeling
    fraud_probs = compute_fraud_probability(amounts=amounts, timestamps=timestamps, base_rate=fraud_rate)
    is_fraud = rng.random(n_tx) < fraud_probs

    # Transaction IDs
    txn_ids = _generate_transaction_ids(n_tx=n_tx, rng=rng)

    df = pd.DataFrame(
        {
            "transaction_id": txn_ids,
            "user_id": user_ids,
            "merchant": merchants,
            "category": cats,
            "amount": amounts,
            "timestamp": pd.to_datetime(timestamps),
            "is_fraudulent": is_fraud.astype(bool),
            "device_type": device,
            "country": country,
        }
    )

    # Sort by time for realism
    df.sort_values("timestamp", inplace=True, kind="mergesort")
    df.reset_index(drop=True, inplace=True)
    return df


@click.command()
@click.option("--n_users", type=int, default=1000, show_default=True, help="Number of users")
@click.option("--n_tx", type=int, default=10000, show_default=True, help="Number of transactions")
@click.option("--seed", type=int, default=42, show_default=True, help="Random seed")
@click.option("--seasonality/--no-seasonality", default=True, show_default=True, help="Enable seasonality")
@click.option("--fraud_rate", type=float, default=0.005, show_default=True, help="Base fraud rate")
@click.option("--categories", type=str, default="", help="Comma-separated allowed categories")
@click.option("--start_date", type=str, default="2023-01-01", show_default=True, help="Start date (YYYY-MM-DD)")
@click.option("--out", type=str, default=None, help="CSV output path")
@click.option("--out_parquet", type=str, default=None, help="Parquet output path")
def main(
    n_users: int,
    n_tx: int,
    seed: int,
    seasonality: bool,
    fraud_rate: float,
    categories: str,
    start_date: str,
    out: Optional[str],
    out_parquet: Optional[str],
) -> None:
    """CLI entry point to generate the dataset and optionally save to disk."""

    cats_list = [c.strip() for c in categories.split(",") if c.strip()] if categories else None
    df = generate_synthetic_payments(
        n_users=n_users,
        n_tx=n_tx,
        seed=seed,
        seasonality=seasonality,
        fraud_rate=fraud_rate,
        categories=cats_list,
        start_date=start_date,
    )

    paths = _ensure_output_paths(out=out, parquet=out_parquet)
    if paths.csv_path:
        Path(paths.csv_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(paths.csv_path, index=False)
        click.echo(f"Saved CSV to {paths.csv_path}")
    if paths.parquet_path:
        Path(paths.parquet_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(paths.parquet_path, index=False)
        click.echo(f"Saved Parquet to {paths.parquet_path}")

    _print_summary(df)


if __name__ == "__main__":
    main()
