"""
Utilities for generating realistic synthetic payments data.

This module provides merchant catalogs, sampling helpers, timestamp
generation with seasonality, per-user spend factors, and fraud
probability utilities used by the Stage 1 data generator.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


# ------------------------------
# Catalogs and constants
# ------------------------------

# Categories aligned to v2 spec
# (merchant, category, popularity_weight) — categories aligned to v2 spec
MERCHANT_CATALOG: List[Tuple[str, str, float]] = [
    ("Amazon", "e-commerce", 1.0),
    ("Walmart", "e-commerce", 0.8),
    ("MercadoLibre", "e-commerce", 0.7),
    ("Alibaba", "e-commerce", 0.5),
    ("Apple", "electronics", 0.5),
    ("BestBuy", "electronics", 0.4),
    ("Microsoft", "electronics", 0.3),
    ("Steam", "gaming", 0.5),
    ("Netflix", "subscriptions", 0.7),
    ("Spotify", "subscriptions", 0.6),
    ("Uber", "mobility", 0.9),
    ("Lyft", "mobility", 0.4),
    ("DoorDash", "food_delivery", 0.5),
    ("iFood", "food_delivery", 0.4),
    ("Starbucks", "coffee", 0.6),
    ("McDonalds", "restaurant", 0.5),
    ("Shell", "fuel", 0.4),
    ("Airbnb", "travel", 0.3),
    ("Delta", "travel", 0.25),
    ("Telcel", "telco", 0.3),
    ("Claro", "telco", 0.3),
    ("Oxxo", "retail", 0.35),
    ("Rappi", "food_delivery", 0.35),
]

CATEGORY_AMOUNT_SCALE: dict[str, float] = {
    "e-commerce": 35.0,
    "mobility": 18.0,
    "subscriptions": 12.0,
    "restaurant": 22.0,
    "coffee": 8.0,
    "food_delivery": 24.0,
    "travel": 140.0,
    "fuel": 45.0,
    "electronics": 160.0,
    "gaming": 30.0,
    "telco": 55.0,
    "retail": 28.0,
}

COUNTRIES: Tuple[str, ...] = ("US", "MX", "AR", "BR", "CL")
COUNTRY_WEIGHTS: Tuple[float, ...] = (0.6, 0.15, 0.07, 0.13, 0.05)

DEVICE_TYPES: Tuple[str, ...] = ("mobile", "desktop", "tablet")
DEVICE_WEIGHTS: Tuple[float, ...] = (0.65, 0.25, 0.10)

# Segments and weights
SEGMENTS: Tuple[str, ...] = ("consumer", "SMB", "merchant")
SEGMENT_WEIGHTS: Tuple[float, ...] = (0.8, 0.15, 0.05)

# Country spend multipliers (US higher vs MX/AR)
COUNTRY_AMOUNT_MULTIPLIER: dict[str, float] = {
    "US": 1.10,
    "MX": 0.92,
    "AR": 0.88,
    "BR": 0.96,
    "CL": 0.94,
}

# Approximate UTC offsets (hours) for local-time fraud window checks
# These are coarse, not accounting for DST or regions.
COUNTRY_UTC_OFFSET_HOURS: dict[str, int] = {
    "US": -5,  # approximate (ET)
    "MX": -6,
    "AR": -3,
    "BR": -3,
    "CL": -4,
}


def set_random_seed(seed: int) -> np.random.Generator:
    """Return a numpy default_rng seeded generator for reproducibility."""

    return np.random.default_rng(seed)


def normalize_weights(weights: Sequence[float]) -> np.ndarray:
    total = float(np.sum(weights))
    if total <= 0:
        raise ValueError("Weights must sum to a positive value.")
    return np.asarray(weights, dtype=float) / total


def weighted_choice(
    items: Sequence[str], weights: Sequence[float], size: int, rng: np.random.Generator
) -> np.ndarray:
    """Vectorized weighted choice over items.

    Returns an array of length `size` of selected items.
    """

    probs = normalize_weights(weights)
    idx = rng.choice(len(items), size=size, replace=True, p=probs)
    return np.asarray(items, dtype=object)[idx]


def sample_merchants(
    size: int, rng: np.random.Generator, allowed_categories: Sequence[str] | None = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample merchants and categories jointly using catalog weights.

    Returns two arrays: merchants, categories.
    """

    catalog = MERCHANT_CATALOG
    if allowed_categories:
        allowed_set = set(allowed_categories)
        catalog = [m for m in MERCHANT_CATALOG if m[1] in allowed_set]
        if not catalog:
            raise ValueError("No merchants available for provided categories.")

    names = [m[0] for m in catalog]
    cats = [m[1] for m in catalog]
    wts = [m[2] for m in catalog]

    idx = rng.choice(len(names), size=size, replace=True, p=normalize_weights(wts))
    merchants = np.asarray(names, dtype=object)[idx]
    categories = np.asarray(cats, dtype=object)[idx]
    return merchants, categories


def generate_user_spend_factors(
    n_users: int,
    rng: np.random.Generator,
    heavy_spender_share: float = 0.15,
    base_mean: float = 1.0,
    heavy_multiplier_mean: float = 2.5,
    heavy_multiplier_std: float = 0.6,
) -> np.ndarray:
    """Per-user multiplicative spend scale factors capturing user heterogeneity."""

    is_heavy = rng.random(n_users) < heavy_spender_share
    factors = np.full(n_users, base_mean, dtype=float)

    heavy_count = int(is_heavy.sum())
    # Lognormal-ish multiplier for heavy spenders; ensure non-negative
    heavy_mult = np.maximum(
        rng.normal(loc=heavy_multiplier_mean, scale=heavy_multiplier_std, size=heavy_count),
        1.1,
    )
    factors[is_heavy] *= heavy_mult
    return factors


def generate_user_activity_weights(
    n_users: int, rng: np.random.Generator
) -> np.ndarray:
    """Per-user activity weights to skew transaction frequency by user.

    Drawn from a gamma distribution to produce a long-tailed activity pattern.
    """

    shape, scale = 1.2, 1.0
    weights = rng.gamma(shape=shape, scale=scale, size=n_users)
    # Avoid zeros so every user can be sampled
    weights = np.clip(weights, 1e-3, None)
    return weights / weights.sum()


def _month_multiplier(ts: pd.Timestamp) -> float:
    # Mild seasonality by month: Nov/Dec uplift, Jan lull
    month = int(ts.month)
    if month == 1:
        return 0.95
    if month in (11, 12):
        return 1.10 if month == 11 else 1.15
    return 1.0


def _weekday_multiplier(ts: pd.Timestamp) -> float:
    # Weekend uplift
    return 1.15 if ts.weekday() >= 5 else 1.0


def hourly_activity_weights() -> np.ndarray:
    """Return 24-length weights emphasizing daytime/evening shopping hours."""

    w = np.array([
        0.2, 0.15, 0.12, 0.12, 0.12, 0.18,  # 0-5 low
        0.35, 0.55, 0.75, 0.85, 0.95, 1.05,  # 6-11 ramp
        1.10, 1.15, 1.10, 1.05, 1.00, 0.95,  # 12-17 peak
        0.90, 0.85, 0.80, 0.70, 0.55, 0.40,  # 18-23 evening
    ], dtype=float)
    return w / w.sum()


def generate_timestamps(
    n: int,
    rng: np.random.Generator,
    start_date: str | datetime,
    end_date: datetime | None = None,
) -> np.ndarray:
    """Generate `n` timestamps between start_date and end_date with hour-of-day bias.

    Hours are sampled from `hourly_activity_weights` to capture diurnal patterns.
    """

    if isinstance(start_date, str):
        start_dt = pd.Timestamp(start_date).to_pydatetime().replace(tzinfo=timezone.utc)
    elif isinstance(start_date, datetime):
        start_dt = start_date if start_date.tzinfo else start_date.replace(tzinfo=timezone.utc)
    else:
        raise TypeError("start_date must be str or datetime")

    end_dt = end_date or datetime.now(tz=timezone.utc)
    if end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=timezone.utc)

    total_seconds = int((end_dt - start_dt).total_seconds())
    if total_seconds <= 0:
        raise ValueError("end_date must be after start_date")

    # Choose days uniformly, then hours by weighted distribution,
    # and minutes/seconds uniformly.
    days = rng.integers(0, max(1, total_seconds // 86400), size=n)
    base_dates = np.array([start_dt + timedelta(days=int(d)) for d in days], dtype=object)

    hour_probs = hourly_activity_weights()
    hours = rng.choice(24, size=n, p=hour_probs)
    minutes = rng.integers(0, 60, size=n)
    seconds = rng.integers(0, 60, size=n)

    timestamps = np.empty(n, dtype=object)
    for i in range(n):
        ts = base_dates[i].replace(hour=int(hours[i]), minute=int(minutes[i]), second=int(seconds[i]))
        timestamps[i] = pd.Timestamp(ts)
    return timestamps


def seasonality_multiplier(ts_array: Sequence[pd.Timestamp]) -> np.ndarray:
    """Compute multiplicative seasonal factors for each timestamp.

    Includes weekend and month multipliers.
    """

    mult = np.ones(len(ts_array), dtype=float)
    for i, ts in enumerate(ts_array):
        m = _weekday_multiplier(ts) * _month_multiplier(ts)
        mult[i] = m
    return mult


def category_amount_scale(categories: Sequence[str]) -> np.ndarray:
    """Return per-row base amount scales derived from category."""

    return np.array([CATEGORY_AMOUNT_SCALE.get(cat, 25.0) for cat in categories], dtype=float)


def compute_fraud_probability(
    amounts: np.ndarray, timestamps: Sequence[pd.Timestamp], base_rate: float
) -> np.ndarray:
    """Compute per-transaction fraud probability based on amount and hour.

    - Base rate is the baseline probability (e.g., 0.005 for 0.5%).
    - Increases with amount and during late-night hours.
    """

    hours = np.array([int(ts.hour) for ts in timestamps], dtype=int)
    late_night = ((hours >= 0) & (hours <= 5)).astype(float)

    # Amount sensitivity: larger amounts more likely to be fraud
    amt_component = np.clip((amounts - 200.0) / 800.0, 0.0, 1.0) * 0.02
    late_night_component = late_night * 0.015
    probs = base_rate + amt_component + late_night_component
    return np.clip(probs, 0.0, 0.5)


@dataclass(frozen=True)
class OutputPaths:
    csv_path: str | None = None
    parquet_path: str | None = None
