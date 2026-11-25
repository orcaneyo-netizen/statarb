"""Synthetic price universes with known cointegration structure.

The generator builds log prices directly, which makes the truth exact:

* every true pair shares one stochastic trend ``c_t`` (a random walk);
* the spread ``s_t = log B_t - beta log A_t - alpha`` is a stationary
  AR(1) whose half-life is drawn from ``[5, 20]`` days;
* the remaining assets are independent random walks.

Because the spread is generated as an AR(1) with ``phi = 2 ** (-1 / hl)``,
the discrete OU half-life is exactly ``hl``.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


def synthetic_pairs_universe(
    n_pairs: int = 6,
    n_extra: int = 10,
    n_days: int = 1000,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict]:
    """Generate a universe of prices with known cointegrated pairs.

    Parameters
    ----------
    n_pairs : int, default=6
        Number of embedded cointegrated pairs.
    n_extra : int, default=10
        Number of independent random-walk assets.
    n_days : int, default=1000
        Number of daily observations.
    seed : int, default=42
        Seed for ``numpy.random.default_rng``; the output is deterministic.

    Returns
    -------
    prices : pandas.DataFrame
        Price levels with columns ``A0, B0, A1, B1, ..., X0, X1, ...``.
    truth : dict
        ``pairs`` (list of ``(asset_a, asset_b)`` tuples), ``hedge_ratios``,
        ``half_lives``, ``intercepts`` (intercept in
        ``log B = alpha + beta log A + spread``), ``spreads`` (the true spread
        series) and ``phi`` (AR(1) coefficient) keyed by pair.

    Raises
    ------
    ValueError
        If any size argument is out of range.

    Notes
    -----
    Asset A is the common-trend leg and asset B the spread leg. In levels,
    ``B = exp(alpha) * A**beta * exp(spread)``, so an Engle-Granger
    regression of ``log B`` on ``log A`` recovers ``beta`` up to sampling
    error (which is small because the trend dominates the spread).
    """
    if not isinstance(n_pairs, (int, np.integer)) or n_pairs < 1:
        raise ValueError("n_pairs must be a positive integer")
    if not isinstance(n_extra, (int, np.integer)) or n_extra < 0:
        raise ValueError("n_extra must be a non-negative integer")
    if not isinstance(n_days, (int, np.integer)) or n_days < 50:
        raise ValueError("n_days must be an integer >= 50")

    rng = np.random.default_rng(seed)
    n_assets = 2 * n_pairs + n_extra
    log_prices = np.empty((n_days, n_assets), dtype=float)
    base = np.log(100.0)
    daily_trend_vol = 0.02

    truth: Dict = {
        "pairs": [],
        "hedge_ratios": {},
        "half_lives": {},
        "intercepts": {},
        "spreads": {},
        "phi": {},
    }

    columns: list[str] = []
    for i in range(n_pairs):
        trend = np.cumsum(rng.normal(0.0, daily_trend_vol, n_days))

        half_life_true = float(rng.uniform(5.0, 20.0))
        phi = float(2.0 ** (-1.0 / half_life_true))
        spread_sd = float(rng.uniform(0.005, 0.015))
        innovation_sd = spread_sd * np.sqrt(1.0 - phi**2)

        spread = np.empty(n_days, dtype=float)
        spread[0] = rng.normal(0.0, spread_sd)
        shocks = rng.normal(0.0, innovation_sd, n_days)
        for t in range(1, n_days):
            spread[t] = phi * spread[t - 1] + shocks[t]

        beta = float(rng.uniform(0.5, 2.0))
        alpha = float(rng.normal(0.0, 0.05))

        log_a = base + trend
        log_b = alpha + beta * log_a + spread
        log_prices[:, 2 * i] = log_a
        log_prices[:, 2 * i + 1] = log_b

        asset_a, asset_b = f"A{i}", f"B{i}"
        pair = (asset_a, asset_b)
        truth["pairs"].append(pair)
        truth["hedge_ratios"][pair] = beta
        truth["half_lives"][pair] = half_life_true
        truth["intercepts"][pair] = alpha
        truth["spreads"][pair] = spread
        truth["phi"][pair] = phi
        columns.extend([asset_a, asset_b])

    for j in range(n_extra):
        walk = base + np.cumsum(rng.normal(0.0, daily_trend_vol, n_days))
        log_prices[:, 2 * n_pairs + j] = walk
        columns.append(f"X{j}")

    prices = pd.DataFrame(np.exp(log_prices), columns=columns)
    return prices, truth
