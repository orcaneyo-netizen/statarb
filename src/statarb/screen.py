"""Universe screening for cointegrated pairs.

Two screening statistics are supported:

``engle-granger``
    For each pair, regress ``log y`` on ``log x`` and run an ADF test on the
    residuals. The statistic is the ADF t-statistic (more negative is better)
    and the p-value comes from the N = 2 cointegration response surface.

``distance``
    Normalize each log price series to zero mean / unit variance over the
    sample and score the pair by the root-mean-square distance between the
    two normalized series (Gatev-style distance method). Smaller is better;
    no p-value is produced.

Screening is a *research* step and therefore uses full-sample statistics.
The trading signal must still be causal (see :mod:`statarb.signal`).
"""

from __future__ import annotations

from itertools import combinations
from typing import Literal, Optional

import numpy as np
import pandas as pd

from .coint import engle_granger, half_life

Method = Literal["engle-granger", "distance"]

_COLUMNS = [
    "pair",
    "asset_y",
    "asset_x",
    "stat",
    "pvalue",
    "hedge_ratio",
    "half_life",
    "corr",
]


def screen_pairs(
    prices: pd.DataFrame,
    method: Method = "engle-granger",
    min_obs: int = 60,
    max_pairs: Optional[int] = None,
    max_pvalue: float = 0.05,
    max_half_life: Optional[float] = 60.0,
    min_corr: float = 0.0,
    use_log_prices: bool = True,
) -> pd.DataFrame:
    """Rank all asset pairs by cointegration strength or price distance.

    Parameters
    ----------
    prices : pandas.DataFrame
        Price levels (or log prices if ``use_log_prices=False``), one column
        per asset. Values must be finite and strictly positive.
    method : {'engle-granger', 'distance'}, default='engle-granger'
        Screening statistic.
    min_obs : int, default=60
        Minimum number of paired observations required for a pair to be
        considered.
    max_pairs : int, optional
        Keep only the best ``max_pairs`` rows after filtering.
    max_pvalue : float, default=0.05
        Engle-Granger only: drop pairs with a larger residual ADF p-value.
    max_half_life : float, optional, default=60.0
        Drop pairs whose estimated half-life exceeds this many periods (or
        does not converge). ``None`` disables the filter.
    min_corr : float, default=0.0
        Drop pairs whose log-price correlation is below this value.
    use_log_prices : bool, default=True
        Convert levels to logs before testing. Set to False when the frame
        already contains log prices.

    Returns
    -------
    pandas.DataFrame
        Columns ``pair, asset_y, asset_x, stat, pvalue, hedge_ratio,
        half_life, corr`` sorted deterministically. For a pair ``(x, y)`` in
        column order the cointegrating regression is ``y = alpha +
        hedge_ratio * x + spread``; ``pair`` is the label ``"y/x"``. The
        distance method reports ``pvalue = NaN``.

    Raises
    ------
    ValueError
        For an unknown method, a non-numeric / non-positive frame, fewer than
        two assets, or invalid filter values.

    Notes
    -----
    Pairs are formed in column order, so the dependent leg is always the
    later column. This fixed convention makes the output reproducible and,
    for the synthetic universe (``A_i`` then ``B_i``), recovers the data
    generating hedge ratio ``B = alpha + beta A``.
    """
    if method not in ("engle-granger", "distance"):
        raise ValueError("method must be 'engle-granger' or 'distance'")
    if not isinstance(prices, pd.DataFrame):
        raise ValueError("prices must be a pandas DataFrame")
    if prices.shape[1] < 2:
        raise ValueError("Need at least two assets to form pairs")
    if not isinstance(min_obs, (int, np.integer)) or min_obs < 10:
        raise ValueError("min_obs must be an integer >= 10")
    if max_pairs is not None and (
        not isinstance(max_pairs, (int, np.integer)) or max_pairs < 1
    ):
        raise ValueError("max_pairs must be a positive integer or None")
    if not 0.0 < max_pvalue <= 1.0:
        raise ValueError("max_pvalue must be in (0, 1]")
    if max_half_life is not None and max_half_life <= 0:
        raise ValueError("max_half_life must be positive or None")
    if not -1.0 <= min_corr <= 1.0:
        raise ValueError("min_corr must be in [-1, 1]")

    values = prices.to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("prices contains NaN or infinite values")
    if use_log_prices and np.any(values <= 0.0):
        raise ValueError("prices must be strictly positive")

    log_values = np.log(values) if use_log_prices else values
    columns = list(prices.columns)
    n_obs = log_values.shape[0]
    if n_obs < min_obs:
        raise ValueError(
            f"min_obs={min_obs} exceeds the number of observations ({n_obs})"
        )
    constant = [columns[k] for k in range(len(columns)) if np.ptp(log_values[:, k]) == 0.0]
    if constant:
        raise ValueError(f"Constant price series cannot be screened: {constant}")

    rows: list[dict] = []
    for i, j in combinations(range(len(columns)), 2):
        x = log_values[:, i]
        y = log_values[:, j]
        pair_label = f"{columns[j]}/{columns[i]}"

        corr = _safe_corr(x, y)
        if corr < min_corr:
            continue

        if method == "engle-granger":
            eg = engle_granger(y, x)
            stat = eg.adf_stat
            pvalue = eg.adf_pvalue
            beta = eg.hedge_ratio
            spread = eg.residuals
        else:
            x_std, y_std = _standardize(x), _standardize(y)
            stat = float(np.sqrt(np.mean((x_std - y_std) ** 2)))
            pvalue = float("nan")
            slope, intercept = np.polyfit(x, y, 1)
            beta = float(slope)
            spread = y - (intercept + slope * x)

        hl = half_life(spread)
        if max_half_life is not None and (
            not hl.converging or hl.half_life > max_half_life
        ):
            continue
        if method == "engle-granger" and pvalue > max_pvalue:
            continue

        rows.append(
            {
                "pair": pair_label,
                "asset_y": columns[j],
                "asset_x": columns[i],
                "stat": float(stat),
                "pvalue": float(pvalue),
                "hedge_ratio": float(beta),
                "half_life": float(hl.half_life),
                "corr": float(corr),
            }
        )

    frame = pd.DataFrame(rows, columns=_COLUMNS)
    if frame.empty:
        return frame

    if method == "engle-granger":
        frame = frame.sort_values(
            ["pvalue", "stat", "half_life", "pair"],
            ascending=[True, True, True, True],
            kind="mergesort",
        )
    else:
        frame = frame.sort_values(
            ["stat", "half_life", "pair"],
            ascending=[True, True, True],
            kind="mergesort",
        )
    frame = frame.reset_index(drop=True)

    if max_pairs is not None:
        frame = frame.iloc[: int(max_pairs)].reset_index(drop=True)
    return frame


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation, returning 0.0 for a zero-variance series."""
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _standardize(x: np.ndarray) -> np.ndarray:
    """Zero-mean / unit-variance transform; constant input becomes zeros."""
    sd = float(np.std(x))
    if sd == 0.0:
        return np.zeros_like(x)
    return (x - np.mean(x)) / sd
