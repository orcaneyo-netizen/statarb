"""Pairs-trading backtester with explicit costs and PnL attribution.

Execution convention
--------------------
A position decided on bar ``t`` (the close) earns the return from ``t`` to
``t + 1``. The implementation therefore applies ``weights[t]`` to
``returns[t + 1]``; there is no look-ahead. Costs are charged on the bar a
weight changes, at ``(commission_bps + slippage_bps)`` of traded notional per
leg.

Sizing
------
``beta-neutral`` (default)
    Long ``1 / (1 + beta)`` of leg A and short ``beta / (1 + beta)`` of leg B.
    Gross exposure is 1 and the PnL tracks the cointegrating spread
    ``log A - beta log B``. Net dollar exposure is ``(1 - beta) / (1 + beta)``,
    which is zero only when ``beta = 1``.
``dollar-neutral``
    Equal dollar legs ``(+0.5, -0.5)``. Net exposure is exactly zero, but the
    return exposure is the ``beta = 1`` spread rather than the estimated one.

Strict dollar neutrality and a regression hedge ratio cannot both hold when
``beta != 1``; the two modes make the trade-off explicit instead of hiding it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd

Sizing = Literal["beta-neutral", "dollar-neutral"]


@dataclass
class BacktestResult:
    """Result of a pairs backtest.

    Attributes
    ----------
    equity_curve : np.ndarray
        Compounded equity, starting at ``capital``.
    returns : np.ndarray
        Net period returns after costs.
    positions : np.ndarray
        Effective position per bar (0, +1, -1).
    weights_a : np.ndarray
        Portfolio weight of leg A per bar.
    weights_b : np.ndarray
        Portfolio weight of leg B per bar.
    trades : pandas.DataFrame
        One row per round trip.
    metrics : dict
        Performance summary (Sharpe, drawdown, hit rate, costs, ...).
    """

    equity_curve: np.ndarray
    returns: np.ndarray
    positions: np.ndarray
    weights_a: np.ndarray
    weights_b: np.ndarray
    trades: pd.DataFrame
    metrics: dict


class PairsBacktester:
    """Backtest a spread strategy with per-leg transaction costs.

    Parameters
    ----------
    commission_bps : float, default=1.0
        Commission per leg in basis points (1 bp = 0.01%).
    slippage_bps : float, default=1.0
        Slippage per leg in basis points.
    capital : float, default=1.0
        Starting capital; returns are scale-free.
    periods_per_year : int, default=252
        Annualization factor for Sharpe and annualized return.
    sizing : {'beta-neutral', 'dollar-neutral'}, default='beta-neutral'
        Leg-weighting convention (see module docstring).

    Raises
    ------
    ValueError
        For negative costs, non-positive capital, or an unknown sizing mode.
    """

    def __init__(
        self,
        commission_bps: float = 1.0,
        slippage_bps: float = 1.0,
        capital: float = 1.0,
        periods_per_year: int = 252,
        sizing: Sizing = "beta-neutral",
    ) -> None:
        if commission_bps < 0.0 or slippage_bps < 0.0:
            raise ValueError("commission_bps and slippage_bps must be non-negative")
        if capital <= 0.0:
            raise ValueError("capital must be positive")
        if not isinstance(periods_per_year, (int, np.integer)) or periods_per_year < 1:
            raise ValueError("periods_per_year must be a positive integer")
        if sizing not in ("beta-neutral", "dollar-neutral"):
            raise ValueError("sizing must be 'beta-neutral' or 'dollar-neutral'")

        self.commission_bps = float(commission_bps)
        self.slippage_bps = float(slippage_bps)
        self.capital = float(capital)
        self.periods_per_year = int(periods_per_year)
        self.sizing = sizing
        self.cost_rate = (self.commission_bps + self.slippage_bps) / 10_000.0

    def _leg_weights(self, position: int, beta: float) -> tuple[float, float]:
        """Leg weights for one unit of spread position."""
        if position == 0:
            return 0.0, 0.0
        if self.sizing == "beta-neutral":
            denom = 1.0 + beta
            return position / denom, -position * beta / denom
        return 0.5 * position, -0.5 * position

    def run(
        self,
        price_a: np.ndarray,
        price_b: np.ndarray,
        beta: np.ndarray,
        signal_positions: np.ndarray,
        z_scores: Optional[np.ndarray] = None,
    ) -> BacktestResult:
        """Run the backtest.

        Parameters
        ----------
        price_a : array-like
            Price levels of leg A (the spread's long leg).
        price_b : array-like
            Price levels of leg B (the hedge leg).
        beta : array-like
            Hedge ratio per bar. The value at entry is held for the life of
            the trade (no daily rebalancing).
        signal_positions : array-like
            Target position per bar: 0, +1 (long spread), -1 (short spread).
        z_scores : array-like, optional
            Z-scores aligned with the prices, stored in the trade log.

        Returns
        -------
        BacktestResult

        Raises
        ------
        ValueError
            For length mismatches, non-finite or non-positive prices, a
            non-positive hedge ratio, or invalid position values.

        Notes
        -----
        ``pnl_spread`` is the first-order log-spread contribution
        ``position * (log A_c - beta log B_c - log A_e + beta log B_e) /
        (1 + beta)``. ``pnl_drift`` is the residual between the actual
        weighted PnL and this term, so it absorbs common-trend exposure,
        log-versus-simple return differences, and hedge-ratio drift.
        ``gross_pnl = pnl_spread + pnl_drift`` exactly.
        """
        a = np.asarray(price_a, dtype=float).ravel()
        b = np.asarray(price_b, dtype=float).ravel()
        betas = np.asarray(beta, dtype=float).ravel()
        raw_positions = np.asarray(signal_positions, dtype=float).ravel()

        n = a.size
        if not (b.size == betas.size == raw_positions.size == n):
            raise ValueError("price_a, price_b, beta and signal_positions must align")
        if n < 2:
            raise ValueError("Need at least two observations to backtest")
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            raise ValueError("Prices must be finite")
        if np.any(a <= 0.0) or np.any(b <= 0.0):
            raise ValueError("Prices must be strictly positive")
        if not np.all(np.isfinite(betas)):
            raise ValueError("beta must be finite")
        if np.any(betas <= 0.0):
            raise ValueError("beta must be strictly positive")
        if not np.all(np.isin(raw_positions, (-1.0, 0.0, 1.0))):
            raise ValueError("signal_positions must contain only -1, 0, 1")
        positions = raw_positions.astype(int)

        if z_scores is None:
            z_arr = np.full(n, np.nan)
        else:
            z_arr = np.asarray(z_scores, dtype=float).ravel()
            if z_arr.size != n:
                raise ValueError("z_scores must have the same length as prices")

        ret_a = np.zeros(n)
        ret_b = np.zeros(n)
        ret_a[1:] = a[1:] / a[:-1] - 1.0
        ret_b[1:] = b[1:] / b[:-1] - 1.0

        # Effective position and entry hedge ratio (held until the trade ends).
        effective = np.zeros(n, dtype=int)
        entry_beta = np.full(n, np.nan)
        current = 0
        held_beta = np.nan
        for t in range(n):
            desired = int(positions[t])
            if desired != current:
                current = desired
                if current != 0:
                    held_beta = float(betas[t])
            effective[t] = current
            entry_beta[t] = held_beta

        weights_a = np.zeros(n)
        weights_b = np.zeros(n)
        for t in range(n):
            if effective[t] != 0:
                weights_a[t], weights_b[t] = self._leg_weights(
                    effective[t], entry_beta[t]
                )

        traded = np.abs(np.diff(weights_a, prepend=0.0)) + np.abs(
            np.diff(weights_b, prepend=0.0)
        )
        costs = self.cost_rate * traded

        gross_returns = np.zeros(n)
        gross_returns[1:] = (
            weights_a[:-1] * ret_a[1:] + weights_b[:-1] * ret_b[1:]
        )
        net_returns = gross_returns - costs
        equity = self.capital * np.cumprod(1.0 + net_returns)

        trades = self._build_trades(
            a, b, effective, entry_beta, ret_a, ret_b, z_arr
        )
        metrics = self._compute_metrics(
            equity, net_returns, trades, weights_a, weights_b, costs, traded
        )

        return BacktestResult(
            equity_curve=equity,
            returns=net_returns,
            positions=effective,
            weights_a=weights_a,
            weights_b=weights_b,
            trades=trades,
            metrics=metrics,
        )

    def _build_trades(
        self,
        price_a: np.ndarray,
        price_b: np.ndarray,
        effective: np.ndarray,
        entry_beta: np.ndarray,
        ret_a: np.ndarray,
        ret_b: np.ndarray,
        z_scores: np.ndarray,
    ) -> pd.DataFrame:
        """Extract round trips and attribute their PnL."""
        n = effective.size
        log_a = np.log(price_a)
        log_b = np.log(price_b)

        rows: list[dict] = []
        open_idx: Optional[int] = None
        open_pos = 0

        def close_trade(entry: int, exit_idx: int, pos: int, at_end: bool) -> None:
            beta_entry = float(entry_beta[entry])
            w_a, w_b = self._leg_weights(pos, beta_entry)

            # Returns from entry+1 through exit_idx were earned with these
            # weights (weights at entry apply to the next bar).
            gross = 0.0
            for u in range(entry + 1, exit_idx + 1):
                gross += w_a * ret_a[u] + w_b * ret_b[u]

            log_spread_entry = log_a[entry] - beta_entry * log_b[entry]
            log_spread_exit = log_a[exit_idx] - beta_entry * log_b[exit_idx]
            pnl_spread = pos * (log_spread_exit - log_spread_entry) / (1.0 + beta_entry)
            pnl_drift = gross - pnl_spread

            gross_exposure = abs(w_a) + abs(w_b)
            entry_cost = self.cost_rate * gross_exposure
            exit_cost = 0.0 if at_end else self.cost_rate * gross_exposure
            total_cost = entry_cost + exit_cost

            rows.append(
                {
                    "entry_idx": int(entry),
                    "exit_idx": int(exit_idx),
                    "position": int(pos),
                    "entry_beta": beta_entry,
                    "entry_z": float(z_scores[entry]),
                    "exit_z": float(z_scores[exit_idx]),
                    "gross_pnl": float(gross),
                    "pnl_spread": float(pnl_spread),
                    "pnl_drift": float(pnl_drift),
                    "costs": float(total_cost),
                    "net_pnl": float(gross - total_cost),
                    "holding_period": int(exit_idx - entry),
                    "exit_reason": "end" if at_end else "signal",
                }
            )

        for t in range(n):
            pos = int(effective[t])
            if open_idx is not None and pos != open_pos:
                close_trade(open_idx, t, open_pos, at_end=False)
                open_idx = None
            if open_idx is None and pos != 0:
                open_idx, open_pos = t, pos

        if open_idx is not None:
            close_trade(open_idx, n - 1, open_pos, at_end=True)

        columns = [
            "entry_idx", "exit_idx", "position", "entry_beta", "entry_z",
            "exit_z", "gross_pnl", "pnl_spread", "pnl_drift", "costs",
            "net_pnl", "holding_period", "exit_reason",
        ]
        return pd.DataFrame(rows, columns=columns)

    def _compute_metrics(
        self,
        equity: np.ndarray,
        returns: np.ndarray,
        trades: pd.DataFrame,
        weights_a: np.ndarray,
        weights_b: np.ndarray,
        costs: np.ndarray,
        traded: np.ndarray,
    ) -> dict:
        """Compute performance metrics. Undefined ratios are reported as 0."""
        n = returns.size
        total_return = float(equity[-1] / equity[0] - 1.0)

        if equity[-1] > 0.0:
            annualized = float((equity[-1] / equity[0]) ** (self.periods_per_year / n) - 1.0)
        else:
            annualized = -1.0

        sd = float(np.std(returns, ddof=1)) if n > 1 else 0.0
        sharpe = (
            float(np.mean(returns) / sd * np.sqrt(self.periods_per_year))
            if sd > 0.0
            else 0.0
        )

        running_max = np.maximum.accumulate(equity)
        max_dd = float(np.min(equity / running_max - 1.0))

        if len(trades):
            hit_rate = float((trades["net_pnl"] > 0.0).mean())
            avg_holding = float(trades["holding_period"].mean())
            avg_pnl = float(trades["net_pnl"].mean())
            gross_profit = float(trades.loc[trades["net_pnl"] > 0.0, "net_pnl"].sum())
            gross_loss = float(-trades.loc[trades["net_pnl"] < 0.0, "net_pnl"].sum())
            if gross_loss > 0.0:
                profit_factor = gross_profit / gross_loss
            else:
                profit_factor = float("inf") if gross_profit > 0.0 else 0.0
        else:
            hit_rate = avg_holding = avg_pnl = 0.0
            profit_factor = 0.0

        active = weights_a + weights_b
        in_position = np.abs(weights_a) + np.abs(weights_b) > 0.0
        net_exposure = float(np.mean(active[in_position])) if np.any(in_position) else 0.0
        avg_gross = float(np.mean(np.abs(weights_a) + np.abs(weights_b)))

        return {
            "total_return": total_return,
            "annualized_return": annualized,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "hit_rate": hit_rate,
            "avg_holding_period": avg_holding,
            "n_trades": int(len(trades)),
            "avg_pnl_per_trade": avg_pnl,
            "profit_factor": float(profit_factor),
            "total_costs": float(costs.sum()),
            "turnover": float(traded.sum()),
            "net_exposure": net_exposure,
            "avg_gross_exposure": avg_gross,
        }
