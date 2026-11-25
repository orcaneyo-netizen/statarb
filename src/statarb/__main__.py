"""Command-line entry point: ``python -m statarb demo``.

The demo runs the full research loop on a synthetic universe:

1. generate cointegrated pairs plus independent random walks;
2. screen all pairs with Engle-Granger;
3. estimate a time-varying hedge ratio with the Kalman filter;
4. trade the causal z-score of the filtered spread;
5. print a per-pair table and an aggregate summary.

Optional ``--plot FILE`` writes an equity-curve PNG instead of opening a
window (the default is no plot at all).
"""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .backtest import PairsBacktester
from .data import synthetic_pairs_universe
from .kalman import KalmanHedgeRatio
from .screen import screen_pairs
from .signal import ZScoreSignal


def _format_table(rows: list[dict]) -> str:
    """Render the per-pair result table as plain text."""
    header = f"{'pair':<10} {'HL':>6} {'trades':>6} {'Sharpe':>7} {'maxDD':>8} {'return':>8}"
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{row['pair']:<10} "
            f"{row['half_life']:>6.1f} "
            f"{row['trades']:>6d} "
            f"{row['sharpe']:>7.2f} "
            f"{row['max_dd']:>7.1%} "
            f"{row['total_return']:>7.1%}"
        )
    return "\n".join(lines)


def run_demo(
    seed: int = 42,
    n_pairs: int = 6,
    n_extra: int = 10,
    n_days: int = 1000,
    top: int = 3,
    plot: Optional[str] = None,
) -> int:
    """Run the end-to-end synthetic demo and print a report.

    Parameters
    ----------
    seed : int, default=42
        Seed for the synthetic universe.
    n_pairs, n_extra, n_days : int
        Universe dimensions.
    top : int, default=3
        Number of screened pairs to backtest.
    plot : str, optional
        Path for an equity-curve PNG.

    Returns
    -------
    int
        Process exit code (0 on success).
    """
    prices, truth = synthetic_pairs_universe(
        n_pairs=n_pairs, n_extra=n_extra, n_days=n_days, seed=seed
    )
    n_assets = prices.shape[1]
    n_candidates = n_assets * (n_assets - 1) // 2

    table = screen_pairs(
        prices,
        method="engle-granger",
        min_obs=60,
        max_pairs=top,
        max_pvalue=0.10,
        max_half_life=60.0,
    )

    print(
        f"statarb demo | seed={seed} | {n_pairs} true pairs + {n_extra} walks "
        f"| {n_days} days | {n_assets} assets / {n_candidates} candidate pairs"
    )
    if table.empty:
        print("No pair passed the screen; nothing to backtest.")
        return 1

    print(f"\nScreened pairs (Engle-Granger, top {len(table)}):")
    print(
        table[["pair", "stat", "pvalue", "hedge_ratio", "half_life", "corr"]]
        .to_string(index=False, float_format=lambda v: f"{v:.4f}")
    )

    backtester = PairsBacktester(
        commission_bps=1.0, slippage_bps=1.0, sizing="beta-neutral"
    )
    signal_model = ZScoreSignal(entry=2.0, exit=0.5, stop=4.0, lookback=60)

    rows: list[dict] = []
    curves: dict[str, np.ndarray] = {}
    for _, pair_row in table.iterrows():
        asset_y = str(pair_row["asset_y"])
        asset_x = str(pair_row["asset_x"])
        log_y = np.log(prices[asset_y].to_numpy(dtype=float))
        log_x = np.log(prices[asset_x].to_numpy(dtype=float))

        # Centre to remove the price level (improves identification of beta)
        # and initialise the state from a causal window of past data only.
        y = log_y - log_y[0]
        x = log_x - log_x[0]
        window = signal_model.lookback
        design = np.column_stack([np.ones(window), x[:window]])
        init = np.linalg.lstsq(design, y[:window], rcond=None)[0]

        kalman = KalmanHedgeRatio(
            delta=1e-5,
            obs_noise=1e-2,
            initial_beta=float(init[1]),
            initial_alpha=float(init[0]),
            initial_cov=1e-4,
        )
        kf = kalman.filter(y, x)
        spread = y - kf.beta * x - kf.alpha
        signals = signal_model.generate(spread)

        beta_series = pd.Series(kf.beta).ffill().bfill().to_numpy(dtype=float)
        beta_series = np.clip(beta_series, 1e-3, None)

        result = backtester.run(
            prices[asset_y].to_numpy(dtype=float),
            prices[asset_x].to_numpy(dtype=float),
            beta_series,
            signals.positions,
            signals.z_scores,
        )
        metrics = result.metrics
        rows.append(
            {
                "pair": str(pair_row["pair"]),
                "half_life": float(pair_row["half_life"]),
                "trades": int(metrics["n_trades"]),
                "sharpe": float(metrics["sharpe"]),
                "max_dd": float(metrics["max_drawdown"]),
                "total_return": float(metrics["total_return"]),
                "hit_rate": float(metrics["hit_rate"]),
                "costs": float(metrics["total_costs"]),
            }
        )
        curves[str(pair_row["pair"])] = result.equity_curve

    print("\nBacktest (Kalman hedge ratio, z-score entry 2 / exit 0.5 / stop 4):")
    print(_format_table(rows))

    total_trades = sum(r["trades"] for r in rows)
    mean_sharpe = float(np.mean([r["sharpe"] for r in rows]))
    mean_return = float(np.mean([r["total_return"] for r in rows]))
    worst_dd = float(np.min([r["max_dd"] for r in rows]))
    mean_hit = float(np.mean([r["hit_rate"] for r in rows]))
    total_costs = float(np.sum([r["costs"] for r in rows]))

    print("\nAggregate:")
    print(
        f"  pairs={len(rows)}  trades={total_trades}  mean Sharpe={mean_sharpe:.2f}  "
        f"mean return={mean_return:.1%}  worst maxDD={worst_dd:.1%}"
    )
    print(f"  mean hit rate={mean_hit:.1%}  total costs={total_costs:.2%} of capital")

    truth_sets = {frozenset(pair) for pair in truth["pairs"]}
    found = {
        frozenset((str(row["asset_y"]), str(row["asset_x"]))) for _, row in table.iterrows()
    }
    recovered = len(truth_sets & found)
    print(f"  true pairs in the screened top-{len(table)}: {recovered}/{len(truth_sets)}")

    if plot:
        _save_plot(curves, plot)
        print(f"  equity plot written to {plot}")

    return 0


def _save_plot(curves: dict[str, np.ndarray], path: str) -> None:
    """Save normalized equity curves to ``path`` using the Agg backend."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for label, curve in curves.items():
        ax.plot(np.asarray(curve) / curve[0], label=label, linewidth=1.4)
    ax.set_title("statarb demo — equity curves (net of costs)")
    ax.set_xlabel("bar")
    ax.set_ylabel("growth of 1.0")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="statarb",
        description="Statistical arbitrage toolkit (synthetic demo).",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=["demo"],
        default="demo",
        help="command to run (default: demo)",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--pairs", type=int, default=6, help="number of true pairs")
    parser.add_argument("--extra", type=int, default=10, help="independent walks")
    parser.add_argument("--days", type=int, default=1000, help="sample length")
    parser.add_argument("--top", type=int, default=3, help="pairs to backtest")
    parser.add_argument("--plot", type=str, default=None, help="save equity PNG here")
    args = parser.parse_args(argv)

    return run_demo(
        seed=args.seed,
        n_pairs=args.pairs,
        n_extra=args.extra,
        n_days=args.days,
        top=args.top,
        plot=args.plot,
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
