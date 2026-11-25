# statarb — statistical arbitrage & pairs trading

Read `..\CONVENTIONS.md` first. Package name: `statarb`.

## Goal

A research + backtest toolkit for relative-value trading: screen a universe for
cointegrated pairs, estimate a *time-varying* hedge ratio with a Kalman filter,
size positions from an Ornstein-Uhlenbeck spread model, and backtest the spread
strategy with realistic costs.

## Modules

`src/quantarb/` → **package directory is `src/statarb/`**

1. `data.py` — `synthetic_pairs_universe(n_pairs=6, n_extra=10, n_days=1000,
   seed=42)` producing prices where a known subset is cointegrated (shared
   stochastic trend + OU spread with half-life 5-20 days) and the rest are
   independent random walks. Returns `(prices: DataFrame, truth: dict)` with the
   true pairs, hedge ratios, half-lives — so tests can check recovery.
2. `coint.py` — pure-numpy implementations with documented math:
   - `adf_test(series, max_lag=None, regression='c')` → dataclass
     `ADFResult(stat, pvalue_approx, used_lag, critical)` (MacKinnon
     interpolation table for p-values, or a documented response-surface approx).
   - `engle_granger(y, x)` → residual ADF stat, hedge ratio, intercept.
   - `johansen(log_prices, det_order=0, k_ar_diff=1)` → trace + max-eigen
     statistics and critical values (implement the eigen-decomposition of the
     residual moment matrices).
   - `half_life(spread)` — OU: `HL = -ln(2) / ln(1 + beta)` from the AR(1)
     regression of Δspread on lagged spread (document, guard beta >= 0).
3. `kalman.py` — `KalmanHedgeRatio` (state = [beta, alpha], random-walk state,
   scalar observation `y_t = beta_t * x_t + alpha_t + eps`): `filter(y, x)` →
   arrays of filtered beta/alpha and the innovation (spread) + its variance;
   `mle_fit` optional via `scipy.optimize.minimize` on the log-likelihood
   (delta and observation variance).
4. `signal.py` — `ZScoreSignal(entry=2.0, exit=0.5, stop=4.0, lookback=60)`:
   rolling z-score of the spread with **causal** rolling stats, hysteresis state
   machine (FLAT → LONG_SPREAD / SHORT_SPREAD → FLAT), explicit rules for
   entry/exit/stop, no repainting.
5. `backtest.py` — `PairsBacktester`: dollar-neutral leg sizing from beta,
   per-leg commission + slippage (reuse simple bps model), equity curve, trade
   log with entry/exit z, PnL attribution (spread convergence vs. drift vs.
   costs), metrics (Sharpe, max DD, hit rate, avg holding period).
6. `screen.py` — `screen_pairs(prices, method='engle-granger'|'distance',
   min_obs, max_pairs)` → ranked `DataFrame` (pair, stat, pvalue, hedge ratio,
   half-life, corr) with p-value and half-life filters; deterministic ordering.
7. `__main__.py` — `demo`: synthetic universe → screen → backtest top 3 pairs →
   print table (pair, half-life, trades, Sharpe, max DD) + aggregate.

## Tests

- `test_data.py`: known cointegrated pairs really are (EG p-value small,
  half-life within ±50% of truth), independent pairs are not, determinism.
- `test_coint.py`: ADF on white noise vs. random walk vs. mean-reverting AR(1)
  gives the expected ordering; EG recovers hedge ratio on a synthetic linear
  relationship within 1%; johansen finds rank 1 for one cointegrated pair,
  rank 0 for independent walks; `half_life` matches a simulated OU's truth.
- `test_kalman.py`: filter tracks a constant beta; adapts to a beta that steps
  at t=500 (late-window error < early-window error); innovation ~0 mean when the
  model is correct.
- `test_signal.py`: constructed spread path triggers exactly one entry and one
  exit at the documented z levels; stop-loss fires at `stop`; no signal before
  warm-up.
- `test_backtest.py`: a mean-reverting spread with a perfect signal yields
  positive PnL before costs and lower PnL after costs; dollar-neutrality means
  net exposure ≈ 0 at entry.
- `test_screen.py`: truth pairs rank in the top-k; deterministic output.

## README sections

Strategy thesis, math (ADF/EG/Johansen/OU/Kalman equations), signal rules table,
backtest assumptions, cost sensitivity, known failure modes (regime break,
beta drift, crowded pairs).
