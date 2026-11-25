"""Cointegration tests and Ornstein-Uhlenbeck half-life estimation.

Everything is implemented with NumPy/SciPy; no statsmodels dependency.

Tests implemented
-----------------
* Augmented Dickey-Fuller (ADF) unit-root test with AIC lag selection and
  MacKinnon (1994/2010) p-values and critical values.
* Engle-Granger two-step cointegration test, using the *cointegration*
  (N = 2) response surface for the residual ADF statistic.
* Johansen reduced-rank test with the eigen-decomposition of the residual
  moment matrices and Osterwald-Lenum critical values.
* OU half-life from the discrete AR(1) representation of the spread.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from scipy import linalg, stats

from ._tables import (
    CRIT_2010_N1,
    CRIT_2010_N2_C,
    LARGE_P,
    LARGE_P_N2_C,
    MAXEIG_CRIT,
    SMALL_P,
    SMALL_P_N2_C,
    TAU_MAX,
    TAU_MAX_N2_C,
    TAU_MIN,
    TAU_MIN_N2_C,
    TAU_STAR,
    TAU_STAR_N2_C,
    TRACE_CRIT,
)

Regression = Literal["c", "ct", "ctt", "n"]


@dataclass
class ADFResult:
    """Result of an Augmented Dickey-Fuller test.

    Attributes
    ----------
    stat : float
        ADF t-statistic on the lagged level (more negative = more stationary).
    pvalue_approx : float
        MacKinnon (1994) response-surface p-value.
    used_lag : int
        Number of lagged differences selected by AIC.
    critical : dict
        Critical values at the 1%, 5% and 10% levels (finite-sample corrected).
    """

    stat: float
    pvalue_approx: float
    used_lag: int
    critical: dict


@dataclass
class EGResult:
    """Result of the Engle-Granger two-step cointegration test.

    Attributes
    ----------
    adf_stat : float
        ADF statistic on the cointegrating residuals.
    adf_pvalue : float
        Approximate p-value using the N = 2 cointegration response surface.
    hedge_ratio : float
        Slope ``beta`` in ``y = alpha + beta * x + u``.
    intercept : float
        Intercept ``alpha``.
    residuals : np.ndarray
        Residuals ``u_t`` (the spread).
    """

    adf_stat: float
    adf_pvalue: float
    hedge_ratio: float
    intercept: float
    residuals: np.ndarray


@dataclass
class JohansenResult:
    """Result of the Johansen cointegration rank test.

    Attributes
    ----------
    trace_stat : np.ndarray
        Trace statistics for H0: rank <= r, r = 0 .. p-1.
    trace_crit : np.ndarray
        Trace critical values, shape (p, 3), columns 90%/95%/99%.
    maxeig_stat : np.ndarray
        Maximum-eigenvalue statistics for H0: rank <= r.
    maxeig_crit : np.ndarray
        Max-eigenvalue critical values, shape (p, 3).
    eigenvectors : np.ndarray
        Cointegrating vectors in columns, normalized so ``v' S11 v = 1``.
    eigenvalues : np.ndarray
        Eigenvalues of the Johansen problem in decreasing order.
    """

    trace_stat: np.ndarray
    trace_crit: np.ndarray
    maxeig_stat: np.ndarray
    maxeig_crit: np.ndarray
    eigenvectors: np.ndarray
    eigenvalues: np.ndarray

    def rank(self, signif: float = 0.05, method: Literal["trace", "maxeig"] = "trace") -> int:
        """Cointegration rank implied by the sequential test.

        Starting at ``r = 0``, reject while the statistic exceeds its critical
        value; the first non-rejection is the estimated rank.
        """
        if signif not in (0.01, 0.05, 0.10):
            raise ValueError("signif must be one of 0.01, 0.05, 0.10")
        if method not in ("trace", "maxeig"):
            raise ValueError("method must be 'trace' or 'maxeig'")
        column = {0.10: 0, 0.05: 1, 0.01: 2}[signif]
        stat = self.trace_stat if method == "trace" else self.maxeig_stat
        crit = self.trace_crit if method == "trace" else self.maxeig_crit
        r = 0
        while r < len(stat) and stat[r] > crit[r, column]:
            r += 1
        return int(r)


@dataclass
class HalfLifeResult:
    """Result of OU half-life estimation.

    Attributes
    ----------
    half_life : float
        Half-life in periods; ``inf`` when the process does not converge.
    beta : float
        Slope from regressing ``d spread_t`` on ``spread_{t-1}``.
    converging : bool
        True when ``-1 < beta < 0``.
    """

    half_life: float
    beta: float
    converging: bool


# ---------------------------------------------------------------------------
# ADF
# ---------------------------------------------------------------------------


def _mackinnon_pvalue(stat: float, regression: Regression, n_vars: int) -> float:
    """MacKinnon (1994) response-surface p-value.

    ``n_vars`` is the number of I(1) series in the test: 1 for a plain ADF
    test, 2 for an Engle-Granger residual test with one regressor.
    """
    if n_vars == 1:
        lo, hi, star = TAU_MIN[regression], TAU_MAX[regression], TAU_STAR[regression]
        coef = SMALL_P[regression] if stat <= star else LARGE_P[regression]
    elif n_vars == 2 and regression == "c":
        lo, hi, star = TAU_MIN_N2_C, TAU_MAX_N2_C, TAU_STAR_N2_C
        coef = SMALL_P_N2_C if stat <= star else LARGE_P_N2_C
    else:
        raise ValueError("p-value surface only supports N=1 or N=2 with regression='c'")

    if stat > hi:
        return 1.0
    if stat < lo:
        return 0.0
    z = float(np.polyval(coef[::-1], stat))
    return float(stats.norm.cdf(z))


def _mackinnon_critical(
    regression: Regression, nobs: int, n_vars: int
) -> dict:
    """MacKinnon (2010) critical values with a 1/T finite-sample correction."""
    if n_vars == 1:
        table = CRIT_2010_N1[regression]
    elif n_vars == 2 and regression == "c":
        table = CRIT_2010_N2_C
    else:
        raise ValueError("critical values only support N=1 or N=2 with regression='c'")

    inv_t = 1.0 / max(int(nobs), 1)
    return {
        level: float(np.polyval(table[j][::-1], inv_t))
        for j, level in enumerate((1, 5, 10))
    }


def _adf_design(
    series: np.ndarray, lag: int, regression: Regression
) -> Optional[tuple[np.ndarray, np.ndarray, int]]:
    """Build the ADF regression for one lag order.

    With ``m = n - 1 - lag`` observations the regression is

        dy_t = d_t' gamma + rho * y_{t-1} + sum_{i=1..lag} phi_i dy_{t-i} + e_t

    where ``d_t`` collects the deterministic terms.
    """
    n = series.size
    dy = np.diff(series)
    m = n - 1 - lag
    if m <= 0:
        return None

    columns: list[np.ndarray] = []
    y_index = 0
    if regression in ("c", "ct", "ctt"):
        columns.append(np.ones(m))
        y_index += 1
    if regression in ("ct", "ctt"):
        trend = np.arange(1, m + 1, dtype=float)
        columns.append(trend)
        y_index += 1
        if regression == "ctt":
            columns.append(trend**2)
            y_index += 1

    columns.append(series[lag : n - 1])
    y_index_used = y_index
    for i in range(1, lag + 1):
        columns.append(dy[lag - i : n - 1 - i])

    return np.column_stack(columns), dy[lag:], y_index_used


def _adf_fit(
    series: np.ndarray, max_lag: int, regression: Regression
) -> tuple[int, float, np.ndarray, int]:
    """Select the ADF lag by AIC and return (lag, t-stat, residuals, nobs)."""
    best: Optional[tuple[float, int, float, np.ndarray, int]] = None
    for lag in range(max_lag + 1):
        design = _adf_design(series, lag, regression)
        if design is None:
            continue
        X, y, y_index = design
        m, k = X.shape
        if m - k < 1:
            continue

        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ coef
        dof = m - k
        sigma2 = float(resid @ resid / dof)
        if not np.isfinite(sigma2):
            continue

        xtx_inv = np.linalg.pinv(X.T @ X)
        var = sigma2 * float(xtx_inv[y_index, y_index])
        if var > 0.0:
            stat = float(coef[y_index] / np.sqrt(var))
        else:  # perfect fit / degenerate design
            stat = -np.inf if coef[y_index] < 0 else np.inf

        aic = m * np.log(max(sigma2, 1e-300)) + 2.0 * k
        if best is None or aic < best[0]:
            best = (aic, lag, stat, resid, m)

    if best is None:
        raise ValueError("ADF regression could not be estimated; series is too short")

    _, lag, stat, resid, m = best
    return lag, stat, resid, m


def adf_test(
    series: np.ndarray,
    max_lag: Optional[int] = None,
    regression: Regression = "c",
) -> ADFResult:
    """Augmented Dickey-Fuller unit-root test.

    H0: the series has a unit root (is I(1)); H1: it is stationary.

    The lagged differences are selected by minimizing
    ``AIC = m log(sigma2) + 2k`` over ``0 .. max_lag``.

    Parameters
    ----------
    series : array-like
        Series to test. NaNs are removed (the test is not designed for gaps).
    max_lag : int, optional
        Maximum augmentation order. Defaults to the Schwert rule
        ``floor(12 * (n/100)**0.25)``.
    regression : {'c', 'ct', 'ctt', 'n'}, default='c'
        Deterministic terms: constant, constant+trend, both trends, or none.

    Returns
    -------
    ADFResult

    Raises
    ------
    ValueError
        If the series is too short, constant, or ``max_lag`` is invalid.
    """
    if regression not in ("c", "ct", "ctt", "n"):
        raise ValueError("regression must be one of 'c', 'ct', 'ctt', 'n'")

    values = np.asarray(series, dtype=float).ravel()
    values = values[np.isfinite(values)]
    n = values.size
    if n < 10:
        raise ValueError("ADF test needs at least 10 observations")
    if np.ptp(values) == 0.0:
        raise ValueError("ADF test is undefined for a constant series")

    if max_lag is None:
        max_lag = int(np.floor(12.0 * (n / 100.0) ** 0.25))
    if not isinstance(max_lag, (int, np.integer)) or max_lag < 0:
        raise ValueError("max_lag must be a non-negative integer")
    if max_lag > (n - 1) // 3:
        raise ValueError("max_lag is too large for the sample size")

    used_lag, stat, _, nobs = _adf_fit(values, int(max_lag), regression)

    return ADFResult(
        stat=float(stat),
        pvalue_approx=_mackinnon_pvalue(stat, regression, n_vars=1),
        used_lag=int(used_lag),
        critical=_mackinnon_critical(regression, nobs, n_vars=1),
    )


# ---------------------------------------------------------------------------
# Engle-Granger
# ---------------------------------------------------------------------------


def engle_granger(y: np.ndarray, x: np.ndarray) -> EGResult:
    """Engle-Granger two-step cointegration test.

    Step 1 regresses ``y_t = alpha + beta x_t + u_t`` by OLS. Step 2 runs an
    ADF test (with constant, AIC lag selection) on ``u_t``. Because ``beta``
    is estimated, the statistic is compared against the cointegration
    response surface (N = 2), not the plain ADF surface.

    Parameters
    ----------
    y : array-like
        Dependent log price.
    x : array-like
        Regressor log price.

    Returns
    -------
    EGResult

    Notes
    -----
    The sign convention is ``spread = y - alpha - beta * x``: a positive
    spread means ``y`` is rich relative to ``x``. Pairs where the two series
    are swapped estimate the reciprocal hedge ratio, so callers should fix a
    convention (see :func:`statarb.screen.screen_pairs`).
    """
    y_arr = np.asarray(y, dtype=float).ravel()
    x_arr = np.asarray(x, dtype=float).ravel()
    if y_arr.size != x_arr.size:
        raise ValueError("y and x must have the same length")

    mask = np.isfinite(y_arr) & np.isfinite(x_arr)
    y_arr, x_arr = y_arr[mask], x_arr[mask]
    if y_arr.size < 20:
        raise ValueError("Engle-Granger test needs at least 20 paired observations")
    if np.ptp(x_arr) == 0.0:
        raise ValueError("Regressor x is constant; hedge ratio is undefined")

    design = np.column_stack([np.ones_like(x_arr), x_arr])
    coef, *_ = np.linalg.lstsq(design, y_arr, rcond=None)
    alpha, beta = float(coef[0]), float(coef[1])
    residuals = y_arr - design @ coef

    if np.ptp(residuals) == 0.0:
        raise ValueError("Cointegrating residuals are constant; test is undefined")

    max_lag = int(np.floor(12.0 * (residuals.size / 100.0) ** 0.25))
    max_lag = min(max_lag, (residuals.size - 1) // 3)
    lag, stat, _, nobs = _adf_fit(residuals, max_lag, "c")

    return EGResult(
        adf_stat=float(stat),
        adf_pvalue=_mackinnon_pvalue(stat, "c", n_vars=2),
        hedge_ratio=beta,
        intercept=alpha,
        residuals=residuals,
    )


# ---------------------------------------------------------------------------
# Johansen
# ---------------------------------------------------------------------------


def johansen(
    log_prices: np.ndarray,
    det_order: int = 0,
    k_ar_diff: int = 1,
) -> JohansenResult:
    """Johansen cointegration rank test.

    The VECM is

        dY_t = Pi Y_{t-1} + sum_{i=1..k} Gamma_i dY_{t-i} + mu + e_t

    with ``Pi = alpha beta'``. The rank of ``Pi`` is the number of
    cointegrating relations. After partialling out the lagged differences and
    deterministic terms, the squared canonical correlations are the
    eigenvalues of

        S01 S00^{-1} S10 v = lambda S11 v

    solved with a symmetric-definite generalized eigen-decomposition. The
    statistics are

        trace_r    = -T sum_{i>r} log(1 - lambda_i)
        maxeig_r   = -T log(1 - lambda_{r+1})

    Parameters
    ----------
    log_prices : array-like, shape (n_obs, n_vars)
        Log prices, one column per asset.
    det_order : {-1, 0, 1}, default=0
        Deterministic terms: -1 none, 0 restricted constant, 1 constant plus
        linear trend (statsmodels convention).
    k_ar_diff : int, default=1
        Number of lagged differences in the VECM.

    Returns
    -------
    JohansenResult

    Raises
    ------
    ValueError
        For non-finite input, fewer than two variables, invalid orders, or an
        insufficient sample.

    Notes
    -----
    Critical values are Osterwald-Lenum (1992) as tabulated by
    MacKinnon-Haug-Michelis (1999). For the test of rank <= r in a p-variable
    system the appropriate table row is the one for dimension ``p - r``.
    """
    data = np.asarray(log_prices, dtype=float)
    if data.ndim != 2:
        raise ValueError("log_prices must be a 2-D array (n_obs, n_vars)")
    if not np.all(np.isfinite(data)):
        raise ValueError("log_prices contains non-finite values")
    n_obs, n_vars = data.shape
    if n_vars < 2:
        raise ValueError("Johansen test needs at least 2 variables")
    if n_vars > 12:
        raise ValueError("Critical values are tabulated only up to 12 variables")
    if det_order not in (-1, 0, 1):
        raise ValueError("det_order must be -1, 0 or 1")
    if not isinstance(k_ar_diff, (int, np.integer)) or k_ar_diff < 0:
        raise ValueError("k_ar_diff must be a non-negative integer")

    k = int(k_ar_diff)
    m = n_obs - k - 1
    if m < max(20, 4 * n_vars):
        raise ValueError(
            "Not enough observations for Johansen test: need at least "
            f"{max(20, 4 * n_vars) + k + 1}, got {n_obs}"
        )

    dy = np.diff(data, axis=0)
    dy_used = dy[k:]  # dY_t, t = k+1 .. n-1
    levels = data[k:-1]  # Y_{t-1}
    short_run = [dy[k - i : n_obs - 1 - i] for i in range(1, k + 1)]

    deterministic: list[np.ndarray] = []
    if det_order == 0:
        deterministic.append(np.ones((m, 1)))
    elif det_order == 1:
        deterministic.append(np.ones((m, 1)))
        deterministic.append(np.arange(1, m + 1, dtype=float).reshape(-1, 1))

    z2_parts = [*short_run, *deterministic]
    z2 = np.column_stack(z2_parts) if z2_parts else None

    def residualize(matrix: np.ndarray) -> np.ndarray:
        if z2 is None:
            return matrix
        coef, *_ = np.linalg.lstsq(z2, matrix, rcond=None)
        return matrix - z2 @ coef

    r0 = residualize(dy_used)
    r1 = residualize(levels)

    s00 = r0.T @ r0 / m
    s01 = r0.T @ r1 / m
    s10 = s01.T
    s11 = r1.T @ r1 / m
    s11 = 0.5 * (s11 + s11.T)
    jitter = 1e-12 * max(float(np.trace(s11)) / n_vars, 1e-12)
    s11 = s11 + jitter * np.eye(n_vars)

    try:
        s00_inv = linalg.inv(s00)
    except linalg.LinAlgError:
        s00_inv = linalg.pinv(s00)

    middle = s10 @ s00_inv @ s01
    middle = 0.5 * (middle + middle.T)

    try:
        eigenvalues, eigenvectors = linalg.eigh(middle, s11)
    except linalg.LinAlgError:
        # Fall back to the non-symmetric problem if S11 is not numerically PD.
        eigenvalues, eigenvectors = linalg.eig(np.linalg.solve(s11, middle))
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)
        for j in range(n_vars):
            norm = float(eigenvectors[:, j] @ s11 @ eigenvectors[:, j])
            if norm > 0:
                eigenvectors[:, j] /= np.sqrt(norm)

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.clip(np.real(eigenvalues[order]), 0.0, 1.0 - 1e-12)
    eigenvectors = np.real(eigenvectors[:, order])

    # Deterministic sign: first non-zero loading positive.
    for j in range(n_vars):
        column = eigenvectors[:, j]
        nonzero = np.flatnonzero(np.abs(column) > 1e-14)
        if nonzero.size and column[nonzero[0]] < 0:
            eigenvectors[:, j] = -column

    log_one_minus = np.log1p(-eigenvalues)
    maxeig = -m * log_one_minus
    trace = np.array([-m * np.sum(log_one_minus[r:]) for r in range(n_vars)])

    trace_crit = np.array([TRACE_CRIT[det_order][n_vars - r - 1] for r in range(n_vars)])
    maxeig_crit = np.array(
        [MAXEIG_CRIT[det_order][n_vars - r - 1] for r in range(n_vars)]
    )

    return JohansenResult(
        trace_stat=trace,
        trace_crit=trace_crit,
        maxeig_stat=maxeig,
        maxeig_crit=maxeig_crit,
        eigenvectors=eigenvectors,
        eigenvalues=eigenvalues,
    )


# ---------------------------------------------------------------------------
# Half-life
# ---------------------------------------------------------------------------


def half_life(spread: np.ndarray) -> HalfLifeResult:
    """Estimate the mean-reversion half-life of a spread.

    Fit the discrete OU / AR(1) relation

        d spread_t = a + beta * spread_{t-1} + e_t

    so ``spread_t = a + (1 + beta) spread_{t-1} + e_t``. For
    ``-1 < beta < 0`` the half-life is

        HL = -ln(2) / ln(1 + beta)

    (equivalently ``ln(0.5) / ln(phi)`` with ``phi = 1 + beta``). The
    intercept is included so that a non-zero sample mean does not bias the
    persistence estimate; it does not enter the half-life formula.

    Parameters
    ----------
    spread : array-like
        Spread series (e.g. Engle-Granger residuals).

    Returns
    -------
    HalfLifeResult
        ``half_life`` is ``inf`` and ``converging`` is False when
        ``beta >= 0`` (no mean reversion) or ``beta <= -1`` (oscillatory
        outside the OU model).

    Raises
    ------
    ValueError
        If fewer than 10 finite observations or the spread is constant.
    """
    values = np.asarray(spread, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if values.size < 10:
        raise ValueError("Half-life estimation needs at least 10 observations")
    if np.ptp(values) == 0.0:
        raise ValueError("Half-life is undefined for a constant spread")

    d_spread = np.diff(values)
    lagged = values[:-1]
    if float(lagged @ lagged) <= 0.0:
        raise ValueError("Lagged spread has zero variance")

    design = np.column_stack([np.ones_like(lagged), lagged])
    coef, *_ = np.linalg.lstsq(design, d_spread, rcond=None)
    beta = float(coef[1])
    if beta >= 0.0 or beta <= -1.0:
        return HalfLifeResult(half_life=np.inf, beta=beta, converging=False)

    hl = -np.log(2.0) / np.log1p(beta)
    return HalfLifeResult(half_life=float(hl), beta=beta, converging=True)
