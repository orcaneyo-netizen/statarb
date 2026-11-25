"""Kalman filter for a time-varying hedge ratio.

State-space model
-----------------
State (ordering fixed as ``[beta, alpha]``)::

    s_t = [beta_t, alpha_t]'
    s_t = s_{t-1} + w_t,        w_t ~ N(0, Q),  Q = delta * I
    y_t = [x_t, 1] s_t + v_t,   v_t ~ N(0, R)

``beta_t`` is the hedge ratio and ``alpha_t`` the intercept. Both follow
independent random walks, so the filter adapts to slow beta drift. The
one-step-ahead prediction error ``y_t - [x_t, 1] s_{t-1}`` is the innovation
(or Kalman spread), with variance ``H P H' + R``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import optimize


@dataclass
class KalmanResult:
    """Output of :meth:`KalmanHedgeRatio.filter`.

    Attributes
    ----------
    beta : np.ndarray
        Filtered hedge ratio, aligned with the input (NaN where input was NaN).
    alpha : np.ndarray
        Filtered intercept.
    spread : np.ndarray
        Innovation ``y_t - beta_{t|t-1} x_t - alpha_{t|t-1}``.
    spread_var : np.ndarray
        Innovation variance ``H P_{t|t-1} H' + R``.
    log_likelihood : float
        Gaussian log-likelihood of the observed pairs.
    """

    beta: np.ndarray
    alpha: np.ndarray
    spread: np.ndarray
    spread_var: np.ndarray
    log_likelihood: float


class KalmanHedgeRatio:
    """Kalman filter for a random-walk hedge ratio.

    Parameters
    ----------
    delta : float, default=1e-5
        Process-noise variance for each state (larger = more adaptive).
    obs_noise : float, default=1e-2
        Observation-noise variance ``R``. For daily log prices the spread
        variance is typically of order ``1e-4`` to ``1e-2``; too small a
        value makes the filter absorb the spread into the hedge ratio.
    initial_beta : float, default=1.0
        Initial hedge ratio.
    initial_alpha : float, default=0.0
        Initial intercept.
    initial_cov : float, default=1.0
        Initial state variance (diagonal of ``P_0``).

    Raises
    ------
    ValueError
        If a variance is non-positive or ``initial_cov`` is negative.
    """

    def __init__(
        self,
        delta: float = 1e-5,
        obs_noise: float = 1e-2,
        initial_beta: float = 1.0,
        initial_alpha: float = 0.0,
        initial_cov: float = 1.0,
    ) -> None:
        if not np.isfinite(delta) or delta <= 0.0:
            raise ValueError("delta must be a positive finite number")
        if not np.isfinite(obs_noise) or obs_noise <= 0.0:
            raise ValueError("obs_noise must be a positive finite number")
        if not np.isfinite(initial_beta) or not np.isfinite(initial_alpha):
            raise ValueError("initial state values must be finite")
        if not np.isfinite(initial_cov) or initial_cov < 0.0:
            raise ValueError("initial_cov must be a non-negative finite number")

        self.delta = float(delta)
        self.obs_noise = float(obs_noise)
        self.initial_beta = float(initial_beta)
        self.initial_alpha = float(initial_alpha)
        self.initial_cov = float(initial_cov)

    def filter(self, y: np.ndarray, x: np.ndarray) -> KalmanResult:
        """Run the filter on a pair of price series.

        Parameters
        ----------
        y : array-like, shape (n,)
            Dependent series (e.g. log price of leg A).
        x : array-like, shape (n,)
            Regressor series (log price of leg B).

        Returns
        -------
        KalmanResult

        Raises
        ------
        ValueError
            If lengths differ or fewer than two complete observations remain.

        Notes
        -----
        Pairs with a NaN in either series are skipped and re-filled with NaN
        in the output; the state is not reset.
        """
        y_arr = np.asarray(y, dtype=float).ravel()
        x_arr = np.asarray(x, dtype=float).ravel()
        if y_arr.size != x_arr.size:
            raise ValueError("y and x must have the same length")

        mask = np.isfinite(y_arr) & np.isfinite(x_arr)
        y_clean = y_arr[mask]
        x_clean = x_arr[mask]
        n = y_clean.size
        if n < 2:
            raise ValueError("Kalman filter needs at least 2 complete observations")

        state = np.array([self.initial_beta, self.initial_alpha], dtype=float)
        cov = np.eye(2) * self.initial_cov
        process = np.eye(2) * self.delta
        obs_var = self.obs_noise
        identity = np.eye(2)

        beta_hist = np.empty(n)
        alpha_hist = np.empty(n)
        spread_hist = np.empty(n)
        spread_var_hist = np.empty(n)
        log_lik = 0.0

        for t in range(n):
            h = np.array([x_clean[t], 1.0])

            # Predict (random walk).
            state_pred = state
            cov_pred = cov + process

            innovation = y_clean[t] - float(h @ state_pred)
            innovation_var = float(h @ cov_pred @ h) + obs_var
            if innovation_var <= 0.0 or not np.isfinite(innovation_var):
                raise ValueError("Non-positive innovation variance; check inputs")

            gain = (cov_pred @ h) / innovation_var
            state = state_pred + gain * innovation

            # Joseph form keeps the covariance symmetric positive semi-definite.
            ikh = identity - np.outer(gain, h)
            cov = ikh @ cov_pred @ ikh.T + obs_var * np.outer(gain, gain)

            beta_hist[t] = state[0]
            alpha_hist[t] = state[1]
            spread_hist[t] = innovation
            spread_var_hist[t] = innovation_var
            log_lik += -0.5 * (np.log(2.0 * np.pi * innovation_var) + innovation**2 / innovation_var)

        beta = np.full_like(y_arr, np.nan)
        alpha = np.full_like(y_arr, np.nan)
        spread = np.full_like(y_arr, np.nan)
        spread_var = np.full_like(y_arr, np.nan)
        beta[mask] = beta_hist
        alpha[mask] = alpha_hist
        spread[mask] = spread_hist
        spread_var[mask] = spread_var_hist

        return KalmanResult(
            beta=beta,
            alpha=alpha,
            spread=spread,
            spread_var=spread_var,
            log_likelihood=float(log_lik),
        )

    def _neg_log_lik(self, params: np.ndarray, y: np.ndarray, x: np.ndarray) -> float:
        """Negative log-likelihood used by :meth:`mle_fit`."""
        delta, obs_noise = float(params[0]), float(params[1])
        if delta <= 0.0 or obs_noise <= 0.0:
            return 1e10

        old_delta, old_obs = self.delta, self.obs_noise
        self.delta, self.obs_noise = delta, obs_noise
        try:
            return -self.filter(y, x).log_likelihood
        except ValueError:
            return 1e10
        finally:
            self.delta, self.obs_noise = old_delta, old_obs

    def mle_fit(
        self,
        y: np.ndarray,
        x: np.ndarray,
        initial_guess: Optional[Tuple[float, float]] = None,
        bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    ) -> Tuple[float, float]:
        """Maximum-likelihood estimate of ``(delta, obs_noise)``.

        Uses ``scipy.optimize.minimize`` (L-BFGS-B) on the Gaussian
        log-likelihood. The fitted values are stored on the instance.

        Parameters
        ----------
        y, x : array-like
            Paired series.
        initial_guess : (delta, obs_noise), optional
            Starting values; defaults to the current parameters.
        bounds : ((lo, hi), (lo, hi)), optional
            Defaults to ``((1e-10, 1e-1), (1e-10, 10.0))``.

        Returns
        -------
        tuple of float
            ``(delta, obs_noise)``.

        Raises
        ------
        RuntimeError
            If the optimizer reports failure.
        """
        if initial_guess is None:
            initial_guess = (self.delta, self.obs_noise)
        if bounds is None:
            bounds = ((1e-10, 1e-1), (1e-10, 10.0))

        result = optimize.minimize(
            self._neg_log_lik,
            np.asarray(initial_guess, dtype=float),
            args=(np.asarray(y, dtype=float), np.asarray(x, dtype=float)),
            bounds=bounds,
            method="L-BFGS-B",
        )
        if not result.success:
            raise RuntimeError(f"MLE optimization failed: {result.message}")

        self.delta, self.obs_noise = float(result.x[0]), float(result.x[1])
        return self.delta, self.obs_noise
