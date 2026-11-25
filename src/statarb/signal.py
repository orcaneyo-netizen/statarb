"""Z-score signal generation with hysteresis.

The signal is a small state machine on the causal rolling z-score of the
spread. The rules are intentionally explicit and non-repainting: the position
at ``t`` depends only on ``spread[:t+1]``.

============= ========================================= ==============
State         Condition                                 Action
============= ========================================= ==============
FLAT          ``z <= -entry``                           enter LONG_SPREAD
FLAT          ``z >= +entry``                           enter SHORT_SPREAD
LONG_SPREAD   ``z >= -exit``                            exit to FLAT
LONG_SPREAD   ``z <= -stop``                            stop out to FLAT
SHORT_SPREAD  ``z <= +exit``                            exit to FLAT
SHORT_SPREAD  ``z >= +stop``                            stop out to FLAT
============= ========================================= ==============

``LONG_SPREAD`` means long leg A and short ``beta`` units of leg B, i.e.
long the spread ``A - beta B``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class Position(Enum):
    """Position state."""

    FLAT = 0
    LONG_SPREAD = 1
    SHORT_SPREAD = -1


@dataclass
class SignalResult:
    """Output of :meth:`ZScoreSignal.generate`.

    Attributes
    ----------
    positions : np.ndarray
        Integer position per bar: 0 flat, +1 long spread, -1 short spread.
    z_scores : np.ndarray
        Rolling z-score; NaN during warm-up and where the window has gaps.
    entries : np.ndarray
        Boolean markers for entry bars.
    exits : np.ndarray
        Boolean markers for profit-target exit bars.
    stops : np.ndarray
        Boolean markers for stop-loss exit bars.
    """

    positions: np.ndarray
    z_scores: np.ndarray
    entries: np.ndarray
    exits: np.ndarray
    stops: np.ndarray


class ZScoreSignal:
    """Hysteresis z-score signal.

    Parameters
    ----------
    entry : float, default=2.0
        Absolute z-score required to open a position.
    exit : float, default=0.5
        Absolute z-score at which an open position is closed.
    stop : float, default=4.0
        Absolute z-score at which a position is stopped out.
    lookback : int, default=60
        Rolling window length. No signal is produced until ``lookback``
        observations are available, so the first possible signal is at index
        ``lookback - 1``.

    Raises
    ------
    ValueError
        Unless ``0 < exit < entry < stop`` and ``lookback >= 2``.

    Notes
    -----
    The rolling mean and standard deviation use the window
    ``spread[t-lookback+1 : t+1]`` (inclusive of the current bar), with the
    sample standard deviation (``ddof=1``). Using the current bar is causal —
    no future observation enters the calculation — and the position is
    applied to the *next* bar's return by the backtester.
    """

    def __init__(
        self,
        entry: float = 2.0,
        exit: float = 0.5,
        stop: float = 4.0,
        lookback: int = 60,
    ) -> None:
        if not (0.0 < exit < entry < stop):
            raise ValueError("Require 0 < exit < entry < stop")
        if not isinstance(lookback, (int, np.integer)) or lookback < 2:
            raise ValueError("lookback must be an integer >= 2")

        self.entry = float(entry)
        self.exit = float(exit)
        self.stop = float(stop)
        self.lookback = int(lookback)

    @property
    def warmup(self) -> int:
        """Number of leading bars that cannot produce a signal."""
        return self.lookback - 1

    def generate(self, spread: np.ndarray) -> SignalResult:
        """Generate positions and event markers from a spread series.

        Parameters
        ----------
        spread : array-like
            Spread series (Engle-Granger residuals, Kalman innovations, ...).

        Returns
        -------
        SignalResult

        Raises
        ------
        ValueError
            If ``spread`` is not one-dimensional or has fewer than two
            observations.

        Notes
        -----
        A zero-variance window yields ``z = 0`` and never triggers a trade.
        NaN observations propagate to ``z_scores`` and freeze the state
        machine (no entry/exit on that bar).
        """
        values = np.asarray(spread, dtype=float).ravel()
        if values.size < 2:
            raise ValueError("spread must contain at least two observations")

        n = values.size
        z_scores = np.full(n, np.nan)
        positions = np.zeros(n, dtype=int)
        entries = np.zeros(n, dtype=bool)
        exits = np.zeros(n, dtype=bool)
        stops = np.zeros(n, dtype=bool)

        current = Position.FLAT

        for t in range(n):
            if t < self.warmup:
                positions[t] = current.value
                continue

            window = values[t - self.warmup : t + 1]
            if not np.all(np.isfinite(window)):
                positions[t] = current.value
                continue

            mean = float(np.mean(window))
            sd = float(np.std(window, ddof=1))
            if sd > 0.0 and np.isfinite(values[t]):
                z = float((values[t] - mean) / sd)
                z_scores[t] = z
            else:
                z = 0.0
                z_scores[t] = 0.0

            if current is Position.FLAT:
                if z <= -self.entry:
                    current = Position.LONG_SPREAD
                    entries[t] = True
                elif z >= self.entry:
                    current = Position.SHORT_SPREAD
                    entries[t] = True
            elif current is Position.LONG_SPREAD:
                if z >= -self.exit:
                    current = Position.FLAT
                    exits[t] = True
                elif z <= -self.stop:
                    current = Position.FLAT
                    stops[t] = True
            else:  # SHORT_SPREAD
                if z <= self.exit:
                    current = Position.FLAT
                    exits[t] = True
                elif z >= self.stop:
                    current = Position.FLAT
                    stops[t] = True

            positions[t] = current.value

        return SignalResult(
            positions=positions,
            z_scores=z_scores,
            entries=entries,
            exits=exits,
            stops=stops,
        )
