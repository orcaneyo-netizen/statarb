"""statarb — statistical arbitrage & pairs-trading toolkit.

Public API
----------
* :func:`statarb.data.synthetic_pairs_universe`
* :func:`statarb.coint.adf_test`, :func:`~statarb.coint.engle_granger`,
  :func:`~statarb.coint.johansen`, :func:`~statarb.coint.half_life`
* :class:`statarb.kalman.KalmanHedgeRatio`
* :class:`statarb.signal.ZScoreSignal`
* :class:`statarb.backtest.PairsBacktester`
* :func:`statarb.screen.screen_pairs`
"""

from .backtest import BacktestResult, PairsBacktester
from .coint import (
    ADFResult,
    EGResult,
    HalfLifeResult,
    JohansenResult,
    adf_test,
    engle_granger,
    half_life,
    johansen,
)
from .data import synthetic_pairs_universe
from .kalman import KalmanHedgeRatio, KalmanResult
from .screen import screen_pairs
from .signal import Position, SignalResult, ZScoreSignal

__all__ = [
    "ADFResult",
    "BacktestResult",
    "EGResult",
    "HalfLifeResult",
    "JohansenResult",
    "KalmanHedgeRatio",
    "KalmanResult",
    "PairsBacktester",
    "Position",
    "SignalResult",
    "ZScoreSignal",
    "adf_test",
    "engle_granger",
    "half_life",
    "johansen",
    "screen_pairs",
    "synthetic_pairs_universe",
]

__version__ = "0.1.0"
