"""Numerical tables for unit-root and cointegration inference.

All values are transcribed from the published response surfaces / tables so
that :mod:`statarb.coint` can stay pure NumPy/SciPy.

References
----------
MacKinnon, J. G. (1994). "Approximate Asymptotic Distribution Functions for
Unit-Root and Cointegration Tests." *Journal of Business & Economic
Statistics* 12(2), 167-176.

MacKinnon, J. G. (2010). "Critical Values for Cointegration Tests." Queen's
University Department of Economics, Working Paper 1227.

MacKinnon, J. G., Haug, A. A., & Michelis, L. (1999). "Numerical
Distribution Functions of Likelihood Ratio Tests for Cointegration."
*Journal of Applied Econometrics* 14(5), 563-577. (Critical values as used by
Osterwald-Lenum, 1992.)
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# MacKinnon (1994) p-value response surfaces, N = number of I(1) series.
# The small-p branch is used below the "star" cut-off, the large-p branch
# above it.  p = Phi(d0 + d1*tau + d2*tau^2 + ...).
# ---------------------------------------------------------------------------

# Upper/lower/star cut-offs for N = 1 (ADF) and N = 2 (Engle-Granger).
TAU_MAX = {"n": np.inf, "c": 2.74, "ct": 0.70, "ctt": 0.54}
TAU_MIN = {"n": -19.04, "c": -18.83, "ct": -16.18, "ctt": -17.17}
TAU_STAR = {"n": -1.04, "c": -1.61, "ct": -2.89, "ctt": -3.21}

SMALL_P = {
    "n": np.array([0.6344, 1.2378, 0.032496]),
    "c": np.array([2.1659, 1.4412, 0.038269]),
    "ct": np.array([3.2512, 1.6047, 0.049588]),
    "ctt": np.array([4.0003, 1.6580, 0.048288]),
}

LARGE_P = {
    "n": np.array([0.4927, 0.93557, -0.006999, 0.0033066]),
    "c": np.array([1.7170, 0.55243, 0.043463, 0.0016671]),
    "ct": np.array([2.7117, 0.45731, 0.022868, 0.0006362]),
    "ctt": np.array([3.4671, 0.43476, 0.019231, 0.0005381]),
}

# N = 2 (Engle-Granger with one regressor) uses the cointegration response
# surface, which has heavier left tails than the N = 1 ADF surface.
TAU_MAX_N2_C = 0.92
TAU_MIN_N2_C = -18.86
TAU_STAR_N2_C = -2.62
SMALL_P_N2_C = np.array([2.9200, 1.5012, 0.039796])
LARGE_P_N2_C = np.array([2.1945, 0.64695, -0.029198, -0.0042377])

# MacKinnon (2010) critical-value response surfaces for N = 1, levels 1/5/10%.
# Each coefficient vector is evaluated as polyval(coef[::-1], 1/nobs).
CRIT_2010_N1 = {
    "n": np.array(
        [
            [-2.56574, -2.2358, -3.6270, 0.0],
            [-1.94100, -0.2686, -3.3650, 31.223],
            [-1.61682, 0.2656, -2.7140, 25.364],
        ]
    ),
    "c": np.array(
        [
            [-3.43035, -6.5393, -16.786, -79.433],
            [-2.86154, -2.8903, -4.2340, -40.040],
            [-2.56677, -1.5384, -2.8090, 0.0],
        ]
    ),
    "ct": np.array(
        [
            [-3.95877, -9.0531, -28.428, -134.155],
            [-3.41049, -4.3904, -9.0360, -45.374],
            [-3.12705, -2.5856, -3.9250, -22.380],
        ]
    ),
    "ctt": np.array(
        [
            [-4.37113, -11.5882, -35.819, -334.047],
            [-3.83239, -5.9057, -12.490, -118.284],
            [-3.55326, -3.6596, -5.2930, -63.559],
        ]
    ),
}

# Engle-Granger (N = 2, constant in the cointegrating regression) critical
# values.  More negative than the ADF values because beta is estimated.
CRIT_2010_N2_C = np.array(
    [
        [-3.89644, -10.9519, -33.527, 0.0],
        [-3.33613, -6.1101, -6.8230, 0.0],
        [-3.04445, -4.2412, -2.7200, 0.0],
    ]
)

# ---------------------------------------------------------------------------
# Johansen critical values (90%, 95%, 99%), Osterwald-Lenum (1992) as
# tabulated by MacKinnon-Haug-Michelis (1999).  Row ``i`` is the first row
# (null rank r = 0) for a system of dimension ``i + 1``.
#
# ``det_order`` follows the statsmodels convention:
#   -1 : no deterministic term
#    0 : restricted constant (default)
#    1 : constant plus linear trend
# ---------------------------------------------------------------------------

TRACE_CRIT = {
    -1: np.array(
        [
            [2.9762, 4.1296, 6.9406],
            [10.4741, 12.3212, 16.3640],
            [21.7781, 24.2761, 29.5147],
            [37.0339, 40.1749, 46.5716],
            [56.2839, 60.0627, 67.6367],
            [79.5329, 83.9383, 92.7136],
            [106.7351, 111.7797, 121.7375],
            [137.9954, 143.6691, 154.7977],
            [173.2292, 179.5199, 191.8122],
            [212.4721, 219.4051, 232.8291],
            [255.6732, 263.2603, 277.9962],
            [302.9054, 311.1288, 326.9716],
        ]
    ),
    0: np.array(
        [
            [2.7055, 3.8415, 6.6349],
            [13.4294, 15.4943, 19.9349],
            [27.0669, 29.7961, 35.4628],
            [44.4929, 47.8545, 54.6815],
            [65.8202, 69.8189, 77.8202],
            [91.1090, 95.7542, 104.9637],
            [120.3673, 125.6185, 135.9825],
            [153.6341, 159.5290, 171.0905],
            [190.8714, 197.3772, 210.0366],
            [232.1030, 239.2468, 253.2526],
            [277.3740, 285.1402, 300.2821],
            [326.5354, 334.9795, 351.2150],
        ]
    ),
    1: np.array(
        [
            [2.7055, 3.8415, 6.6349],
            [16.1619, 18.3985, 23.1485],
            [32.0645, 35.0116, 41.0815],
            [51.6492, 55.2459, 62.5202],
            [75.1027, 79.3422, 87.7748],
            [102.4674, 107.3429, 116.9829],
            [133.7852, 139.2780, 150.0778],
            [169.0618, 175.1584, 187.1891],
            [208.3582, 215.1268, 228.2226],
            [251.6293, 259.0267, 273.3838],
            [298.8836, 306.8988, 322.4264],
            [350.1125, 358.7190, 375.3203],
        ]
    ),
}

MAXEIG_CRIT = {
    -1: np.array(
        [
            [2.9762, 4.1296, 6.9406],
            [9.4748, 11.2246, 15.0923],
            [15.7175, 17.7961, 22.2519],
            [21.8370, 24.1592, 29.0609],
            [27.9160, 30.4428, 35.7359],
            [33.9271, 36.6301, 42.2333],
            [39.9085, 42.7679, 48.6606],
            [45.8930, 48.8795, 55.0335],
            [51.8528, 54.9629, 61.3449],
            [57.7954, 61.0404, 67.6415],
            [63.7248, 67.0756, 73.8856],
            [69.6513, 73.0946, 80.0937],
        ]
    ),
    0: np.array(
        [
            [2.7055, 3.8415, 6.6349],
            [12.2971, 14.2639, 18.5200],
            [18.8928, 21.1314, 25.8650],
            [25.1236, 27.5858, 32.7172],
            [31.2379, 33.8777, 39.3693],
            [37.2786, 40.0763, 45.8662],
            [43.2947, 46.2299, 52.3069],
            [49.2855, 52.3622, 58.6634],
            [55.2412, 58.4332, 64.9960],
            [61.2041, 64.5040, 71.2525],
            [67.1307, 70.5392, 77.4877],
            [73.0563, 76.5734, 83.7105],
        ]
    ),
    1: np.array(
        [
            [2.7055, 3.8415, 6.6349],
            [15.0006, 17.1481, 21.7465],
            [21.8731, 24.2522, 29.2631],
            [28.2398, 30.8151, 36.1930],
            [34.4202, 37.1646, 42.8612],
            [40.5244, 43.4183, 49.4095],
            [46.5583, 49.5875, 55.8171],
            [52.5858, 55.7302, 62.1741],
            [58.5316, 61.8051, 68.5030],
            [64.5292, 67.9040, 74.7434],
            [70.4630, 73.9355, 81.0678],
            [76.4081, 79.9878, 87.2395],
        ]
    ),
}
