"""Tests for the synthetic universe generator."""

from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from statarb.coint import engle_granger, half_life
from statarb.data import synthetic_pairs_universe


def test_shape_columns_and_positivity():
    prices, truth = synthetic_pairs_universe(n_pairs=3, n_extra=4, n_days=300)

    assert prices.shape == (300, 3 * 2 + 4)
    assert list(prices.columns) == [
        "A0", "B0", "A1", "B1", "A2", "B2", "X0", "X1", "X2", "X3",
    ]
    assert np.all(prices.to_numpy() > 0.0)
    assert len(truth["pairs"]) == 3
    assert set(truth["hedge_ratios"]) == set(truth["pairs"])
    assert set(truth["half_lives"]) == set(truth["pairs"])
    assert set(truth["spreads"]) == set(truth["pairs"])
    assert all(5.0 <= hl <= 20.0 for hl in truth["half_lives"].values())


def test_determinism():
    prices_a, truth_a = synthetic_pairs_universe(seed=7)
    prices_b, truth_b = synthetic_pairs_universe(seed=7)

    pd.testing.assert_frame_equal(prices_a, prices_b)
    assert truth_a["pairs"] == truth_b["pairs"]
    assert truth_a["hedge_ratios"] == truth_b["hedge_ratios"]
    pair = truth_a["pairs"][0]
    np.testing.assert_array_equal(truth_a["spreads"][pair], truth_b["spreads"][pair])


def test_different_seeds_differ():
    prices_a, _ = synthetic_pairs_universe(seed=1)
    prices_b, _ = synthetic_pairs_universe(seed=2)
    assert not np.allclose(prices_a.to_numpy(), prices_b.to_numpy())


def test_known_pairs_are_cointegrated():
    prices, truth = synthetic_pairs_universe()

    for pair in truth["pairs"]:
        asset_a, asset_b = pair
        result = engle_granger(
            np.log(prices[asset_b].to_numpy()), np.log(prices[asset_a].to_numpy())
        )
        assert result.adf_pvalue < 0.05
        assert result.adf_stat < -2.5

        estimated = half_life(result.residuals).half_life
        truth_hl = truth["half_lives"][pair]
        assert 0.5 * truth_hl < estimated < 1.5 * truth_hl, (
            pair, estimated, truth_hl
        )


def test_independent_pairs_are_not_cointegrated():
    prices, _ = synthetic_pairs_universe()
    noise = [column for column in prices.columns if column.startswith("X")]
    pvalues = [
        engle_granger(
            np.log(prices[b].to_numpy()), np.log(prices[a].to_numpy())
        ).adf_pvalue
        for a, b in combinations(noise, 2)
    ]

    assert np.median(pvalues) > 0.10
    # 5% nominal size: allow some false positives but not many.
    assert np.mean(np.asarray(pvalues) < 0.05) <= 0.15

    specific = engle_granger(
        np.log(prices["X1"].to_numpy()), np.log(prices["X0"].to_numpy())
    )
    assert specific.adf_pvalue > 0.10


def test_truth_spread_half_life_matches_generator():
    _, truth = synthetic_pairs_universe()
    for pair in truth["pairs"]:
        estimated = half_life(truth["spreads"][pair]).half_life
        assert abs(estimated / truth["half_lives"][pair] - 1.0) < 0.5


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_pairs": 0},
        {"n_extra": -1},
        {"n_days": 10},
    ],
)
def test_invalid_sizes_raise(kwargs):
    with pytest.raises(ValueError):
        synthetic_pairs_universe(**kwargs)
