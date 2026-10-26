"""Cost model correctness: the P0 definition of done charges a realistic cost,
and the shock variant pessimizes the slippage-type terms."""
from __future__ import annotations

import numpy as np
import pandas as pd

from alpha.costs.model import CostModel, calibrated_cost, capacity_curve


def test_fixed_cost_includes_commission_and_spread() -> None:
    c = CostModel(commission_bps=0.5, half_spread_bps=2.0)
    # at zero participation the impact term is 0, so cost == fixed
    assert np.isclose(c.unit_cost(0.0), (0.5 + 2.0) / 10_000.0)


def test_participation_capped() -> None:
    c = CostModel(max_participation=0.25)
    assert np.isclose(c.unit_cost(0.5), c.unit_cost(0.25))  # above cap -> capped


def test_shock_pessimizes_slippage() -> None:
    c = CostModel(impact_k=0.1, half_spread_bps=2.0)
    shocked = c.shock(3.0)
    part = 0.1
    assert shocked.unit_cost(part) > c.unit_cost(part)
    # commission is unchanged by the shock
    assert np.isclose(shocked.commission_bps, c.commission_bps)


def test_calibrated_cost_scales_with_trade() -> None:
    c = CostModel()
    adv = pd.Series([1e7] * 5)
    small = calibrated_cost(adv, 1e6, c)
    big = calibrated_cost(adv, 5e6, c)
    assert big > small  # larger trade -> higher cost


def test_capacity_curve_breaks_even() -> None:
    c = CostModel()
    adv = pd.Series([1e7] * 5)
    df = capacity_curve(c, adv, gross_edge=0.01, n_points=30)
    assert (df["net_edge_per_dollar"] > 0).any()
    assert df["break_even"].any()  # at some AUM the edge is eaten
