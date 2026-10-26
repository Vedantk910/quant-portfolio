"""Label correctness: forward returns use forward information only, and the
market-neutralization subtracts the same-window market return."""
from __future__ import annotations

import numpy as np
import pandas as pd

from alpha.labels.forward import forward_return_label, triple_barrier


def _tiny_prices() -> pd.DataFrame:
    # wide panel: BOTH tickers on every date (A and B share the calendar),
    # which is what forward_return_label's date x ticker pivot expects
    idx = pd.date_range("2020-01-01", periods=10)
    a = np.array([100.0, 101.0, 99.0, 102.0, 103.0, 105.0, 104.0, 106.0, 107.0, 108.0])
    b = np.array([50.0, 51.0, 52.0, 51.5, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0])
    wide = pd.DataFrame({"A": a, "B": b}, index=idx)
    long = wide.stack().rename("adj_close").reset_index()
    long.columns = ["date", "ticker", "adj_close"]
    return long


def test_forward_label_is_forward_only() -> None:
    p = _tiny_prices()
    lab = forward_return_label(p, horizon=2)
    # y(t) = P(t+2)/P(t) - 1
    row = lab[(lab["ticker"] == "A") & (lab["date"] == pd.Timestamp("2020-01-01"))]
    assert len(row) == 1
    assert np.isclose(row["y"].iloc[0], 99.0 / 100.0 - 1.0)


def test_market_neutralization() -> None:
    p = _tiny_prices()
    mkt = pd.DataFrame({"date": p["date"], "mkt_ret": np.full(len(p), 0.001)})
    raw = forward_return_label(p, horizon=2)
    neutral = forward_return_label(p, horizon=2, market=mkt)
    # neutral label = raw - market forward return over the same window
    mkt_fwd = (1 + 0.001) ** 2 - 1.0
    m = raw.merge(neutral, on=["date", "ticker"], how="inner", suffixes=("_r", "_n"))
    assert np.allclose(m["y_r"] - m["y_n"], mkt_fwd), "market-neutral label is off"


def test_triple_barrier_values() -> None:
    p = _tiny_prices()
    tb = triple_barrier(p, profit=0.02, loss=0.02, timebar=3)
    assert set(tb["y_tb"].dropna().unique()).issubset({-1.0, 0.0, 1.0})
