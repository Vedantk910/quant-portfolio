"""Backtest engine: gross >= net (costs drag), the turnover series is the
per-rebalance |w| sum (zero on non-rebalance days), and the metric dict is
populated. The engine treats each rebalance as a full re-establish of the
book — a deliberately conservative P0 cost assumption that P3 replaces with
a delta-based calibrated model (see ARCHITECTURE.md §5)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from alpha.backtest.engine import run_backtest
from alpha.costs.model import CostModel


def _prices() -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=120)
    rng = np.random.default_rng(3)
    rows = []
    for tk, base in [("A", 100.0), ("B", 50.0)]:
        lvl = base
        for d in dates:
            lvl *= 1 + float(rng.normal(0, 0.01))
            rows.append({"date": d, "ticker": tk, "adj_close": lvl})
    return pd.DataFrame(rows)


def _flat_weights(prices: pd.DataFrame, rebalance: int = 21) -> pd.DataFrame:
    close = prices.pivot(index="date", columns="ticker", values="adj_close")
    dates = close.index[::rebalance]
    return pd.DataFrame(0.5, index=dates, columns=close.columns)  # 50/50, constant


def test_net_is_not_above_gross() -> None:
    p = _prices()
    res = run_backtest(_flat_weights(p), p, rebalance=21, cost=CostModel())
    # costs drag: net can never exceed gross on any day
    joined = pd.concat([res.gross, res.net], axis=1).dropna()
    assert (joined["net"] <= joined["gross"] + 1e-12).all(), "net must be <= gross (cost drag)"
    assert res.metrics["net_sharpe"] <= res.metrics["gross_sharpe"] + 1e-9


def test_turnover_series_matches_weight_abs_sum() -> None:
    p = _prices()
    w = _flat_weights(p)
    res = run_backtest(w, p, rebalance=21, cost=CostModel())
    turn = res.turnover
    # on a rebalance day the turnover is the full |w| sum (a 50/50 book -> 1.0);
    # on non-rebalance days it is 0
    reb_day = turn[turn.index.isin(w.index)]
    non_reb = turn[~turn.index.isin(w.index)]
    assert np.allclose(reb_day, 1.0), "rebalance turnover = |w| sum = 1.0 for a 50/50 book"
    assert (non_reb == 0.0).all(), "no turnover on non-rebalance days"
    assert res.metrics["avg_turnover_per_rebalance"] > 0.0


def test_metrics_present() -> None:
    p = _prices()
    res = run_backtest(_flat_weights(p), p, rebalance=21, cost=CostModel())
    for k in ("gross_sharpe", "net_sharpe", "gross_cagr", "net_cagr",
              "avg_turnover_per_rebalance", "n_rebalances", "ann_factor"):
        assert k in res.metrics, f"missing metric {k}"
    assert res.metrics["n_rebalances"] == len(_flat_weights(p)) - 0


def test_cost_shock_lowers_net_sharpe() -> None:
    p = _prices()
    w = _flat_weights(p)
    base = run_backtest(w, p, rebalance=21, cost=CostModel())
    shocked = run_backtest(w, p, rebalance=21, cost=CostModel().shock(3.0))
    assert shocked.metrics["net_sharpe"] <= base.metrics["net_sharpe"], \
        "3x slippage must not improve the net result"
