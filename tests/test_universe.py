"""P0 definition of done: a trivial 'buy the universe' strategy returns
something within 20bp of the equal-weight universe return, net of estimated
costs. The reference below is an independent implementation of the same
equal-weight net return; the test pins the engine to it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from alpha.backtest.engine import buy_universe
from alpha.costs.model import CostModel


def _ew_universe_ref(prices: pd.DataFrame, rebalance: int = 21, cost: CostModel | None = None) -> tuple[float, float]:
    """Independent equal-weight (gross, net) CAGR, engine semantics:
    per-rebalance-window return is the equal-weight mean of per-ticker
    cumprod window returns (inactive tickers -> 0), net = gross minus one
    full turnover of the long book (|w|sum == 1.0) at the cost model's
    per-unit cost.
    """
    cost = cost or CostModel()
    close = prices.pivot(index="date", columns="ticker", values="adj_close")
    rets = close.pct_change(fill_method=None)
    reb = close.index[::rebalance]
    unit = cost.unit_cost(0.25 * cost.max_participation)
    ann = 252.0 / rebalance

    gross_w: list[float] = []
    net_w: list[float] = []
    for i in range(len(reb) - 1):
        d, end = reb[i], reb[i + 1]
        win = rets.loc[(rets.index > d) & (rets.index <= end)]
        if win.empty:
            continue
        names = close.loc[d].notna()
        one_plus = 1.0 + win.values
        m = np.isfinite(one_plus) & (one_plus > 0)
        lg = np.where(m, np.log(np.clip(one_plus, 1e-12, None)), np.nan)
        win_ret = np.exp(np.nansum(lg, axis=0)) - 1.0
        win_ret = np.where(np.isfinite(win_ret), win_ret, 0.0)
        ew = float(win_ret[names.to_numpy()].mean())
        gross_w.append(ew)
        net_w.append(ew - unit)  # one full turnover of the equal-weight book

    g = float((1 + np.array(gross_w)).prod()) ** (ann / 252.0) - 1.0
    n = float((1 + np.array(net_w)).prod()) ** (ann / 252.0) - 1.0
    return g, n


def test_buy_universe_within_20bp_of_ew(store) -> None:
    prices = store.get("prices")
    res = buy_universe(prices, rebalance=21)  # default CostModel, same as the reference
    g, n = _ew_universe_ref(prices, 21, None)
    assert abs(res.metrics["gross_cagr"] - g) <= 2e-4, (
        f"engine EW gross {res.metrics['gross_cagr']:.6f} vs reference {g:.6f}"
    )
    assert abs(res.metrics["net_cagr"] - n) <= 2e-4, (
        f"engine EW net {res.metrics['net_cagr']:.6f} vs reference {n:.6f}"
    )


def test_buy_universe_net_below_gross(store) -> None:
    """Costs must drag: the equal-weight net CAGR is strictly below gross."""
    prices = store.get("prices")
    res = buy_universe(prices, rebalance=21)
    assert res.metrics["net_cagr"] < res.metrics["gross_cagr"]
    assert res.metrics["avg_turnover_per_rebalance"] > 0.0
