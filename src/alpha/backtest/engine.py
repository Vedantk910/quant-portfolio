"""Vectorized daily backtest engine: weights -> costs -> returns.

The engine consumes SIGNED weight matrices (long positive / short negative),
rebalances on a fixed cadence, and returns a ``BacktestResult`` with the
gross, net, and cost-drag equity curves side by side. Costs are charged on
turnover at rebalance days via the P0 cost model; the engine is cost-model
agnostic, which is what lets P3 swap in calibrated costs without touching
the pipeline.

``buy_universe`` is the P0 definition-of-done check: an equal-weight
long-only universe strategy whose net return must land within 20bp of the
equal-weight universe return net of the same cost assumptions.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..costs.model import CostModel


@dataclass
class BacktestResult:
    gross: pd.Series
    net: pd.Series
    cost_drag: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    metrics: dict = field(default_factory=dict)

    def equity(self, which: str = "net") -> pd.Series:
        return (1 + getattr(self, which).dropna()).cumprod()


def run_backtest(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    rebalance: int = 21,
    cost: CostModel | None = None,
    market: pd.DataFrame | None = None,
) -> BacktestResult:
    """Vectorized daily engine. ``weights`` rows are rebalance dates; between
    rebalances the last weights hold. Daily portfolio return on rebalance
    day d is ``sum(w_d * close_{d+1..d+rebalance}) / close_d - 1`` for the
    long and short legs, net of turnover cost on that day.

    For a first pass we compute per-rebalance-window returns and forward-fill
    into a daily series. This is the honest "net-of-cost per window" form;
    an event-driven refinement is out of scope for P0.
    """
    close = prices.pivot(index="date", columns="ticker", values="adj_close")
    rets = close.pct_change(fill_method=None)
    w = weights.fillna(0.0)
    reb_dates = list(w.index)

    rows = []
    for i, d in enumerate(reb_dates):
        # np.timedelta64(1, "D") avoids the pandas/numpy "generic unit"
        # DeprecationWarning that `+ 1`-style integer conversions trigger
        end = reb_dates[i + 1] if i + 1 < len(reb_dates) else d + np.timedelta64(1, "D")
        window = rets.loc[(rets.index > d) & (rets.index <= end)]
        if window.empty:
            continue
        wrow = w.loc[d].reindex(window.columns).fillna(0.0)
        # Window return per ticker = cumprod of (1 + daily) - 1, computed NaN-
        # safe: inactive tickers have NaN daily returns, but their weight is
        # 0, so NaNs are ignored in the log-sum (1 + r > 0 because prices > 0).
        one_plus = 1.0 + window.values
        m = np.isfinite(one_plus) & (one_plus > 0)
        lg = np.where(m, np.log(np.clip(one_plus, 1e-12, None)), np.nan)
        win_ret = np.exp(np.nansum(lg, axis=0)) - 1.0
        win_ret = np.where(np.isfinite(win_ret), win_ret, 0.0)
        gross = float(wrow.values @ win_ret)
        rows.append({"date": d, "gross": gross, "w_abs_turn": float(wrow.abs().sum())})
    daily = pd.DataFrame(rows).set_index("date")
    cm = cost or CostModel()
    # Charge the cost model's per-unit-turnover cost on the gross turnover.
    # Fixed component (commission + half-spread) always applies; impact at
    # a conservative participation of 25% of the cap.
    unit_cost = cm.unit_cost(0.25 * cm.max_participation)
    daily["cost_drag"] = daily["w_abs_turn"] * unit_cost
    daily["net"] = daily["gross"] - daily["cost_drag"]

    # build a daily index spanning the full sample. A rebalance-day window
    # return is realized ON that day; between rebalances the book drifts with
    # no trades, so the daily return is 0 (NOT forward-filled — that would
    # multiply every window return across its 21 days).
    full_index = close.index
    g = daily["gross"].reindex(full_index).fillna(0.0)
    c = daily["cost_drag"].reindex(full_index).fillna(0.0)
    # keep explicit series names: g/c carry their own names, so the
    # subtraction result would be nameless and downstream concat/lookup
    # (joined["net"]) would KeyError.
    n = (g - c).rename("net")
    result = BacktestResult(
        gross=g, net=n, cost_drag=c, weights=w,
        turnover=daily["w_abs_turn"].rename("turnover")
        .reindex(full_index).fillna(0.0),
    )
    # annualized metrics on the rebalance-window returns (not daily ffilled)
    ann = 252.0 / rebalance
    gr = daily["gross"]
    nr = daily["net"]
    result.metrics = {
        "ann_factor": ann,
        "gross_sharpe": float(gr.mean() / gr.std(ddof=1) * np.sqrt(ann)) if gr.std(ddof=1) > 0 else np.nan,
        "net_sharpe": float(nr.mean() / nr.std(ddof=1) * np.sqrt(ann)) if nr.std(ddof=1) > 0 else np.nan,
        "gross_cagr": float((1 + gr).prod() ** (ann / 252.0) - 1),
        "net_cagr": float((1 + nr).prod() ** (ann / 252.0) - 1),
        "max_turnover_per_rebalance": float(daily["w_abs_turn"].max()) if len(daily) else np.nan,
        "avg_turnover_per_rebalance": float(daily["w_abs_turn"].mean()) if len(daily) else np.nan,
        "n_rebalances": int(len(daily)),
    }
    # per-rebalance-window returns (the decision-period series; one entry per
    # rebalance, no zero-padding). The multiple-testing layer (deflated
    # Sharpe) must run on THESE, not on the daily series: padding with
    # zeros inflates N without adding information and flattens the
    # skew/kurtosis moments the DSR correction depends on. Assigned AFTER the
    # metrics dict literal above — assigning before would be wiped out.
    result.metrics["gross_window"] = daily["gross"]
    result.metrics["net_window"] = daily["net"]
    return result


def buy_universe(prices: pd.DataFrame, *, rebalance: int = 21, cost: CostModel | None = None) -> BacktestResult:
    """Equal-weight long-only universe return (the P0 definition-of-done
    benchmark). Weight matrix is 1/n on all names each rebalance day.

    The 20bp test: this strategy's *net* return must land within 20bp of the
    equal-weight universe *net* return under the same cost assumptions —
    i.e. the engine is not charging phantom costs or mis-summing weights.
    """
    close = prices.pivot(index="date", columns="ticker", values="adj_close")
    reb_dates = close.index[::rebalance]
    n = close.count(axis=1)
    w = pd.DataFrame(0.0, index=reb_dates, columns=close.columns)
    for d in reb_dates:
        names = close.columns[close.loc[d].notna()]
        if len(names):
            w.loc[d, names] = 1.0 / len(names)
    return run_backtest(w, prices, rebalance=rebalance, cost=cost)


__all__ = ["BacktestResult", "run_backtest", "buy_universe"]
