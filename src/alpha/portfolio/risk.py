"""Risk measures: volatility, expected shortfall (CVaR), drawdown, tracking
error, and turnover. All on the *net* return series the backtest emits."""
from __future__ import annotations

import numpy as np
import pandas as pd


def risk_metrics(returns: pd.Series, ann: float = 252.0) -> dict:
    r = returns.dropna()
    if r.empty:
        return {}
    vol = float(r.std(ddof=1) * np.sqrt(ann))
    sharpe = float(r.mean() / r.std(ddof=1) * np.sqrt(ann)) if r.std(ddof=1) > 0 else np.nan
    return {
        "vol": vol,
        "sharpe": sharpe,
        "cv_var_95": expected_shortfall(r, alpha=0.05, ann=ann),
        "max_drawdown": max_drawdown(r),
        "worst_day": float(r.min()),
        "ann": ann,
    }


def expected_shortfall(returns: pd.Series, alpha: float = 0.05, ann: float = 252.0) -> float:
    """CVaR at level alpha, annualized (negative = expected shortfall)."""
    r = returns.dropna()
    if r.empty:
        return np.nan
    cutoff = r.quantile(alpha)
    tail = r[r <= cutoff]
    return float(tail.mean() * ann) if len(tail) else np.nan


def max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown of the cumulative equity curve."""
    eq = (1 + returns.dropna()).cumprod()
    peak = eq.cummax()
    dd = eq / peak - 1.0
    return float(dd.min())


def turnover(weights: pd.DataFrame, ann: float = 252.0) -> dict:
    """Single-sided turnover per rebalance and annualized turnover.

    ``weights`` rows are consecutive rebalances; single-sided turnover is
    half the L1 distance between consecutive weight rows. The first row has
    no prior rebalance to diff against (the initial position build is
    charged separately by the backtest engine), so it is excluded rather
    than silently zeroed — a zero would dilute the mean turnover estimate.
    """
    w = weights.fillna(0.0)
    d = w.diff().abs().sum(axis=1) / 2.0
    d = d.iloc[1:]
    per_reb = float(d.mean()) if len(d) else np.nan
    return {"turnover_per_rebalance": per_reb, "turnover_ann": per_reb * ann / 252.0 if not np.isnan(per_reb) else np.nan}


def tracking_error(returns: pd.Series, benchmark: pd.Series, ann: float = 252.0) -> float:
    te = (returns - benchmark).dropna()
    return float(te.std(ddof=1) * np.sqrt(ann)) if len(te) > 2 else np.nan


__all__ = ["risk_metrics", "expected_shortfall", "max_drawdown", "turnover", "tracking_error"]
