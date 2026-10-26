"""Labels. Forward returns over a fixed horizon, optional neutralization
against a market proxy, plus the triple-barrier variant."""
from __future__ import annotations

import numpy as np
import pandas as pd


def forward_return_label(prices: pd.DataFrame, horizon: int = 21, market: pd.DataFrame | None = None) -> pd.DataFrame:
    """Long-frame label: y(t) = total return from t+1 .. t+horizon, in excess
    of the market over the same window when ``market`` is supplied.

    Uses ``adj_close`` (corporate-action adjusted) so the label is a total-
    return construct; the last ``horizon`` rows per ticker are NaN and are
    dropped. The label is a function of *forward* information only — that is
    what makes the (train, label) pair leak-safe when the decision date is
    the feature date.
    """
    close = prices.pivot(index="date", columns="ticker", values="adj_close")
    fwd = close.shift(-horizon) / close - 1.0
    out = (
        fwd.stack()
        .rename("y")
        .dropna()
        .reset_index()
    )  # columns: date, ticker, y
    out["date"] = pd.to_datetime(out["date"])
    out = out.dropna(subset=["ticker"])

    if market is not None:
        # dedupe on date: callers may pass a market table with repeated date
        # rows (e.g. one row per (date, ticker) in a long panel); keep the
        # per-date observation so the forward-market series stays unique
        m = market.drop_duplicates("date").set_index("date")
        mkt_close = (1 + m["mkt_ret"]).cumprod()
        mkt_fwd = mkt_close.shift(-horizon) / mkt_close - 1.0
        out["mkt_fwd"] = out["date"].map(mkt_fwd).values
        out = out.dropna(subset=["mkt_fwd"])
        out["y"] = out["y"] - out["mkt_fwd"]
        out = out.drop(columns=["mkt_fwd"])
    return out


def neutralize_label(y: pd.Series, X: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Regress y on X (market + size loadings) and return the residual.
    Callers must pass train-only X — fitting on OOS data is a leak."""
    mask = y.notna() & X[cols].notna().all(axis=1)
    beta, *_ = np.linalg.lstsq(X.loc[mask, cols].values, y.loc[mask].values, rcond=None)
    resid = y - X[cols].values @ beta
    return pd.Series(resid, index=y.index)


def triple_barrier(prices: pd.DataFrame, profit: float, loss: float, timebar: int) -> pd.DataFrame:
    """López de Prado triple-barrier label: +1 / -1 / 0 by which barrier the
    forward path hits first over ``timebar`` days. Used by the event-driven
    sub-sample (P1 step 2) and by P5 replications."""
    close = prices.pivot(index="date", columns="ticker", values="adj_close")
    dates = list(close.index)
    rows = []
    for i, t0 in enumerate(dates):
        end = min(i + timebar, len(dates) - 1)
        if end <= i:
            continue
        p0 = close.iloc[i]
        window = close.iloc[i + 1 : end + 1]
        for tk in window.columns:
            if not np.isfinite(p0[tk]):
                continue
            # per-ticker barriers: upper/lower are level thresholds on this
            # name's own start price, not the whole cross-section
            hi = p0[tk] * (1 + profit)
            lo = p0[tk] * (1 - loss)
            w = window[tk].values
            up_hits = np.flatnonzero(w >= hi)
            dn_hits = np.flatnonzero(w <= lo)
            hit_up = int(up_hits[0]) if up_hits.size else np.inf
            hit_dn = int(dn_hits[0]) if dn_hits.size else np.inf
            y = 0.0 if (np.isinf(hit_up) and np.isinf(hit_dn)) else (1.0 if hit_up < hit_dn else -1.0)
            rows.append((t0, tk, y))
    return pd.DataFrame(rows, columns=["date", "ticker", "y_tb"]).dropna(subset=["y_tb"])


__all__ = ["forward_return_label", "neutralize_label", "triple_barrier"]
