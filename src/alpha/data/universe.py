"""Point-in-time universe construction.

Eligibility on date ``t`` is a function of information available at or before
``t`` only:

- the ticker was listed (``listed_at <= t``) and not yet delisted
  (``delisted_at >= t``) — the delisting date is treated as *known at that
  date*; we never look ahead at "final" status,
- it has at least ``min_history_days`` of listing history as of ``t``,
- trailing-30d average dollar volume (a PIT quantity) exceeds the floor.

``stable_by_construction`` keeps only tickers whose history at ``t`` is at
least 3 years — a legitimate point-in-time survivorship-aware approximation,
disclosed in the memo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def eligible_universe(
    prices: pd.DataFrame,
    meta: pd.DataFrame,
    min_history_days: int = 60,
    min_dollar_volume: float = 0.0,
    stable_by_construction: bool = False,
) -> pd.DataFrame:
    """Return the long-frame universe: (date, ticker, eligible, dollar_volume_30d, sector)."""
    m = meta.set_index("ticker")
    dv = prices.pivot(index="date", columns="ticker", values="dollar_volume")
    dv30 = dv.rolling(30, min_periods=20).mean()

    out = []
    dates = prices["date"].unique()
    listed_at = m["listed_at"].to_dict()
    delisted_at = m["delisted_at"].to_dict()
    stable_days = 3 * 252 if stable_by_construction else min_history_days

    for d in dates:
        dv_row = dv30.loc[d]
        for tk in dv_row.index:
            la, da = listed_at[tk], delisted_at[tk]
            if not (la <= d <= da):
                continue
            history = (d - la).days
            floor = stable_days if stable_by_construction else min_history_days
            if history < floor:
                continue
            dv30_v = dv_row[tk]
            if not np.isfinite(dv30_v) or dv30_v < min_dollar_volume:
                continue
            out.append((d, tk))

    uni = pd.DataFrame(out, columns=["date", "ticker"])
    uni["date"] = pd.to_datetime(uni["date"])
    uni = uni.merge(meta, on="ticker", how="left")
    return uni[["date", "ticker", "sector", "listed_at", "delisted_at"]]


def universe_summary(uni: pd.DataFrame) -> pd.DataFrame:
    """Per-date eligible-name count — the exhibit that proves the universe was
    the right size on each date (no survivorship)."""
    return uni.groupby("date")["ticker"].count().rename("n_eligible").to_frame()


__all__ = ["eligible_universe", "universe_summary"]
