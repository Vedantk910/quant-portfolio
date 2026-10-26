"""Feature families.

Each ``family_*`` function returns a dict ``{feature_name: wide(date x ticker)}``
whose columns carry no prefix (the prefix is applied in ``build_feature_table``).
``build_feature_table`` melts them onto the long panel, producing one column
per feature: ``f_<family>_<name>``.

PIT rule enforced here: every value at (date, ticker) is a function of rows
with ``date' <= date`` only. Rolling windows use trailing data; the PEAD
family uses only announcements dated on or before ``t`` — the drift is already
in the price as of ``t``, and the label is *forward* from ``t`` (honest form).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_FAMILIES = ["technical", "fundamental", "pead"]


def _pivot(prices: pd.DataFrame, col: str) -> pd.DataFrame:
    return prices.pivot(index="date", columns="ticker", values=col)


def _log(x: pd.DataFrame) -> pd.DataFrame:
    return np.log(x.clip(lower=1e-9))


def family_technical(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    close = _pivot(prices, "adj_close")
    vol = _pivot(prices, "volume")
    dvol = _pivot(prices, "dollar_volume")
    mcap = _pivot(prices, "mkt_cap")

    ret_5 = close / close.shift(5) - 1.0
    ret_63 = close / close.shift(63) - 1.0
    return {
        "mom_5_63": ret_5 - ret_63,                               # 5-63d momentum
        "ret_1d": close.pct_change(fill_method=None),
        "ret_21d": close / close.shift(21) - 1.0,
        "ret_126d": close / close.shift(126) - 1.0,
        "rev_5d": -ret_5,                                          # short-term reversal
        "vol_20d": close.pct_change(fill_method=None).rolling(20).std(),  # realized vol
        "log_dvol_30d": _log(dvol.rolling(30).mean()),            # liquidity
        "turnover_20d": (vol / mcap.clip(lower=1e3)).rolling(20).mean(),
        "log_mkt_cap": _log(mcap),                                 # size
        "dist_52w_high": close / close.rolling(252).max() - 1.0,
    }


def family_fundamental(fundamentals: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Slowly-revealed ROA observations (the PIT stand-in for XBRL)."""
    wide = fundamentals.set_index("date").astype(float)
    roa_z = wide.sub(wide.mean(axis=1), axis=0).div(wide.std(axis=1).clip(lower=1e-9), axis=0)
    roa_growth = wide.diff(252) / wide.abs().rolling(252).mean().clip(lower=1e-9)
    return {"roa_z": roa_z, "roa_growth": roa_growth}


def family_pead(earnings: pd.DataFrame, prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Post-earnings-announcement features computed only from announcements
    dated on or before t (PIT-safe)."""
    close = _pivot(prices, "adj_close")
    e = earnings.pivot(index="date", columns="ticker", values="surprise")
    e = e.reindex(index=close.index, columns=close.columns).fillna(0.0)

    last_surprise = e.where(e != 0).ffill(limit=63)
    sd = e.rolling(252).std().clip(lower=1e-6)
    return {
        "pead_surprise_rec": last_surprise.fillna(0.0),
        "pead_cum_63d": e.rolling(63).sum(),
        "pead_surprise_z": last_surprise.fillna(0.0) / sd,
    }


def build_feature_table(store, families: list[str]) -> pd.DataFrame:
    """Assemble the long feature table: (date, ticker, f_*) over eligible rows."""
    prices = store.get("prices")
    all_wide: dict[str, pd.DataFrame] = {}
    if "technical" in families:
        for k, v in family_technical(prices).items():
            all_wide[f"f_technical_{k}"] = v
    if "fundamental" in families:
        for k, v in family_fundamental(store.get("fundamentals")).items():
            all_wide[f"f_fundamental_{k}"] = v
    if "pead" in families:
        for k, v in family_pead(store.get("earnings"), prices).items():
            all_wide[f"f_pead_{k}"] = v

    out = prices[["date", "ticker"]].copy()
    for name, wide in all_wide.items():
        s = wide.stack().rename(name).reset_index()  # date, ticker, name
        s.columns = ["date", "ticker", name]
        out = out.merge(s, on=["date", "ticker"], how="left")
    return out


__all__ = ["build_feature_table", "FEATURE_FAMILIES",
           "family_technical", "family_fundamental", "family_pead"]
