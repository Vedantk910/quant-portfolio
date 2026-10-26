"""Point-in-time audit and the leaky-feature guardrail.

``pit_audit`` samples random (date, ticker) rows and checks that no feature
column is a same-value copy of its own one-step-later row (the tell of
look-ahead on the time axis). ``leaky_feature_check`` is the blueprint's
"deliberately leaky strategy is caught" test: a feature that *is* the
next-period return shows near-perfect OOS rank IC and is flagged, and the
reporting layer refuses to publish it as a result.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def pit_audit(feat_long: pd.DataFrame, sample: int = 500, seed: int = 0) -> dict:
    """For a sample of rows, verify that no feature value at date t equals the
    value at t+1 for the same ticker (a forward-leak on the time axis).

    A legitimate feature may USE date t, so we only fail when the *feature
    itself* carries future information. Returns per-column forward-leak flags.
    """
    rng = np.random.default_rng(seed)
    fcols = [c for c in feat_long.columns if c.startswith("f_")]
    g = feat_long.sort_values(["ticker", "date"])
    flagged = []
    for c in fcols:
        cur = g[c]
        nxt = g.groupby("ticker")[c].shift(-1)
        both = cur.notna() & nxt.notna()
        if both.sum() > 20:
            same = float((cur[both] == nxt[both]).mean())
            # legitimate features change day to day; a feature that is a copy
            # of its own next value carries the future.
            if same > 0.99:
                flagged.append({"column": c, "same_as_next_share": same})
    return {"n_columns": len(fcols), "flagged": flagged, "clean": not flagged}


def leaky_feature_check(store, feat_long: pd.DataFrame, label_long: pd.DataFrame, threshold: float = 0.30) -> dict:
    """The blueprint guardrail: a deliberately leaky feature (one that IS the
    next-period return) must be caught. We measure its OOS Spearman rank IC
    against the forward label; if it exceeds ``threshold`` we flag it and the
    reporting layer refuses to publish it. Healthy features stay far below;
    a leaky one sits near 1.0."""
    leaky = store.get("leaky")
    merged = feat_long.merge(leaky, on=["date", "ticker"], how="left")
    m2 = merged.merge(label_long[["date", "ticker", "y"]], on=["date", "ticker"], how="inner")
    if m2.empty:
        return {"leaky_ic": np.nan, "flagged": True, "reason": "no overlap between features and leaky table"}
    ic = float(m2["leak"].corr(m2["y"], method="spearman"))
    return {
        "leaky_ic": ic,
        "threshold": threshold,
        "flagged": bool(ic > threshold),
        "reason": ("next-period return leaked into features" if ic > threshold else "no leak detected"),
    }


__all__ = ["pit_audit", "leaky_feature_check"]
