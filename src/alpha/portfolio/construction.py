"""Forecast -> implementable positions.

``rank_signal`` converts raw forecasts into cross-sectional ranks (the
blueprint's "turn forecasts into a ranking, not raw predicted returns").

``construct_portfolio`` produces the signed weight matrix the backtest
consumes: long the top quantile, short the bottom, dollar-neutral per date,
optionally sector-neutral (within-sector mean subtracted, then re-dollar-
neutralized), with a per-name cap and a turnover penalty that blends toward
the previous weights. Ledoit-Wolf (see covariance.py) is available for the
min-variance sub-step in P4; the first pass is rank-sleeve construction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rank_signal(preds: pd.DataFrame) -> pd.DataFrame:
    """Per-date cross-sectional rank in [0,1]; NaN -> 0.5 (neutral)."""
    return preds.rank(axis=1, method="average", pct=True).fillna(0.5)


def _sleeve_weights(r: pd.Series, hi: float, lo: float, max_weight: float) -> pd.Series:
    """Equal-weight sleeves on both legs, each name capped at ``max_weight``.

    No post-cap re-normalization: if ``n_names * max_weight < 1`` the leg is
    deliberately under-invested rather than silently blowing the cap. Both
    legs then get equalized to the smaller gross so the row stays dollar
    neutral without rescaling past the cap.
    """
    w = pd.Series(0.0, index=r.index)
    long_mask = r >= hi
    short_mask = r <= lo
    n_l, n_s = int(long_mask.sum()), int(short_mask.sum())
    leg_l, leg_s = 0.0, 0.0
    if n_l:
        base = np.minimum(np.full(n_l, 1.0 / n_l), max_weight)
        leg_l = float(base.sum())
        w.loc[r.index[long_mask]] = base
    if n_s:
        base = np.minimum(np.full(n_s, 1.0 / n_s), max_weight)
        leg_s = float(base.sum())
        w.loc[r.index[short_mask]] = -base
    # equalize legs (preserves the cap; shrinks the bigger leg only)
    if leg_l > 0 and leg_s > 0 and leg_l != leg_s:
        scale = min(leg_l, leg_s) / max(leg_l, leg_s)
        if leg_s > leg_l:
            w[w > 0] *= scale
        else:
            w[w < 0] *= scale
    return w


def construct_portfolio(
    preds: pd.DataFrame,
    *,
    long_q: float = 0.10,
    short_q: float = 0.10,
    max_weight: float = 0.05,
    sector_neutral: bool = False,
    sectors: pd.Series | None = None,
    turnover_penalty: float = 0.0,
    prev_weights: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Signed, dollar-neutral weight matrix (long positive, short negative).

    Each row sums to ~0: the long leg and short leg are each dollar-normalized
    to 100% of that side. ``sector_neutral`` dithers the within-sector mean
    out of the weights, then re-imposes dollar neutrality, so the portfolio
    is not a disguised sector bet.
    """
    ranks = rank_signal(preds)
    w = pd.DataFrame(0.0, index=preds.index, columns=preds.columns)
    for d in preds.index:
        r = ranks.loc[d].dropna()
        if r.empty:
            continue
        hi = r.quantile(1 - long_q)
        lo = r.quantile(short_q)
        sw = _sleeve_weights(r, hi, lo, max_weight)
        w.loc[d, sw.index[sw != 0]] = sw[sw != 0]

    # sector neutrality: subtract the within-sector mean weight per date, then
    # re-center to exact dollar neutrality. After demeaning, each sector's
    # net exposure is ~0 by construction (up to the re-centering constant).
    if sector_neutral and sectors is not None:
        sec = sectors.reindex(w.columns)
        sector_mean = w.T.groupby(sec).transform("mean").T
        w = w - sector_mean

    # turnover penalty: blend toward previous weights
    if turnover_penalty > 0 and prev_weights is not None:
        for d in w.index:
            if d in prev_weights.index:
                prev = prev_weights.loc[d].reindex(w.columns).fillna(0.0)
                w.loc[d] = (1 - turnover_penalty) * w.loc[d] + turnover_penalty * prev

    # final dollar neutrality
    w = w.sub(w.mean(axis=1), axis=0)
    # snap floating-point residuals (~1e-18) to exact zero so that a
    # "long" test like (w > 0) counts only real positions
    w = w.mask(np.abs(w.values) < 1e-12, 0.0)
    return w


__all__ = ["rank_signal", "construct_portfolio"]
