"""Metrics that mean something: OOS rank IC, IC IR, bootstrap CI, deflated
Sharpe, and the per-date IC series the reporting layer turns into the
headline sentence."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats


def rank_ic_series(ic: pd.DataFrame) -> pd.Series:
    """Clean an IC series: drop NaN dates, return date -> IC."""
    s = ic.set_index("date")["ic"]
    return s.dropna().sort_index()


def ic_stats(ic: pd.DataFrame) -> dict:
    s = rank_ic_series(ic)
    if s.empty:
        return {"ic_mean": np.nan, "ic_ir": np.nan, "n_dates": 0}
    mean, std = float(s.mean()), float(s.std(ddof=1))
    ir = mean / std if std > 0 else np.nan
    return {"ic_mean": mean, "ic_ir": ir, "n_dates": int(len(s)), "ic_series": s}


def bootstrap_ic_ci(ic: pd.DataFrame, n_boot: int = 2000, ci: float = 0.95, seed: int = 0) -> dict:
    s = rank_ic_series(ic)
    rng = np.random.default_rng(seed)
    if s.empty:
        return {"ic_mean": np.nan, "ci": (np.nan, np.nan), "n_boot": 0}
    boots = np.empty(n_boot)
    v = s.values
    for i in range(n_boot):
        boots[i] = v[rng.integers(0, len(v), len(v))].mean()
    lo, hi = np.quantile(boots, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return {"ic_mean": float(v.mean()), "ci": (float(lo), float(hi)), "n_boot": n_boot, "se": float(boots.std())}


_EULER = 0.5772156649


def deflated_sharpe(returns: pd.Series, n_trials: int, alpha: float = 0.05, ann: float = 252.0) -> dict:
    """Bailey & López de Prado (2014) deflated Sharpe ratio (eqs. 22–23).

    ``returns`` are the strategy's per-PERIOD returns (one per decision
    period — e.g. rebalance windows, not zero-padded calendar days; padding
    with zeros would inflate N with no information and flatten the skew).

    Corrects the achieved per-period Sharpe ``SR`` for (a) the number of
    independent trials tried — the expected max of ``n_trials`` trial Sharpes
    is positive even under pure noise, and that floor, E[max], grows with
    ``n_trials`` and shrinks with sample size — and (b) the skew/kurtosis of
    the return distribution. ``n_trials`` comes from the experiment log.

    ``dss`` = P(DSR > 0 | n_trials): the probability the deflated Sharpe is
    positive. A raw Sharpe seen during model selection is not a result;
    neither is a Sharpe whose deflated value dies after the trial correction.
    """
    r = returns.dropna()
    N = len(r)
    if N < 10:
        return {"sharpe": np.nan, "dss": np.nan, "significant": False, "n_trials": int(n_trials)}

    sd = float(r.std(ddof=1))
    if not np.isfinite(sd) or sd <= 0.0:
        # a zero-variance series has no Sharpe and no DSR to report
        return {"sharpe": np.nan, "dss": np.nan, "significant": False,
                "n_trials": int(n_trials), "n_obs": int(N)}

    sr = float(r.mean()) / sd
    sk = float(stats.skew(r)) if N > 3 else 0.0
    ku_ex = float(stats.kurtosis(r)) if N > 3 else 0.0  # excess kurtosis; normal = 0

    M = max(int(n_trials), 1)
    # Var(SR_hat) under the null, per-period and skew/kurtosis-adjusted
    # (B&LP 2014, eq. 27):  (1/N)[ 1 - gamma1*SR + 0.5*(gamma4-3)*SR^2 ],
    # where gamma4-3 is the excess kurtosis. The 0.5 coefficient is exact.
    var_n = max(1.0 - sk * sr + 0.5 * ku_ex * sr**2, 1e-12)
    var_sr = var_n / N

    # E[max] of the per-period SR over M independent trials (eq. 22,
    # Gumbel approximation of the expected max of M normals, in SR units).
    # Monotone-increasing in M: the more backtests tried, the higher the
    # selection-bias floor that a "significant" Sharpe must clear.
    if M > 1:
        z1 = stats.norm.ppf(1.0 - alpha / M)
        z2 = stats.norm.ppf(1.0 - alpha * math.exp(-2.0) / M)
        e_max = ((1.0 - _EULER) * z1 + _EULER * z2) * math.sqrt(var_sr)
    else:
        e_max = 0.0

    # DSR statistic (eq. 23): achieved SR minus the trial-corrected floor,
    # in units of the per-trial SR standard error
    z_stat = (sr - e_max) / math.sqrt(max(var_sr, 1e-12))
    dss = float(stats.norm.cdf(z_stat))
    return {
        "sharpe": sr * math.sqrt(ann),
        "sharpe_per_period": sr,
        "dss": dss,
        "significant": dss > (1.0 - alpha),
        "n_trials": int(n_trials),
        "skew": sk,
        "kurt": ku_ex,
        "n_obs": int(N),
    }


__all__ = [
    "rank_ic_series",
    "ic_stats",
    "bootstrap_ic_ci",
    "deflated_sharpe",
]
