"""Portfolio layer: rank->weights construction, dollar/sector neutrality,
caps, and the risk metrics the memo reports."""
from __future__ import annotations

import numpy as np
import pandas as pd

from alpha.portfolio.construction import construct_portfolio, rank_signal
from alpha.portfolio.covariance import eigenvalue_spectrum, ledoit_wolf
from alpha.portfolio.risk import expected_shortfall, max_drawdown, risk_metrics, tracking_error, turnover


def _preds() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2020-01-01", periods=252)
    cols = [f"T{i:02d}" for i in range(100)]
    return pd.DataFrame(rng.normal(0, 1, (252, 100)), index=dates, columns=cols)


def test_ranks_in_unit_interval() -> None:
    r = rank_signal(_preds())
    assert (r.values >= 0).all() and (r.values <= 1).all()


def test_weights_are_dollar_neutral() -> None:
    w = construct_portfolio(_preds(), long_q=0.1, short_q=0.1)
    row_sums = w.sum(axis=1)
    assert np.allclose(row_sums, 0.0, atol=1e-9), "long/short legs must net to zero"


def test_long_short_counts_match_quantiles() -> None:
    w = construct_portfolio(_preds(), long_q=0.1, short_q=0.1)
    long_n = (w > 0).sum(axis=1).mean()
    short_n = (w < 0).sum(axis=1).mean()
    assert abs(long_n - 10) <= 2 and abs(short_n - 10) <= 2


def test_sector_neutrality_removes_sector_exposure() -> None:
    preds = _preds()
    sectors = pd.Series([f"S{i % 5}" for i in range(len(preds.columns))], index=preds.columns)
    w_plain = construct_portfolio(preds, sector_neutral=False, sectors=sectors)
    w_neutral = construct_portfolio(preds, sector_neutral=True, sectors=sectors)
    # sector exposure = mean weight per sector; neutrality should shrink the
    # cross-sector variance of sector-mean weights
    sec_plain = w_plain.T.groupby(sectors.values).sum().var().mean()
    sec_neutral = w_neutral.T.groupby(sectors.values).sum().var().mean()
    assert sec_neutral < sec_plain


def test_max_weight_cap() -> None:
    w = construct_portfolio(_preds(), long_q=0.1, short_q=0.1, max_weight=0.01)
    assert (w.abs() <= 0.01 + 1e-9).to_numpy().all(), "a name blew the per-weight cap"


def test_ledoit_wolf_shrinkage_is_in_unit_interval() -> None:
    rng = np.random.default_rng(0)
    R = rng.normal(0, 0.01, (200, 50))
    S, delta = ledoit_wolf(R)
    assert 0.0 <= delta <= 1.0
    assert S.shape == (50, 50)
    assert np.allclose(S, S.T)


def test_sample_cov_eigen_spectrum_is_unstable() -> None:
    rng = np.random.default_rng(0)
    R = rng.normal(0, 0.01, (60, 50))  # T < N region -> eigenvalue blow-up
    spec = eigenvalue_spectrum(R)
    assert spec["condition"] > 100, "sample covariance should be ill-conditioned"


def test_risk_metrics_on_known_series() -> None:
    r = pd.Series([0.01, -0.01, 0.02, -0.005, 0.01] * 50)
    m = risk_metrics(r)
    assert m["max_drawdown"] <= 0
    assert np.isfinite(m["sharpe"])


def test_max_drawdown_sign_and_bound() -> None:
    eq_ret = pd.Series([0.1, -0.5, 0.1, 0.1, 0.1])
    dd = max_drawdown(eq_ret)
    assert dd < 0


def test_turnover_counts_delta() -> None:
    w = pd.DataFrame({"A": [0.5, 0.0], "B": [0.5, 1.0]})
    t = turnover(w)
    # moving all long legs A->B = 1.0 single-sided turnover
    assert np.isclose(t["turnover_per_rebalance"], 0.5)


def test_tracking_error_vs_benchmark() -> None:
    r = pd.Series(np.linspace(0, 0.01, 100))
    te = tracking_error(r, r * 0.5)
    assert te > 0
