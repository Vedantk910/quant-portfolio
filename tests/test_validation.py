"""Validation library: purged K-fold boundaries, embargo, walk-forward shape,
and multiple-testing metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from alpha.validation.metrics import bootstrap_ic_ci, deflated_sharpe, ic_stats
from alpha.validation.purged import PurgedKFold
from alpha.validation.walkforward import AnchoredWalkForward


def test_purged_kfold_removes_boundary_dates() -> None:
    dates = np.array(pd.bdate_range("2020-01-01", "2022-12-31").values, dtype="datetime64[D]")
    f = PurgedKFold(k=4, purge=21, embargo=5, dates=dates)
    splits = list(f.splits())
    assert len(splits) == 4
    for tr, te in splits:
        te_min, te_max = te.min(), te.max()
        # this is SYMMETRIC purged CV by design: the train set spans BEFORE
        # and AFTER the test block (time-ordered folds keep the test block in
        # the middle, so the ordering claim "train strictly before test" does
        # not apply — the purge halo on both sides is the point).
        before = tr[tr < te_min]
        after = tr[tr > te_max]
        # purge: no train date within `purge` of either test boundary
        if len(before):
            assert before.max() <= te_min - np.timedelta64(f.purge, "D"), \
                "a train date sits inside the pre-test purge zone"
        if len(after):
            assert after.min() >= te_max + np.timedelta64(f.purge, "D"), \
                "a train date sits inside the post-test purge zone"
        # embargo: the last few train dates immediately before the test start
        # are also gone
        if len(before):
            assert before.max() <= te_min - np.timedelta64(f.embargo + 1, "D")


def test_walkforward_blocks_are_contiguous_and_disjoint() -> None:
    wf = AnchoredWalkForward(pd.bdate_range("2018-01-01", "2022-12-31"),
                             n_test_blocks=6, test_block_days=63, min_train_days=252)
    blocks = wf.blocks
    assert len(blocks) > 0
    prev_te_max = None
    for i, (tr, te) in enumerate(blocks):
        # expanding train: each train end precedes its test start
        assert tr.max() < te.min()
        # contiguous tests: each test block starts after the previous one ends
        if i > 0:
            assert te.min() > prev_te_max
        prev_te_max = te.max()


def test_ic_stats_handles_empty_series() -> None:
    empty = pd.DataFrame(columns=["date", "ic", "n"])
    s = ic_stats(empty)
    assert s["ic_mean"] != s["ic_mean"]  # NaN
    assert s["n_dates"] == 0


def test_bootstrap_ci_contains_mean() -> None:
    rng = np.random.default_rng(0)
    ics = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=100),
                         "ic": rng.normal(0.03, 0.02, 100)})
    r = bootstrap_ic_ci(ics, n_boot=500, seed=0)
    lo, hi = r["ci"]
    assert lo < r["ic_mean"] < hi or np.isclose(lo, hi)


def test_deflated_sharpe_penalizes_many_trials() -> None:
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0005, 0.01, 500))
    few = deflated_sharpe(r, n_trials=2)
    many = deflated_sharpe(r, n_trials=500)
    assert few["dss"] >= many["dss"], "more trials must not make a result look better"


def test_deflated_sharpe_detects_strong_signal_and_no_signal() -> None:
    rng = np.random.default_rng(0)
    strong = pd.Series(rng.normal(0.004, 0.01, 500))
    weak = pd.Series(rng.normal(0.0, 0.01, 500))
    assert deflated_sharpe(strong, n_trials=24)["significant"] is True
    assert deflated_sharpe(weak, n_trials=24)["significant"] is False


def test_deflated_sharpe_runs_on_window_returns_not_daily_padding() -> None:
    # the engine must expose the rebalance-window series, and the deflator
    # must accept it — the zero-padded daily series is the wrong input
    from alpha.backtest.engine import run_backtest
    from alpha.data.synthetic import MarketSpec, generate_market
    import tempfile

    spec = MarketSpec(seed=7, n_tickers=20, start="2020-01-01", end="2021-12-31")
    t = generate_market(spec)
    with tempfile.TemporaryDirectory() as d:
        from alpha.data.synthetic import write_parquet
        from alpha.data.store import PITStore

        write_parquet(d, t, spec)
        s = PITStore.from_parquet(d)
        prices = s.get("prices")
        close = prices.pivot(index="date", columns="ticker", values="adj_close")
        import pandas as _pd

        w = _pd.DataFrame(0.0, index=close.index[::21], columns=close.columns)
        from alpha.costs.model import CostModel

        bt = run_backtest(w, prices, rebalance=21, cost=CostModel())
        nw = bt.metrics["net_window"]
        assert len(nw) == len(w), "net_window must be one entry per rebalance"
        # window-level DSR on a handful of periods returns a small N, not 4000+
        r = deflated_sharpe(nw, n_trials=4)
        assert r.get("n_obs", 10**6) < 200, "deflated Sharpe is running on the padded daily series, not the windows"
