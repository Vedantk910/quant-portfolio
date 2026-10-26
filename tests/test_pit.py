"""Point-in-time discipline: no feature at (date, ticker) may equal a copy of
its own future value on the time axis."""
from __future__ import annotations

from alpha.data.universe import eligible_universe
from alpha.features.families import build_feature_table


def test_no_feature_copies_its_own_future(store) -> None:
    uni = eligible_universe(store.get("prices"), store.get("meta"), min_history_days=60, min_dollar_volume=0.0)
    fl = build_feature_table(store, ["technical", "pead"]).merge(
        uni[["date", "ticker"]], on=["date", "ticker"], how="inner"
    )
    fl = fl.sort_values(["ticker", "date"])
    fcols = [c for c in fl.columns if c.startswith("f_")]
    for c in fcols:
        cur = fl[c]
        nxt = fl.groupby("ticker")[c].shift(-1)
        both = cur.notna() & nxt.notna()
        if both.sum() < 20:
            continue
        same = float((cur[both] == nxt[both]).mean())
        # a legitimate feature changes day to day; it cannot be a verbatim
        # copy of tomorrow's value for the same ticker.
        assert same < 0.99, f"feature {c} is a same-value copy of its next row (share {same:.3f})"


def test_universe_is_point_in_time(store) -> None:
    """No eligible row appears before its listing date."""
    uni = eligible_universe(store.get("prices"), store.get("meta"), min_history_days=60, min_dollar_volume=0.0)
    listed = store.get("meta").set_index("ticker")["listed_at"]
    u = uni.set_index("ticker")["date"]
    bad = (u < u.index.map(listed)).any()
    assert not bad, "universe contains a row earlier than the ticker's listing date"
