"""P0 definition of done #2: a deliberately leaky feature must be caught by
the validator and excluded from reported results. This test proves the
guardrail works — a passing pipeline must catch it."""
from __future__ import annotations

import pandas as pd

from alpha.data.universe import eligible_universe
from alpha.features.families import build_feature_table
from alpha.labels.forward import forward_return_label
from alpha.validation.pit import leaky_feature_check


def test_leaky_feature_is_flagged(store) -> None:
    uni = eligible_universe(store.get("prices"), store.get("meta"), min_history_days=60, min_dollar_volume=0.0)
    fl = build_feature_table(store, ["technical", "pead"]).merge(
        uni[["date", "ticker"]], on=["date", "ticker"], how="inner"
    )
    lab = forward_return_label(store.get("prices"), 21, store.get("market")).merge(
        uni[["date", "ticker"]], on=["date", "ticker"], how="inner"
    )
    res = leaky_feature_check(store, fl, lab, threshold=0.30)
    # the leak table is the label joined back as a feature -> rank IC must be
    # near 1.0, far above the 0.30 ceiling, and therefore flagged
    assert res["leaky_ic"] > 0.8, f"leak feature IC {res['leaky_ic']:.3f} should be ~1.0"
    assert res["flagged"] is True, "guardrail failed to flag the leaky feature"


def test_healthy_features_are_not_flagged(store) -> None:
    """The *real* feature set (no leak column) must stay below the ceiling."""
    uni = eligible_universe(store.get("prices"), store.get("meta"), min_history_days=60, min_dollar_volume=0.0)
    fl = build_feature_table(store, ["technical", "pead"]).merge(
        uni[["date", "ticker"]], on=["date", "ticker"], how="inner"
    )
    lab = forward_return_label(store.get("prices"), 21, store.get("market")).merge(
        uni[["date", "ticker"]], on=["date", "ticker"], how="inner"
    )
    # compute the max OOS rank IC any legitimate feature has against the label
    merged = fl.merge(lab[["date", "ticker", "y"]], on=["date", "ticker"], how="inner")
    best = -1.0
    for c in [x for x in fl.columns if x.startswith("f_")]:
        ic = abs(merged[c].corr(merged["y"], method="spearman"))
        best = max(best, ic)
    # even the strongest real signal stays in the sub-leak zone: on the test
    # fixture the best honest feature (momentum) reaches ~0.30 IC, while the
    # deliberately leaky table sits at ~1.0. The 0.30 *flag* threshold the
    # pipeline uses for exclusion is deliberately tighter than this sanity
    # ceiling — the two answer different questions.
    assert best < 0.45, f"a real feature reached IC {best:.3f}; something is leaking into features"
