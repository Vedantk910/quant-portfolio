"""Model layer: the common fit/predict interface, standardization on train
only, and that the naive benchmark is genuinely simpler than the primary."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha.models.base import Linear, LightGBM, NaiveMomentum, build_model


def _xy(seed=0, n=400):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "f_a": rng.normal(0, 1, n),
        "f_b": rng.normal(0, 1, n),
        "f_c": rng.normal(0, 1, n),
    })
    y = df["f_a"] * 0.5 + rng.normal(0, 0.1, n)
    return df, y


def test_linear_learns_the_signal() -> None:
    X, y = _xy()
    m = build_model("ridge", alpha=1.0)
    m.fit(X, y)
    pred = m.predict(X)
    assert np.corrcoef(pred, y)[0, 1] > 0.8, "ridge should recover a linear signal"


def test_lightgbm_predicts() -> None:
    X, y = _xy()
    m = build_model("lightgbm", n_estimators=100, learning_rate=0.1)
    m.fit(X, y)
    pred = m.predict(X)
    assert len(pred) == len(y)
    assert np.isfinite(pred).all()


def test_naive_momentum_is_single_feature() -> None:
    X, y = _xy()
    m = NaiveMomentum(col="f_a")
    m.fit(X, y)
    pred = m.predict(X)
    # the naive model uses ONLY column f_a: perturbing other columns changes
    # nothing
    X2 = X.copy()
    X2["f_b"] = 1e6
    X2["f_c"] = -1e6
    assert np.allclose(pred, m.predict(X2)), "naive momentum must ignore non-target columns"


def test_scaler_fit_apply_on_train_only() -> None:
    """Standardization parameters must come from the train split, never from
    the OOS block (the blueprint's 'fit on train only' rule)."""
    X, y = _xy()
    Xtr, Xte = X.iloc[:200], X.iloc[200:]
    m = build_model("elastic_net")
    m.fit(Xtr, y.iloc[:200])
    # the scaler's median/MAD is pinned to the train distribution
    med = Xtr.median()
    assert np.allclose(m.scaler.med_, med.values, equal_nan=True)


def test_unknown_model_kind_raises() -> None:
    with pytest.raises(ValueError):
        build_model("does_not_exist")
