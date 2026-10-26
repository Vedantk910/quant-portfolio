"""Comparable model interface: ridge / elastic-net baseline, LightGBM, and a
deliberately simple single-feature benchmark so the note can say honestly how
much the complex model adds.

All models expose ``fit(X, y)`` and ``predict(X)`` on a wide (date x ticker)
feature matrix, and accept a per-fold ``scaler`` so standardization happens
on train data only (``apply`` is what goes to OOS).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.preprocessing import StandardScaler


class Scaler:
    """Stateful, train-fit / apply-only standardizer with NaN-safe median fill."""

    def fit(self, X: pd.DataFrame) -> "Scaler":
        self.med_ = X.median()
        self.mad_ = (X.sub(self.med_, axis=1).abs().median()).replace(0.0, np.nan).fillna(1.0)
        return self

    def apply(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.sub(self.med_, axis=1).div(self.mad_, axis=1)


class Model:
    name = "base"

    def fit(self, X, y, **kw) -> "Model":  # pragma: no cover - interface
        raise NotImplementedError

    def predict(self, X) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


class Linear(Model):
    """Ridge / ElasticNet baseline.

    Both the features *and the target* are standardized, and both sets of
    statistics come from the training block only — fitting on the raw
    target (std ~0.05 in daily forward returns) with alpha=1.0 would zero
    every coefficient and the model degenerates to a constant predictor.
    """

    def __init__(self, kind: str = "elastic_net", alpha: float = 1e-3, l1: float = 0.5, **_ignored):
        self.kind = kind
        self.alpha = alpha
        self.l1 = l1
        self.scaler = Scaler()
        self.y_mean_ = 0.0
        self.y_std_ = 1.0
        self.params_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series, alpha: float | None = None, l1: float | None = None) -> "Linear":
        Xs = self.scaler.fit(X)
        Xv = Xs.apply(X)
        mask = y.notna() & Xv.notna().all(axis=1)
        alpha = self.alpha if alpha is None else alpha
        l1 = self.l1 if l1 is None else l1
        # target standardization, train statistics only
        self.y_mean_ = float(y.loc[mask].mean())
        self.y_std_ = float(y.loc[mask].std(ddof=0)) or 1.0
        ys = (y.loc[mask] - self.y_mean_) / self.y_std_
        est = (
            ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=100_000, random_state=0, tol=1e-6)
            if self.kind == "elastic_net"
            else Ridge(alpha=alpha)
        )
        est.fit(Xv.loc[mask].fillna(0.0).values, ys.values)
        self.params_ = est
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Xv = self.scaler.apply(X).fillna(0.0)
        pred = self.params_.predict(Xv.values)
        return pred * self.y_std_ + self.y_mean_


class LightGBM(Model):
    def __init__(self, n_estimators: int = 400, learning_rate: float = 0.03, num_leaves: int = 15):
        self.cfg = dict(
            objective="regression",
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            min_child_samples=50,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.7,
            random_state=0,
            n_jobs=-1,
            verbose=-1,
        )
        self.scaler = Scaler()
        self.model = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LightGBM":
        import lightgbm as lgb

        Xs = self.scaler.fit(X)
        Xv = Xs.apply(X)
        mask = y.notna() & Xv.notna().any(axis=1)
        self.model = lgb.LGBMRegressor(**self.cfg)
        self.model.fit(Xv.loc[mask].values, y.loc[mask].values)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Xv = self.scaler.apply(X).fillna(0.0)
        return self.model.predict(Xv.values)


class NaiveMomentum(Model):
    """Single-feature ranking on the 5-63d momentum column. The 'simple third
    benchmark' the blueprint wants reported so we know what boosting adds."""

    def __init__(self, col: str = "f_technical_mom_5_63"):
        self.col = col
        self.scale = 1.0

    def fit(self, X, y) -> "NaiveMomentum":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        s = X[self.col] if self.col in X.columns else X.iloc[:, 0]
        return s.fillna(0.0).values


def build_model(kind: str, **kw) -> Model:
    kind = kind.lower()
    if kind in ("ridge", "elastic_net", "linear"):
        return Linear(kind if kind != "linear" else "ridge", **kw)
    if kind in ("lightgbm", "gbm", "boosting"):
        return LightGBM(**kw)
    if kind in ("naive", "momentum", "naive_momentum"):
        return NaiveMomentum(**kw)
    raise ValueError(f"unknown model kind: {kind}")


__all__ = ["Model", "Linear", "LightGBM", "NaiveMomentum", "Scaler", "build_model"]
