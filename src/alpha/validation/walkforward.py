"""Anchored walk-forward — the headline OOS split.

Expanding train window, consecutive test blocks, refit on each block. Every
stateful transform (the model's internal scaler) is fit on the train block
only; each OOS block sees the model for the first time. No shuffle, no
random split on the time axis — the blueprint's central rule.

Purge: because forward labels overlap into the test window, the train side
drops the final ``purge`` dates (the label horizon) before each test block.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class AnchoredWalkForward:
    def __init__(self, dates: pd.DatetimeIndex, n_test_blocks: int = 8, test_block_days: int = 63,
                 min_train_days: int = 252, purge_days: int = 0):
        dates = pd.DatetimeIndex(dates).sort_values().unique()
        self.dates = dates
        self.n_test_blocks = n_test_blocks
        self.purge_days = purge_days
        self.blocks: list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]] = []
        # position-based walk-forward: the first `min_train_days` TRADING days
        # are the initial train window; test blocks are consecutive chunks.
        # The purge gap is subtracted from the train side, so start after it.
        start_pos = min(min_train_days + self.purge_days, len(dates) - 1)
        # the headline OOS period is the LAST n_test_blocks windows of the
        # sample (anchored at the end), so the reported OOS span is the most
        # recent one and the trial count matches the experiment log
        if n_test_blocks > 0:
            start_pos = max(start_pos, len(dates) - n_test_blocks * test_block_days)
        cursor = start_pos
        emitted = 0
        while cursor < len(dates):
            te = dates[cursor : cursor + test_block_days]
            if len(te) < 10:
                break
            end_pos = cursor + len(te)
            # train = everything before the test block, minus the trailing
            # `purge_days` adjacent to it (purge); expanding across blocks
            tr = dates[: max(0, cursor - self.purge_days)]
            if len(tr) < min_train_days:
                break
            self.blocks.append((tr, te))
            emitted += 1
            if n_test_blocks > 0 and emitted >= n_test_blocks:
                break
            cursor = end_pos

    def fit_predict(self, feat_long: pd.DataFrame, labels: pd.DataFrame, factory, fcols: list[str]) -> dict:
        """Expanding-window fit on the long panel; per-date OOS rank IC.

        ``feat_long``: (date, ticker, <f_*>) decision-time feature rows.
        ``labels``: (date, ticker, y) forward labels.
        ``factory``: callable returning a fresh model with fit(X, y)/predict(X).
        """
        dates = pd.DatetimeIndex(feat_long["date"].unique())
        lab = labels.set_index(["date", "ticker"])["y"]

        pred_parts, ic_rows = [], []
        for tr_dates, te_dates in self.blocks:
            tr_set, te_set = set(tr_dates), set(te_dates)
            Xtr = feat_long[feat_long["date"].isin(tr_set)].set_index(["date", "ticker"])
            ytr = labels[labels["date"].isin(tr_set)].set_index(["date", "ticker"])["y"].reindex(
                Xtr.index
            )
            ok = ytr.notna()
            model = factory()
            model.fit(Xtr.loc[ok, fcols].fillna(0.0), ytr[ok])

            Xte = feat_long[feat_long["date"].isin(te_set)]
            pred = model.predict(Xte[fcols].fillna(0.0))
            pred_df = Xte[["date", "ticker"]].copy()
            pred_df["pred"] = pred
            pred_parts.append(pred_df)

            # per-date rank IC vs the realized forward label
            m = pred_df.merge(labels[["date", "ticker", "y"]], on=["date", "ticker"], how="inner")
            for d, g in m.groupby("date"):
                g = g.dropna(subset=["y", "pred"])
                if len(g) >= 10 and g["pred"].nunique() > 1 and g["y"].nunique() > 1:
                    # a constant cross-section has no rank to correlate; skip
                    # it rather than emit a ConstantInputWarning and a NaN IC
                    ic_rows.append({
                        "date": d,
                        "ic": float(g["pred"].corr(g["y"], method="spearman")),
                        "n": int(len(g)),
                    })

        predictions = pd.concat(pred_parts, ignore_index=True)
        ic_series = pd.DataFrame(ic_rows).sort_values("date")
        return {"predictions": predictions, "ic_series": ic_series, "n_blocks": len(self.blocks)}


def run_anchored_walk_forward(
    feat_long: pd.DataFrame,
    labels: pd.DataFrame,
    factory,
    fcols: list[str],
    n_test_blocks: int = 8,
    test_block_days: int = 63,
    min_train_days: int = 252,
    purge_days: int = 21,
) -> dict:
    dates = pd.DatetimeIndex(feat_long["date"].unique())
    wf = AnchoredWalkForward(dates, n_test_blocks, test_block_days, min_train_days, purge_days)
    return wf.fit_predict(feat_long, labels, factory, fcols)


__all__ = ["AnchoredWalkForward", "run_anchored_walk_forward"]
