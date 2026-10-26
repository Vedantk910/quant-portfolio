"""Purged K-fold with embargo (López de Prado, mlfinlab style).

Time is never shuffled. Folds are consecutive blocks of the sorted date
index. Around each test block, ``purge`` dates (the label horizon) on either
side are removed from the *training* set, and an ``embargo`` gap is placed
just before each test start — because overlapping labels make near-boundary
observations correlated with the test fold.

(The deliberately-leaky-feature guardrail lives in
``alpha.validation.pit.leaky_feature_check`` — this class only owns the
time-axis split mechanics.)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PurgedKFold:
    k: int = 5
    purge: int = 21          # = label horizon (trading days)
    embargo: int = 5
    dates: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        self.dates = np.asarray(self.dates, dtype="datetime64[D]")
        self.order = np.argsort(self.dates, kind="stable")
        self.sorted_dates = self.dates[self.order]

    def _folds(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Consecutive, time-ordered folds: (train_pos, test_pos) in positions
        on the sorted-date axis."""
        n = len(self.sorted_dates)
        edges = np.linspace(0, n, self.k + 1).astype(int)
        out = []
        for i in range(self.k):
            test_pos = np.arange(edges[i], edges[i + 1])
            test_dates = self.sorted_dates[test_pos]
            lo_d = test_dates[0] - np.timedelta64(self.purge, "D")
            hi_d = test_dates[-1] + np.timedelta64(self.purge, "D")
            # train = everything except this test fold and its purged halo
            mask_test = np.isin(self.sorted_dates, test_dates)
            mask_purge = (self.sorted_dates >= lo_d) & (self.sorted_dates <= hi_d) & ~mask_test
            train_pos = np.where(~(mask_test | mask_purge))[0]
            # embargo: drop the last `embargo` train dates just before each test start
            if len(train_pos) and self.embargo > 0:
                before = train_pos[train_pos < test_pos[0]]
                if len(before) >= self.embargo:
                    drop = before[-self.embargo :]
                    train_pos = np.setdiff1d(train_pos, drop, assume_unique=False)
            out.append((train_pos, test_pos))
        return out

    def splits(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Yields (train_dates, test_dates) as datetime64 arrays, time-ordered."""
        for tr, te in self._folds():
            yield self.sorted_dates[tr], self.sorted_dates[te]


__all__ = ["PurgedKFold"]
