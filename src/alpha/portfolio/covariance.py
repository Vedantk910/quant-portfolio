"""Shrinkage covariance.

Sample covariance is unusable at these asset counts: the eigenvalue spectrum
blows up (Lindeberg's law / Marchenko-Pastur edge). ``ledoit_wolf`` returns
the oracle-approximating shrinkage target (identity scaled) with the
standard Ledoit-Wolf delta and the shrinkage intensity ``delta``. P4 uses
this in the mean-variance sub-step; P1 only reports the spectrum as the
"why sample covariance is unusable" exhibit.
"""
from __future__ import annotations

import numpy as np


def ledoit_wolf(returns: np.ndarray, shrink_to_diagonal: bool = True) -> tuple[np.ndarray, float]:
    """Return (shrunken covariance, intensity delta in [0,1]).

    ``returns`` is (n_obs x n_assets); the returned covariance is
    (n_assets x n_assets). Higher delta -> more shrinkage toward the
    diagonal (safer, less fit); delta near 0 means the sample matrix was
    already trustworthy.
    """
    X = np.asarray(returns, dtype=float)
    T, N = X.shape
    if T <= N:
        # not enough observations to estimate the off-diagonals reliably
        return np.diag(np.var(X, axis=0, ddof=1)), 1.0

    mu = X.mean(axis=0)
    F = X - mu
    S = np.cov(F, rowvar=False, ddof=1)
    d2 = np.diag(S).copy()
    # diagonal target
    if shrink_to_diagonal:
        target = np.diag(d2)
    else:
        mu_all = F.mean(axis=0)
        target = np.outer(mu_all, mu_all)

    beta = (np.sum(F**2, axis=0).sum()) / T
    phi = np.trace(np.cov(F, rowvar=False, ddof=0))  # stable estimator
    delta = np.clip(beta / phi, 0.0, 1.0)
    return (1 - delta) * S + delta * target, float(delta)


def eigenvalue_spectrum(returns: np.ndarray) -> dict:
    """Sample covariance eigenvalue spectrum + Marchenko-Pastur upper edge.
    The exhibit for the memo: how many eigenvalues sit above the MP edge
    (i.e. are 'real' signal vs. noise)."""
    X = np.asarray(returns, dtype=float).T
    T, N = X.shape
    S = np.cov(X, rowvar=False, ddof=1)
    eig = np.sort(np.linalg.eigvalsh(S))[::-1]
    q = T / N if T > N else N / T
    edge = (1 + np.sqrt(q)) ** 2 * eig.mean()
    above = int((eig > edge).sum())
    return {"eig": eig, "mp_edge": float(edge), "n_above_edge": above, "condition": float(eig[0] / eig[-1]) if eig[-1] > 0 else np.inf}


__all__ = ["ledoit_wolf", "eigenvalue_spectrum"]
