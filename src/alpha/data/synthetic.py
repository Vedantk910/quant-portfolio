"""Deterministic synthetic equity market with a real PIT structure.

The default dataset for the whole pipeline: seeded, offline, ~19 years of
business days, with

- staggered listings and delistings      -> survivorship structure is real
- splits and quarterly dividends        -> adjusted-price discipline is testable
- latent channels                       -> market factor, momentum carry,
                                            post-earnings drift, quality
- a separate ``leaky`` table           -> one feature equal to the NEXT-period
                                          return, used ONLY by the validator
                                          guardrail test to prove look-ahead
                                          is caught.

Everything is a pure function of ``MarketSpec(seed, ...)``.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SECTORS = ["Technology", "Industrials", "Financials", "Consumer", "Materials"]


@dataclass
class MarketSpec:
    seed: int = 20270101
    n_tickers: int = 120
    start: str = "2008-01-01"
    end: str = "2025-12-31"
    earnings_cycle: int = 63        # trading days between earnings (~quarterly)
    pead_horizon: int = 40          # drift window after an announcement
    pead_strength: float = 0.0012   # per-day drift scaled by surprise
    momentum_coef: float = 0.004    # carry on the lagged 5-63d return
    quality_coef: float = 0.00025   # per-day drift per unit quality
    delist_frac: float = 0.45       # weakest-quality tickers exit mid-sample
    late_list_frac: float = 0.25    # tickers that list 1-3y into the sample
    leak_horizon: int = 21          # the "leaky feature" = fwd return over this
                                    # many days; matched to the label horizon


def _calendar(start: str, end: str) -> pd.DatetimeIndex:
    return pd.bdate_range(pd.Timestamp(start), pd.Timestamp(end))


def _garch_market_returns(cal: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    """GARCH(1,1)-flavored market return path with rare crash days."""
    n = len(cal)
    eps = rng.standard_normal(n)
    omega, alpha, beta, base = 0.06, 0.05, 0.94, 0.011
    s2 = np.empty(n)
    s2[0] = base**2
    for t in range(1, n):
        s2[t] = omega * base**2 + alpha * (eps[t - 1] * np.sqrt(s2[t - 1])) ** 2 + beta * s2[t - 1]
    ret = 0.0004 + np.sqrt(s2) * eps
    crash = rng.random(n) < 0.003
    ret[crash] -= 0.015 * rng.uniform(0.5, 2.0, int(crash.sum()))
    return ret


def _pead_drift(surprise: np.ndarray, spec: MarketSpec) -> np.ndarray:
    """Vectorized post-earnings drift: sum of decaying surprise contributions."""
    coeff = spec.pead_strength * np.exp(-np.arange(1, spec.pead_horizon + 1) / 15.0)
    drift = np.zeros_like(surprise)
    for k, c in enumerate(coeff, start=1):
        drift[k:] += c * surprise[: -k]
    return drift


def generate_market(spec: MarketSpec | None = None) -> dict[str, pd.DataFrame]:
    spec = spec or MarketSpec()
    rng = np.random.default_rng(spec.seed)
    cal = _calendar(spec.start, spec.end)
    T, N = len(cal), spec.n_tickers
    names = [f"T{i:03d}" for i in range(N)]
    mkt = _garch_market_returns(cal, rng)

    # ---- latent traits -----------------------------------------------------
    beta = np.clip(rng.normal(1.0, 0.25, N), 0.6, 1.4)
    quality = rng.standard_normal(N)
    base_vol = 0.015 * np.exp(rng.normal(-0.3, 0.4, N))
    base_dvol = np.exp(rng.normal(np.log(5e6), 0.9, N))
    shares = np.exp(rng.normal(np.log(60e6), 0.9, N))
    sector = [SECTORS[i % len(SECTORS)] for i in range(N)]
    log_start_price = rng.normal(np.log(60.0), 0.8, N)

    # listing / delisting structure (scale-aware: clamped so short samples
    # — e.g. the 4-year test fixture — never get an empty integer range)
    listed_at = np.zeros(N, dtype=int)
    n_late = int(spec.late_list_frac * N)
    late_idx = rng.choice(N, n_late, replace=False)
    if n_late:
        late_hi = int(min(3 * 365, max(151, T - 253)))
        listed_at[late_idx] = rng.integers(150, late_hi, n_late) if late_hi > 150 else 150

    delisted_at = np.full(N, T - 1, dtype=int)
    weak = np.argsort(quality)[: int(spec.delist_frac * N)]
    if len(weak):
        delist_hi = T - 120
        delist_lo = int(min(3 * 365, delist_hi - 1))
        if delist_lo < delist_hi:
            delisted_at[weak] = rng.integers(delist_lo, delist_hi, int(weak.size))
    delisted_at = np.clip(np.maximum(delisted_at, listed_at + 252), 0, T - 1)

    # earnings: per-ticker offset, one standardized surprise per cycle
    surprise = np.zeros((T, N))
    ann_offset = (np.arange(N) * 17) % spec.earnings_cycle
    for i in range(N):
        days = np.arange(listed_at[i], delisted_at[i] + 1)
        ann_days = days[(days - ann_offset[i]) % spec.earnings_cycle == 0]
        surprise[ann_days, i] = rng.standard_normal(ann_days.size)
    pead = _pead_drift(surprise, spec)
    # no drift before the first day of the sample
    pead[0] = 0.0

    # ---- daily returns -------------------------------------------------------
    active = np.zeros((T, N), dtype=bool)
    for i in range(N):
        active[listed_at[i] : delisted_at[i] + 1, i] = True

    # ---- daily returns -------------------------------------------------------
    # logp[t, i] is finite on day t iff ticker i is active on t and has a
    # finite price the day before. First-active days are seeded with the
    # start price *inside* the loop, so no anchor can be clobbered.
    logp = np.full((T, N), np.nan)
    ret = np.zeros((T, N))
    for t in range(T):
        a = active[t]
        if not a.any():
            continue
        mom = np.zeros(N)
        if t >= 68:
            m = a & active[t - 5] & active[t - 68] & np.isfinite(logp[t - 5]) & np.isfinite(logp[t - 68])
            mom[m] = spec.momentum_coef * (np.exp(logp[t - 5, m] - logp[t - 68, m]) - 1.0)
        r = (
            beta * mkt[t]
            + mom
            + spec.quality_coef * quality
            + pead[t]
            + base_vol * rng.standard_normal(N)
        )
        ret[t] = np.where(a, r, 0.0)
        first_today = a & (listed_at == t)
        cont = a & (listed_at < t) & np.isfinite(logp[t - 1])
        logp[t, first_today] = log_start_price[first_today]
        logp[t, cont] = logp[t - 1, cont] + ret[t, cont]
        logp[t, ~a] = np.nan

    close_base = np.exp(logp)  # continuous "no-split" reference path

    # splits: halving whenever the level tops ~400, at most one per 4y
    splits_rows = []
    fac = np.ones((T, N))
    for i in range(N):
        last = -4 * 365
        for d in range(listed_at[i], delisted_at[i] + 1):
            if close_base[d, i] > 400.0 and (d - last) >= 4 * 365:
                fac[d, i] = 0.5
                last = d
                splits_rows.append((cal[d], names[i], 0.5))
    cumlogfac = np.cumsum(np.log(fac), axis=0)  # factors s <= t
    close = close_base * np.exp(cumlogfac)      # raw prices carry the split drop
    adj = close_base                             # split-adjusted, continuous
    open_ = np.roll(close, 1, axis=0)  # raw open ~ previous raw close
    open_[0] = close[0]
    splits_df = (
        pd.DataFrame(splits_rows, columns=["date", "ticker", "factor"])
        if splits_rows
        else pd.DataFrame(columns=["date", "ticker", "factor"])
    )

    # volume: baseline x lognormal noise x volatility multiple
    vol_mult = np.where(np.abs(mkt) > 0.02, 2.5, 1.0)
    volume = np.zeros((T, N))
    for i in range(N):
        n_act = int(active[:, i].sum())
        volume[active[:, i], i] = (
            base_dvol[i] * rng.lognormal(0.0, 0.5, n_act) * vol_mult[active[:, i]]
        )
    dollar_volume = volume * close
    mcap = close * shares[None, :]

    # dividends: quarterly, small yield on announcement days
    div_rows = []
    for i in range(N):
        d_yield = 0.01 * np.exp(rng.normal(-0.4, 0.5))
        days = np.arange(listed_at[i], delisted_at[i] + 1)
        ann_days = days[(days - ann_offset[i]) % spec.earnings_cycle == 0]
        div_rows += [(cal[d], names[i], float(close[d, i]) * d_yield) for d in ann_days]
    dividends_df = (
        pd.DataFrame(div_rows, columns=["date", "ticker", "dps"])
        if div_rows
        else pd.DataFrame(columns=["date", "ticker", "dps"])
    )

    # fundamentals: slowly-revealed ROA observation per ticker (PIT stand-in)
    roa = np.zeros((T, N))
    for i in range(N):
        obs = 0.0
        arr = np.zeros(int(active[:, i].sum()))
        for j in range(len(arr)):
            obs = 0.95 * obs + 0.05 * 0.02 * quality[i] + rng.normal(0, 0.004)
            arr[j] = obs
        days = np.nonzero(active[:, i])[0]
        roa[days, i] = arr
    roa_df = pd.DataFrame(roa, index=cal).rename_axis("date").reset_index()
    roa_df.columns = ["date"] + [f"roa_{n}" for n in names]

    # the leaky table: forward return over `leak_horizon` days at each date —
    # i.e. the LABEL, joined back as a feature. This is the classic rookie
    # leak ("label as feature"); the validator guardrail must catch it. It is
    # built to be *consistent with the market-neutral label* the pipeline
    # actually uses: when forward_return_label is handed a market table it
    # subtracts the market's forward return over the same window, so the
    # leaky construct is that same market-neutral quantity — Spearman IC
    # against the real label ~= 1.0, far above any honest ceiling.
    h = spec.leak_horizon
    fwd = np.full((T, N), np.nan)
    if h < T:
        raw_fwd = close_base[h:T, :] / close_base[: T - h, :] - 1.0
        mkt_close = np.cumprod(1.0 + mkt)
        mkt_fwd = np.full(T, np.nan)
        mkt_fwd[: T - h] = mkt_close[h:T] / mkt_close[: T - h] - 1.0
        fwd[: T - h, :] = raw_fwd - mkt_fwd[: T - h, None]

    long_idx = np.repeat(np.arange(T), N)
    long_tick = np.tile(names, T)
    def _long(x: np.ndarray) -> np.ndarray:
        # x is (T, N); long form repeats each date's N tickers in order
        return x.T[np.tile(np.arange(N), T), long_idx]

    prices = pd.DataFrame(
        {
            "date": cal[long_idx],
            "ticker": long_tick,
            "open": _long(open_),
            "close": _long(close),
            "adj_close": _long(adj),
            "volume": _long(volume),
            "dollar_volume": _long(dollar_volume),
            "mkt_cap": _long(mcap),
            "listed": active.T[np.tile(np.arange(N), T), long_idx].astype(float),
        }
    ).query("listed == 1")

    earnings = pd.DataFrame(
        {"date": cal[long_idx], "ticker": long_tick, "surprise": _long(surprise)}
    ).query("surprise != 0")

    out = {
        "prices": prices,
        "market": pd.DataFrame({"date": cal, "mkt_ret": mkt}),
        "dividends": dividends_df,
        "splits": splits_df,
        "earnings": earnings,
        "fundamentals": roa_df,
        "leaky": pd.DataFrame(
            {"date": cal[long_idx], "ticker": long_tick, "leak": _long(fwd)}
        ).dropna(subset=["leak"]),
        "meta": pd.DataFrame(
            {
                "ticker": names,
                "sector": sector,
                "listed_at": cal[list(int(x) for x in listed_at)],
                "delisted_at": cal[list(int(x) for x in delisted_at)],
                "shares_out": shares,
            }
        ),
    }
    return out


def manifest_for(tables: dict[str, pd.DataFrame], spec: MarketSpec) -> dict:
    man = {}
    for name, df in tables.items():
        man[name] = {
            "sha256": hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()[:16],
            "rows": int(len(df)),
            "cols": list(df.columns),
        }
    man["spec"] = {k: getattr(spec, k) for k in vars(spec)}
    man["seed"] = spec.seed
    return man


def build(spec: MarketSpec | None = None) -> dict[str, pd.DataFrame]:
    return generate_market(spec)


def write_parquet(out_dir: str | Path, tables: dict[str, pd.DataFrame], spec: MarketSpec) -> dict:
    """Persist the dataset + manifest (integrity checks for later runs)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_parquet(out_dir / f"{name}.parquet", index=False)
    man = manifest_for(tables, spec)
    (out_dir / "manifest.json").write_text(json.dumps(man, indent=2, default=str))
    return man


__all__ = ["MarketSpec", "generate_market", "build", "write_parquet", "manifest_for", "SECTORS"]
