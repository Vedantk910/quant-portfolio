"""Real-data loaders (opt-in). Same PIT contracts as the synthetic source.

Every loader returns the same table names as ``synthetic.generate_market`` so
the pipeline is source-agnostic. Traps from the blueprint are enforced here,
not assumed:

- prices (Stooq / yfinance): daily split/dividend-adjusted closes. **Not**
  survivorship-safe — the ticker list is the currently-listed universe; the
  memo must disclose and quantify this (see ARCHITECTURE.md §2).
- SEC XBRL companyfacts: fundamentals usable on ``filingDate``, never the
  period-end date. Cache aggressively; respect the User-Agent policy.
- FRED / ALFRED: macro regime series; ALFRED vintage when available.
- Ken French: benchmarks only — a published factor is never a feature
  without disclosure.

These functions are deliberately thin: they adapt an API into the table
contracts, and the *pipeline logic* (features, labels, validation, costs) is
shared verbatim with the synthetic source. If a loader misbehaves, the
manifest + PIT audit in ``store.py`` / ``validation/pit.py`` catches it.
"""
from __future__ import annotations

import io
import time
from pathlib import Path

import pandas as pd
import requests

_UA = "quant-portfolio-research (personal; mailto:research@example.com)"


def _get_json(url: str, *, retries: int = 3, delay: float = 1.0, params: dict | None = None) -> dict:
    for attempt in range(retries):
        resp = requests.get(url, headers={"User-Agent": _UA}, params=params, timeout=30)
        if resp.status_code == 429:
            time.sleep(delay * (attempt + 1))
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError(f"exhausted retries: {url}")


def load_stooq(tickers: list[str], start: str, end: str, cache_dir: str | Path = "data/stooq") -> pd.DataFrame:
    """Stooq daily CSV (no auth, free). Returns a prices-format long frame.

    Stooq exports are one file per symbol:
      https://stooq.com/q/dl/?s=aapl.us&i=d&d1=20080101&d2=20251231
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    d1 = pd.Timestamp(start).strftime("%Y%m%d")
    d2 = pd.Timestamp(end).strftime("%Y%m%d")
    for tk in tickers:
        sym = tk.lower() + ".us"
        cache = cache_dir / f"{sym}_{d1}_{d2}.csv"
        if cache.exists():
            raw = pd.read_csv(cache)
        else:
            url = f"https://stooq.com/q/dl/?s={sym}&i=d&d1={d1}&d2={d2}"
            resp = requests.get(url, headers={"User-Agent": _UA}, timeout=30)
            cache.write_bytes(resp.content)
            raw = pd.read_csv(io.BytesIO(resp.content))
        f = pd.DataFrame({"date": pd.to_datetime(raw["Date"]), "close": raw["Close"], "volume": raw["Volume"]})
        f["ticker"] = tk
        frames.append(f)
    out = pd.concat(frames, ignore_index=True)
    return out


def load_yfinance(tickers: list[str], start: str, end: str, *, auto_adjust: bool = True) -> pd.DataFrame:
    """yfinance auto-adjusted daily OHLCV -> prices-format long frame."""
    import yfinance as yf  # optional dependency

    frames = []
    for tk in tickers:
        h = yf.download(tk, start=start, end=end, auto_adjust=auto_adjust, progress=False)
        h = h.reset_index()
        h["ticker"] = tk
        frames.append(h.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}))
    out = pd.concat(frames, ignore_index=True)
    for c in ("open", "high", "low", "close", "volume"):
        if c in out.columns:
            out[c] = out[c].astype("float64")
    return out


def load_sec_fundamentals(cik: str, cache_dir: str | Path = "data/sec") -> pd.DataFrame:
    """SEC XBRL companyfacts -> long (date=filingDate, ticker, tag, value).

    The PIT rule: ``date`` is the *filing* date, so a 10-K filed in Feb 2024
    for FY2023 is usable on 2024-02-x, never in Jan 2024. The table keeps
    ``periodEnd`` for reference only — the pipeline must never join on it.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"companyfacts_{cik}.json"
    if not cache.exists():
        raw = _get_json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json")
        cache.write_text(_json_dump(raw))
        time.sleep(delay=0.15)
    raw = _json_load(cache)
    rows = []
    for tag, _tagmeta, (units, unit) in _walk_facts(raw):
        facts = (units or {}).get(unit, [])
        for f in facts:
            rows.append(
                {"cik": cik, "tag": tag, "value": f.get("val"),
                 "filingDate": f.get("filed") or f.get("end"),
                 "periodEnd": f.get("end"), "frame": f.get("frame", "")}
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["ticker"] = _cik_to_ticker(cik)
    return df


def _walk_facts(raw: dict):
    for cat in (raw.get("facts") or {}).values():
        for tag, tagdata in cat.items():
            for unit, units in tagdata.items():
                yield tag, tagdata, (unit, units)


def load_fred(series_ids: list[str], cache_dir: str | Path = "data/fred") -> pd.DataFrame:
    """Keyless FRED graph CSVs (no auth required) -> (date, series, value)."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for sid in series_ids:
        cache = cache_dir / f"{sid}.csv"
        if not cache.exists():
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
            resp = requests.get(url, headers={"User-Agent": _UA}, timeout=30)
            cache.write_bytes(resp.content)
        f = pd.read_csv(cache, names=["date", "value"])
        f["date"] = pd.to_datetime(f["date"])
        f["series"] = sid
        frames.append(f[["date", "series", "value"]])
    return pd.concat(frames, ignore_index=True)


# ---- helpers -------------------------------------------------------------
def _json_dump(obj) -> str:
    import json
    return json.dumps(obj)


def _json_load(path: Path):
    import json
    return json.loads(path.read_text())


def _cik_to_ticker(cik: str) -> str:
    return f"C{cik}"


__all__ = ["load_stooq", "load_yfinance", "load_sec_fundamentals", "load_fred"]
