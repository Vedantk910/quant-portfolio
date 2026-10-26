"""Point-in-time data store.

The store owns the assembled dataset and answers PIT queries:

- ``get("prices")`` -> long panel of tradeable rows
- ``as_of(ticker, date)`` -> the row's fundamental state as of ``date``
  (never later, by construction)
- ``panel(features, universe)`` -> a decision-time-clean (date, ticker) panel

The manifest (sha256 per table, row counts, spec) is committed alongside the
Parquet files; a run that loads a store verifies the manifest before use.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class PITStore:
    tables: dict[str, pd.DataFrame]
    manifest: dict

    # ------------------------------------------------------------------ load
    @classmethod
    def from_parquet(cls, data_dir: str | Path) -> "PITStore":
        data_dir = Path(data_dir)
        tables = {
            p.stem: pd.read_parquet(data_dir / f"{p.stem}.parquet")
            for p in sorted(data_dir.glob("*.parquet"))
        }
        manifest = json.loads((data_dir / "manifest.json").read_text())
        store = cls(tables, manifest)
        store.verify_integrity()
        return store

    def verify_integrity(self) -> None:
        """Re-hash every table and compare against the committed manifest."""
        import hashlib

        for name, df in self.tables.items():
            entry = self.manifest.get(name)
            if entry is None:
                continue
            digest = hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()[:16]
            if digest != entry["sha256"]:
                raise ValueError(f"manifest mismatch for {name}: got {digest}, expected {entry['sha256']}")

    # ------------------------------------------------------------------ access
    def get(self, name: str) -> pd.DataFrame:
        return self.tables[name]

    def as_of(self, table: str, date, ticker: str) -> dict | None:
        """The record for ``ticker`` in ``table`` as of ``date`` — the PIT rule:
        only rows with ``date <= given`` participate. Fundamentals carry their
        own availability semantics (filing date, not period end)."""
        df = self.tables[table]
        sel = df[(df["date"] <= pd.Timestamp(date))]
        sel = sel[sel["ticker"] == ticker]
        if sel.empty:
            return None
        return sel.iloc[-1].to_dict()

    def panel(self, universe: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
        """Assemble (date, ticker, *features) over the eligible universe, joined
        on (date, ticker) — the as-of join. Feature columns come from the
        pre-joined price feature tables; missing = NaN, never forward-filled
        beyond their availability date."""
        prices = self.get("prices")
        panel = universe.merge(prices[["date", "ticker"] + feature_cols], on=["date", "ticker"], how="left")
        return panel

    def vintage(self) -> dict:
        return {
            "seed": self.manifest.get("seed"),
            "tables": {k: self.manifest[k] for k in self.tables if k in self.manifest},
        }
