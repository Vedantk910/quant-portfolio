"""Shared fixtures. The full-sample pipeline (``make report``) uses the big
seeded market; the test suite uses a small market so the suite stays fast.
Everything is deterministic: same seed, same data, same results.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# make the src/ layout importable without installation
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from alpha.data.synthetic import MarketSpec, generate_market, write_parquet  # noqa: E402
from alpha.data.store import PITStore  # noqa: E402

TEST_SPEC = MarketSpec(seed=7, n_tickers=30, start="2018-01-01", end="2021-12-31")


@pytest.fixture(scope="session")
def tables() -> dict[str, pd.DataFrame]:
    return generate_market(TEST_SPEC)


@pytest.fixture(scope="session")
def store(tmp_path_factory) -> PITStore:
    d = tmp_path_factory.mktemp("pitstore")
    write_parquet(d, generate_market(TEST_SPEC), TEST_SPEC)
    return PITStore.from_parquet(d)
