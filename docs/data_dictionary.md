# Data Dictionary — Point-in-Time Equity Store

Every table the pipeline reads or writes, with its columns, units, and the
point-in-time (PIT) rule that makes it safe to join. The store is a set of
parquet files plus a `manifest.json` (per-table SHA-256, row count, columns,
and the full generator spec). The generator spec is part of the manifest
because the default dataset is seeded: the same spec + seed reproduces the
same bytes, and a `PITStore.from_parquet` load re-hashes every table against
the manifest and refuses to run on a tampered or stale file.

The unit of exchange between modules is one long panel keyed on
`(date, ticker)`; `date` is the decision date (information available at that
day's close), and the label is a forward function of information *after*
`date`. Nothing below the `validation/` or `portfolio/` layers ever shuffles
time.

## `prices` (long, ~440k rows)

| column | type | meaning |
|---|---|---|
| `date` | date | decision date, business-day calendar |
| `ticker` | str | name, `T000`–`T119` |
| `open` | float | raw open (carries the split drop, previous close) |
| `close` | float | **raw** close, including split/dividend price drops |
| `adj_close` | float | **split- and dividend-adjusted** continuous close. Labels, returns, and features all use this — it bakes corporate actions into the *past* so forward-return construction is clean and no post-`date` information leaks into a decision-time feature |
| `volume` | float | share volume for the day |
| `dollar_volume` | float | `volume × close`, used by the liquidity floor |
| `mkt_cap` | float | `close × shares_out` |
| `listed` | float | 1.0 on days the name was tradeable; rows are dropped where 0, so the universe is survivorship-aware by construction |

Raw and adjusted closes are kept separate on purpose: a split is a price
drop with no economic content, and the adjusted series is the one that makes
forward returns economically meaningful. Total return is recovered by the
`dividends` table, not a magic flag on `close`.

## `market` (~4.7k rows)

| column | meaning |
|---|---|
| `date` | business day |
| `mkt_ret` | the market factor's one-day return. A GARCH(1,1)-flavored path with rare crash days; the neutralization proxy |

Forward-return labels are market-neutralized by subtracting the market's own
forward return over the *same* window, so a signal that is just beta gets
measured out.

## `dividends` (~7k rows)

| column | meaning |
|---|---|
| `date`, `ticker` | announcement day |
| `dps` | cash dividend per share that day |

## `splits` (~180 rows)

| column | meaning |
|---|---|
| `date`, `ticker` | split day |
| `factor` | split factor (0.5 = a 2-for-1) |

## `earnings` (~7k rows)

| column | meaning |
|---|---|
| `date`, `ticker` | announcement day (one standardized surprise per ~quarterly cycle) |
| `surprise` | standardized earnings surprise, the input to the PEAD drift channel |

The PEAD feature family reads only announcements dated on or before the
decision date; the post-announcement drift is already embedded in `adj_close`
as of `date`, so the feature is a clean, PIT-safe *recency* construct, not a
look-ahead.

## `fundamentals` (~4.7k rows, wide)

| column | meaning |
|---|---|
| `date` | business day |
| `roa_T###` | slowly-revealed ROA observation per ticker — the PIT stand-in for SEC XBRL `companyfacts` (in the real-data path, `available_at = filing_date`, never period-end) |

## `leaky` (~437k rows) — guardrail only

| column | meaning |
|---|---|
| `date`, `ticker` | |
| `leak` | **the forward return over the label horizon, joined back as if it were a feature.** This table is the classic rookie mistake ("the label is the feature"), and it exists for exactly one reason: the validator guardrail must catch it. Its Spearman IC against the forward label is ≈ 1.0, far above the 0.30 flag ceiling, and the reporting layer refuses to publish it. The test that proves the guardrail works is `tests/test_guardrail.py`; a green suite means a leak *would* be flagged, not that one slipped through. |

## `meta` (120 rows)

| column | meaning |
|---|---|
| `ticker` | name |
| `sector` | one of 5 sectors (for sector-neutral construction) |
| `listed_at`, `delisted_at` | the name's live window; delistings are real (≈45% of names exit mid-sample), so the universe is survivorship-aware |
| `shares_out` | shares outstanding, for market cap |

## Feature columns (`f_<family>_<name>`)

Built by `src/alpha/features/families.py` onto the long panel. Three families,
each a pure function of rows with `date' <= date`:

- **technical** (9): `mom_5_63` (5–63d momentum), `ret_1d`, `ret_21d`,
  `ret_126d`, `rev_5d` (short-term reversal), `vol_20d` (realized vol),
  `log_dvol_30d` (liquidity), `turnover_20d`, `log_mkt_cap` (size),
  `dist_52w_high`.
- **fundamental** (2): `roa_z` (cross-sectional z of ROA), `roa_growth`.
- **pead** (3): `pead_surprise_rec` (most-recent surprise, forward-filled ≤63d),
  `pead_cum_63d`, `pead_surprise_z`.

The primary model consumes all 14; the naive single-feature benchmark uses
only `f_technical_mom_5_63` so the note can say honestly how much the
complex model adds.

## Reproducibility anchor

`configs/default.yaml` names the seed (`20270101`), the generator spec, the
feature families, the model list, the validation split, the portfolio
constraints, and the cost model. `make report` (or
`python -m alpha.run --config configs/default.yaml`) regenerates
`data/*.parquet` and every figure and number in the research note from those
two config files. Commit the config and the manifest; do not commit the raw
prices.
