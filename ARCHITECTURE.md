# Architecture — Cross-Sectional Equity Alpha Research Pipeline

Companion to `docs/RESEARCH_NARRATIVE.md` (the program-level blueprint) and
`docs/research_note_flagship.md` (the results memo). This document fixes the
system design: module boundaries, data contracts, the point-in-time (PIT)
discipline, and the validation standard that every component obeys.

Status: **v1.0 — implemented.** The default dataset is a seeded synthetic
market so the whole pipeline is reproducible offline in one command; real
data loaders (Stooq/yfinance, SEC XBRL, FRED) plug into the same contracts
and the identical guardrails apply to them.

---

## 1. What the system is, in one paragraph

A vectorized cross-sectional research pipeline: given a date `t` and a
point-in-time universe `U(t)`, the pipeline assembles a decision-time-clean
panel of features and labels, fits a model on a time-ordered split, converts
forecasts into a market/beta/sector-neutral long-short portfolio, charges
turnover-proportional costs, and reports only out-of-sample, net-of-cost
metrics. Every stage is a pure function of `(panel, config, rng)` and every
run writes a reproducible experiment record. The system is intentionally
boring: pandas DataFrames, one function per concern, pytest on every module,
no framework whose behavior we cannot audit.

## 2. Data contract (the part that makes or breaks it)

The unit of exchange between modules is one long panel:

```
(date, ticker, <feature...>, <label...>)
```

- Rows are `(date, ticker)` decision points; each ticker appears once per
  date, on the **universe calendar** (business days when the ticker was
  tradeable and eligible).
- `date` = the decision date. Features are functions of information
  available **at the close of `date`**; labels are forward functions of
  information **after** `date`.
- **`available_at` discipline:** every non-price feature carries its
  availability semantics implicitly through its construction date. For
  fundamentals (P2 path) the rule is `available_at = filing_date + lag_days`,
  never the period-end date. The PIT store enforces this at load time and the
  validator re-checks a random sample (Section 5).
- Prices are corporate-action adjusted (splits and cash dividends), and the
  adjustment is applied **once, at the back end**: `adj_close(d)` for any
  earlier `d` bakes in all actions after `d`. This gives clean forward-return
  construction and, crucially, does not leak post-`d` information into
  point-in-time features. Price-return and total-return are separated by the
  dividend table, not by a magic flag.
- The universe is a DataFrame `U(date, ticker, listed_at, delisted_at)`.
  Eligibility on `t` is `listed_at <= t <= delisted_at` **and** a
  liquidity/size floor computed from trailing-30d dollar volume, which is
  itself a point-in-time quantity (we never look ahead at "final" status).

### The synthetic dataset (default)

`src/alpha/data/synthetic.py` generates a deterministic market:
- 120 tickers, 2008-01-01 → 2025-12-31, business-day calendar.
- A per-ticker latent "quality" factor + momentum + a PEAD-like post-earnings
  drift channel, with **delistings** (small names exit over time) so the
  survivorship structure is real and measurable: the generator records each
  ticker's `listed_at`/`delisted_at`, and the loader proves the universe was
  the right size on each date.
- A seeded RNG throughout; the seed is part of the experiment record.
- A deliberately *leaky* variant (`leaky=True`) exposes a feature equal to the
  next-period return. It exists for exactly one purpose: the guardrail test
  must reject it (Section 5).

### Real data (opt-in)

| Source          | Module                     | PIT rule                                   |
|-----------------|----------------------------|-------------------------------------------|
| Stooq / yfinance | `data/prices.py`           | adjusted series; universe **not** survivorship-safe — flagged |
| SEC XBRL companyfacts | `data/sec.py`        | usable on `filingDate`, never period end   |
| FRED / ALFRED    | `data/macro.py`            | ALFRED vintage used if available           |
| Ken French       | `data/benchmarks.py`       | benchmark only, never a feature            |

## 3. Module map

```
src/alpha/
  data/        loaders + PIT store + universe construction + manifest
  features/    one family per file: technical, fundamental, macro, composite
  labels/      forward excess returns, triple-barrier, neutralized labels
  models/      Ridge/ENet baseline, LightGBM, naive benchmark; common fit/predict
  validation/  purged K-fold w/ embargo, anchored walk-forward, PIT audit,
               deflated Sharpe, bootstrap IC CI
  portfolio/   rank -> neutral long-short weights (beta/sector neutral,
               liquidity caps), Ledoit-Wolf, risk metrics, attribution
  costs/       spread + commission + square-root impact + borrow; capacity curve
  backtest/    vectorized daily engine: weights -> costs -> net returns;
               turnover & cost-drag accounting
  reporting/   experiment records (JSON), plots, research-note fragments
```

Rules that keep this honest:
- No module below `validation/` or `portfolio/` ever shuffles time.
- All fitting (scalers, neutralization regressors, feature selection) happens
  on train folds only; `Transformers` are stateful objects with `fit`/`apply`.
- The backtest engine consumes **weights**, not model outputs; the portfolio
  layer owns forecast->weights. This split is what P4 (risk) plugs into.

## 4. The validation standard (enforced, not documented)

For every headline number in `docs/research_note_flagship.md`:

1. **PIT audit** — `validation/pit.py` samples N random rows, asserts each
   feature column's value at `(date, ticker)` equals its recomputation from
   strictly-prior data; fails the run otherwise.
2. **Purged K-fold + embargo** — `validation/purged.py`: k folds on the
   time-ordered panel; k-1 neighbors of each fold boundary in label-horizon
   distance are purged; an embargo gap after each train block. No random
   shuffle on the time axis, ever.
3. **Anchored walk-forward** — the headline OOS split: expanding train,
   rolling test, refit each test block. This is the number that goes in the
   abstract.
4. **Leak guardrail** — `tests/test_guardrail.py` runs the validator on the
   leaky dataset and asserts the flagged feature is *rejected* (its OOS rank
   IC is not reported as a result). A passing pipeline must catch it.
5. **Multiple testing** — deflated Sharpe with the number of trials logged
   from the experiment store; raw and deflated reported side by side.
6. **Kill criteria (pre-registered)** — `configs/preregistration.yaml`:
   OOS rank IC < 0.02, IC-IR < 0.3, or net Sharpe < 0.5 ⇒ the memo reports a
   negative result, full stop. Dated and committed before tuning.

## 5. Cost and capacity (first-pass model; P3 replaces it)

`costs/model.py`:

```
per-unit-turnover cost = commission + half_spread
                          + k * sigma_daily * sqrt(participation)
                          + borrow_rate/252 * short_weight
```

- Square-root impact, `k` calibrated on our own volume series from the data;
  participation capped at 25% of ADV.
- The engine reports **gross, net, and cost-drag** equity curves together —
  the trio, always. A single Sharpe is not a result.
- `costs/capacity.py` sweeps AUM at fixed monthly dollars traded → the
  break-even curve; 2×/3× slippage shocks as the robustness check.

## 6. Portfolio construction

- Signal → cross-sectional rank → long top decile / short bottom decile.
- Weights solved as the minimum-variance solution inside the long/short sets
  subject to: dollar-neutral, beta-neutral (market-proxy regression), sector-
  neutral (GICS-level buckets; 5 sectors in synthetic), per-name cap,
  turnover penalty (a `λ` multiplier on `‖w − w_prev‖₁`).
- Ledoit-Wolf shrinkage covariance; eigenvalue-spectrum plot in the note as
  the "why sample covariance is unusable" exhibit.
- Attribution: realized P&L regressed onto {market, sector-neutral, size,
  value, momentum, quality, residual} — so the memo can say whether the "alpha"
  is just levered momentum in disguise.

## 7. Reproducibility target

`make report` (scripts) = `uv sync && python -m alpha.run --config
configs/default.yaml`. That single command reproduces every number in the
research note from the seeded synthetic market, writes experiment JSON, and
renders the plots. For real data the same command with
`configs/real_data.yaml` is the only difference — same contracts, same
guardrails, same memo structure.

## 8. Definition of done (P0, per the blueprint)

- [x] "Buy the universe" net return within 20 bp of equal-weight universe
  net of estimated costs — `tests/test_universe.py` pins the number.
- [x] Deliberately leaky strategy caught by the validator —
  `tests/test_guardrail.py`.
- [x] PIT data dictionary (`docs/data_dictionary.md`), 5–8 pytest tests
  (we ship more; every module has tests).
- [x] Every backtest writes a JSON experiment record (config hash, data
  vintage, metrics, seed).

## 9. Known traps, and where each is handled

| Trap | Handled in |
|---|---|
| Survivorship bias (universe of living names only) | synthetic delistings + `data/universe.py` eligibility + memo paragraph |
| 10-K period-end vs filing-date | `data/sec.py` `as_of(filing_date)` rule, PIT audit |
| Random split on overlapping labels | `validation/purged.py` only; shuffles banned by lint test |
| In-sample Sharpe reported as a result | `reporting` emits `in_sample`/`oos` fields; note template only reads `oos` |
| Unrealistic costs | pessimistic defaults (10 bps + impact + borrow); 2× shock test |
| Multiple-testing inflation | deflated Sharpe, trial count from experiment log |
| No falsification | `configs/preregistration.yaml` kill criteria, checked by `reporting` |
