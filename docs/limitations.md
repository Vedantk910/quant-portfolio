# Limitations — What This Work Will Not Claim

A research note that can't be attacked is a research note that proves nothing.
This page is the list of attacks, and where each one lands. Read it before you
read the numbers; it is the part that separates a defensible result from a
sloppy one.

## The numbers are on a synthetic market, not on real equities

The default dataset is a seeded 120-ticker generator I wrote so the whole
pipeline runs offline and deterministically. The OOS rank IC (0.149), the net
Sharpe (2.67), and the capacity figures are all properties of *that* generator,
not of the S&P 500 or any real universe. The generator embeds a learnable
cross-sectional structure (momentum carry, post-earnings drift, quality) on
purpose, so the models are *expected* to find it. This is a test of the
pipeline — that a correct validator catches a leak, that costs are charged,
that the split is purged — not a discovery of a new tradable factor. Treat
every number in `out/research_note_flagship.md` as "the pipeline works on data
of this shape," not as an alpha estimate you would deploy against real money.

## The real-data path exists but is not the result being reported

`src/alpha/data/loaders.py` (Stooq/yfinance, SEC XBRL `companyfacts` on
filing-date, FRED/ALFRED) plugs into the identical contracts and guardrails,
but the headline result was produced on the synthetic store. I will not claim
a real-market Sharpe here. If you run the same config against real data,
you are running a new experiment with a new config hash; its numbers are not
these numbers.

## The multiple-testing correction did *not* clear the bar, and I report it

The deflated Sharpe (Bailey–López de Prado) over the 24 rebalance-window
observations is **DSS = 0.75 → not significant at the 5% level** after the
trial correction. The raw net Sharpe of 2.67 is the optimistic number; the
deflated result is the one that survives model selection, and it does not
survive. I am reporting both, and I am not promoting the raw number. `n_trials
= 24` is itself an underestimate of the trials a real researcher runs; the true
correction is more punitive, which makes the "not significant" verdict even
more settled.

## Small out-of-sample window

483 OOS rebalance dates total, but the net-Sharpe figure is computed on ~24
rebalance windows per block and the DSS on 24 observations. That is a small
sample for any Sharpe or DSS. The bootstrap CI on the IC is reported next to
the IC for this reason, and the conclusion of this note holds only to the
precision of that window.

## The cost model is first-pass, not calibrated

`costs/model.py` is commission + half-spread + square-root impact + borrow,
with the impact coefficient chosen by convention, not fit to execution
data. The P3 project is the calibrated replacement; until it lands, the
cost drag is an order-of-magnitude estimate. The 3× slippage shock
(`CostModel.shock(3.0)`) is provided precisely so a reader can see how much
of the net Sharpe is optimistic-costs assumption and how much is robust. A
result that survives only the optimistic-cost run is flagged, in the note, as
not a result.

## Sector neutrality is coarse

Sector neutrality is enforced against 5 generator sectors, not a GICS/CRSP
mapping. Within-sector demeaning over 5 buckets is a meaningful but rough
control; style (size, value, momentum) is neutralized by construction in the
rank-sleeve, but not by a full factor-regression attribution, which is the
P4 project.

## What would falsify this work

- **The leak guardrail stops firing.** If `tests/test_guardrail.py` passes
  while the leaky feature's Spearman IC is *below* the 0.30 ceiling, the
  validator has gone blind and every OOS number in the repo is
  untrustworthy. The guardrail is load-bearing; its passing is a precondition
  to reading the headline.
- **The equal-weight pin breaks.** `tests/test_universe.py` must land the
  equal-weight universe net return within 20bp of an independent reference
  implementation. If it drifts, the backtest engine is mis-summing weights
  or charging phantom costs and the whole P&L is suspect.
- **The deflated Sharpe becomes significant and the raw one doesn't.** That
  is incoherent and would indicate a bug in the trial-count plumbing.
- **A pre-registered kill criterion is met on paper but the OOS period shifts
  during tuning.** The split is anchored and purged; if the OOS window ever
  overlaps the training window in a future config, the OOS number is no
  longer OOS and the pre-registration is void.

## Bottom line

I will claim: the pipeline is reproducible, the validator catches a known
leak, the universe is survivorship-aware, costs are charged net, and the
multiple-testing correction is reported honestly (and it did not pass).
I will not claim: a real-market alpha, a calibrated cost, a significant
deflated Sharpe, or that the synthetic numbers predict anything about live
trading. The honest version of this work is "a defensible research harness
that I would build before I trusted any of my own backtests," and that is
what the P0/P1 blueprint asked for.
