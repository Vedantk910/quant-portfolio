"""The one-command pipeline: ``python -m alpha.run --config configs/default.yaml``

Reproduces every number in ``docs/research_note_flagship.md`` from the raw
(seed-generated) data, writes the experiment JSON, and renders the figures +
research note. This is the reproducibility target the README promises.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .backtest.engine import run_backtest
from .costs.model import CostModel
from .data.synthetic import MarketSpec, generate_market, write_parquet
from .data.store import PITStore
from .data.universe import eligible_universe
from .features.families import build_feature_table
from .labels.forward import forward_return_label
from .models.base import build_model
from .portfolio.construction import construct_portfolio
from .portfolio.risk import risk_metrics, turnover as turnover_stats
from .reporting.experiment import (
    ExperimentRecord,
    config_hash,
    headline_sentence,
    load_config,
    write_experiment_record,
)
from .reporting.note import render_research_note
from .validation.metrics import deflated_sharpe, ic_stats
from .validation.pit import leaky_feature_check
from .validation.walkforward import run_anchored_walk_forward


def run(config_path: str = "configs/default.yaml", pre_path: str = "configs/preregistration.yaml",
        out_root: Path | str = "out") -> dict:
    cfg = load_config(config_path)
    pre = load_config(pre_path)
    out_root = Path(out_root)
    out_dir = out_root / "experiments"
    fig_dir = out_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(cfg.get("output", {}).get("data_dir", "data"))

    # 1. Data ---------------------------------------------------------
    dcfg = cfg["data"]
    spec = MarketSpec(
        seed=cfg.get("seed", 20270101),
        n_tickers=dcfg.get("n_tickers", 120),
        start=dcfg.get("start"),
        end=dcfg.get("end"),
    )
    tables = generate_market(spec)
    write_parquet(data_dir, tables, spec)
    store = PITStore.from_parquet(data_dir)

    # 2. Universe -----------------------------------------------------
    ucfg = cfg["universe"]
    uni = eligible_universe(
        store.get("prices"),
        store.get("meta"),
        min_history_days=ucfg.get("min_history_days", 60),
        min_dollar_volume=dcfg.get("min_dollar_volume_30d", 0.0),
        stable_by_construction=ucfg.get("stable_by_construction", False),
    )

    # 3. Features -----------------------------------------------------
    feat_long = build_feature_table(store, cfg["features"]["families"])
    feat_long = feat_long.merge(uni[["date", "ticker", "sector"]], on=["date", "ticker"], how="inner")
    fcols = [c for c in feat_long.columns if c.startswith("f_")]
    tech_cols = [c for c in fcols if c.startswith("f_technical_")]

    # 4. Labels -------------------------------------------------------
    lab = forward_return_label(store.get("prices"), cfg["labels"]["horizon"], store.get("market"))
    lab = lab.merge(uni[["date", "ticker"]], on=["date", "ticker"], how="inner")

    # 5. Validation: anchored walk-forward per model ------------------
    vcfg = cfg["validation"]
    specs = [
        (cfg["models"]["baseline"], fcols),
        (cfg["models"]["primary"], fcols),
    ]
    if cfg["models"].get("naive_benchmark"):
        specs.append(("naive_momentum", ["f_technical_mom_5_63"]))

    oos_metrics: dict[str, dict] = {}
    for name, cols in specs:
        factory = (lambda: build_model("naive_momentum", col="f_technical_mom_5_63")) if name == "naive_momentum" \
            else (lambda: build_model(name))
        res = run_anchored_walk_forward(
            feat_long, lab, factory, cols,
            n_test_blocks=vcfg.get("n_test_blocks", 8),
            test_block_days=vcfg.get("test_block_days", 63),
            purge_days=vcfg.get("purge", 21),
        )
        ic = ic_stats(res["ic_series"])
        oos_metrics[name] = {
            "ic_mean": ic["ic_mean"],
            "ic_ir": ic["ic_ir"],
            "n_dates": ic["n_dates"],
            "ic_series": res["ic_series"],
            "predictions": res["predictions"],
            "fcols": cols,
        }

    # 6. Portfolio + backtest on the primary model --------------------
    primary = cfg["models"]["primary"]
    pred = oos_metrics[primary]["predictions"]
    pwide = pred.pivot(index="date", columns="ticker", values="pred")
    sectors = feat_long.drop_duplicates("ticker").set_index("ticker")["sector"]
    pcfg = cfg["portfolio"]
    weights = construct_portfolio(
        pwide,
        long_q=pcfg.get("long_quantile", 0.10),
        short_q=pcfg.get("short_quantile", 0.10),
        max_weight=pcfg.get("max_weight", 0.05),
        sector_neutral=pcfg.get("sector_neutral", False),
        sectors=sectors,
        turnover_penalty=pcfg.get("turnover_penalty", 0.0),
    )
    ccfg = cfg["costs"]
    cost = CostModel(
        commission_bps=ccfg.get("commission_bps", 0.5),
        half_spread_bps=ccfg.get("half_spread_bps", 2.0),
        impact_k=ccfg.get("impact_k", 0.1),
        max_participation=ccfg.get("max_participation", 0.25),
        borrow_annual_pct=ccfg.get("borrow_annual_pct", 2.0),
    )
    # The engine treats every row of `weights` as a rebalance; thin the daily
    # OOS predictions to the execution cadence so turnover (and therefore
    # cost drag) is measured at the intended frequency, not daily.
    rb = cfg["execution"]["rebalance"]
    weights_rb = weights.iloc[::rb]
    bt = run_backtest(weights_rb, store.get("prices"), rebalance=rb, cost=cost)
    oos_metrics[primary].update(
        gross_sharpe=bt.metrics["gross_sharpe"],
        net_sharpe=bt.metrics["net_sharpe"],
        turnover=bt.metrics["avg_turnover_per_rebalance"],
        risk=risk_metrics(bt.net),
        bt=bt,
    )
    tstats = turnover_stats(weights)

    # 7. Multiple testing (deflated Sharpe) --------------------------
    # The DSR statistic runs on per-decision-period (rebalance-window)
    # returns, not the zero-padded daily series — padding with zeros
    # inflates N and flattens the skew/kurtosis moments it corrects.
    n_trials = len(specs) * vcfg.get("n_test_blocks", 8)
    # annualize with the REBALANCE frequency, not calendar days: the
    # decision period is `rb` days, so the "sharpe" field matches the
    # engine's net_sharpe (which uses 252/rb)
    oos_metrics[primary]["deflated_sharpe"] = deflated_sharpe(
        bt.metrics.get("net_window", bt.net), n_trials, ann=252.0 / rb
    )

    # 8. Guardrails ---------------------------------------------------
    leak = leaky_feature_check(store, feat_long, lab, threshold=0.30)

    # 9. Figures ------------------------------------------------------
    _figures(bt, oos_metrics, fig_dir)

    # 10. Experiment record + research note --------------------------
    # keep only scalar metrics in the record (drop DataFrames / BT objects)
    rec_metrics = {}
    for name, m in oos_metrics.items():
        slim = {
            "ic_mean": m.get("ic_mean"),
            "ic_ir": m.get("ic_ir"),
            "n_dates": m.get("n_dates"),
        }
        if "net_sharpe" in m:
            slim["gross_sharpe"] = m["gross_sharpe"]
            slim["net_sharpe"] = m["net_sharpe"]
            slim["turnover"] = m["turnover"]
            slim["deflated_sharpe"] = m.get("deflated_sharpe")
        rec_metrics[name] = slim

    rec = ExperimentRecord(
        run_id=f"run_{cfg.get('seed', 0)}",
        config_hash=config_hash(cfg),
        seed=int(cfg.get("seed", 0)),
        data_vintage=store.vintage().get("tables", {}).get("prices", {}).get("sha256", ""),
        metrics=rec_metrics,
        notes=json.dumps({"leaky_flag": leak, "turnover": tstats, "n_trials": n_trials}, default=str),
    )
    p = write_experiment_record(out_dir, rec, out_dir / "trials.json")

    note_metrics = {"oos": rec_metrics}
    note = render_research_note(
        note_metrics, pre, out_path=fig_dir.parent / "research_note_flagship.md",
        primary=primary,
    )

    primary_m = oos_metrics[primary]
    # the headline period is the OOS window (where the numbers come from),
    # not the full sample — leading with the training span would misstate
    # what was measured out of sample
    oos_ic = primary_m.get("ic_series")
    period = (
        f"{oos_ic['date'].min().date()} to {oos_ic['date'].max().date()} (OOS)"
        if oos_ic is not None and len(oos_ic)
        else f"{lab['date'].min().date()} to {lab['date'].max().date()}"
    )
    print(headline_sentence(
        primary_m["ic_mean"],
        primary_m["ic_ir"],
        period,
        (ccfg.get("commission_bps", 0.5) + ccfg.get("half_spread_bps", 2.0)) * 2,
    ))
    print(f"  OOS net Sharpe (primary): {primary_m.get('net_sharpe')}")
    print(f"  leaky guardrail: {'FLAGGED (excluded)' if leak['flagged'] else 'clean'}  leaky_ic={leak['leaky_ic']:.3f}")
    print(f"  experiment record -> {p}")
    print(f"  research note     -> {note}")
    return {"metrics": oos_metrics, "record": rec, "leak": leak}


def _figures(bt, oos_metrics: dict, fig_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        eq = bt.equity("net")
        axes[0, 0].plot(eq.index, eq.values)
        axes[0, 0].set_title("Net equity (primary)")
        axes[0, 1].plot(eq.index, bt.equity("gross").values, alpha=0.5, label="gross")
        axes[0, 1].plot(eq.index, eq.values, label="net")
        axes[0, 1].legend()
        axes[0, 1].set_title("Gross vs net")
        ic = oos_metrics.get("primary", {}).get("ic_series")
        if ic is not None and len(ic):
            axes[1, 0].plot(ic.set_index("date")["ic"], marker=".", ms=3)
            axes[1, 0].axhline(0, color="k", lw=0.5)
            axes[1, 0].set_title("OOS rank IC by date")
        axes[1, 1].bar(bt.cost_drag.index, bt.cost_drag.values, width=1.0)
        axes[1, 1].set_title("Cost drag per rebalance")
        fig.tight_layout()
        fig.savefig(fig_dir / "equity_curves.png", dpi=110)
        plt.close(fig)
    except Exception as e:  # pragma: no cover
        print(f"  (figure render skipped: {e})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--prereg", default="configs/preregistration.yaml")
    ap.add_argument("--out", default="out")
    args = ap.parse_args()
    run(args.config, args.prereg, args.out)


if __name__ == "__main__":
    main()
