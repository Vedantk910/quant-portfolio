"""Experiment records: every backtest writes a JSON record (config hash,
data vintage, metrics, seed) so that any number in the research note is
traceable to a reproducible run. The trial log is what feeds the deflated
Sharpe.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import yaml


def config_hash(config: dict) -> str:
    blob = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _sanitize(x):
    """NaN/Inf -> None, recursively; JSON forbids those tokens."""
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: _sanitize(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_sanitize(v) for v in x]
    return x


@dataclass
class ExperimentRecord:
    run_id: str
    config_hash: str
    seed: int
    data_vintage: str
    metrics: dict
    notes: str = ""

    def to_json(self, path: Path) -> None:
        # strict JSON: NaN/Inf tokens are not valid JSON (other tools refuse
        # to parse them); NaN -> null is the honest substitution
        text = json.dumps(_sanitize(asdict(self)), indent=2, default=str)
        path.write_text(text)


def write_experiment_record(out_dir: Path, record: ExperimentRecord, trial_log: Path | None = None) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{record.run_id}.json"
    record.to_json(p)
    if trial_log is not None:
        trial_log = Path(trial_log)
        trial_log.parent.mkdir(parents=True, exist_ok=True)
        try:
            log = json.loads(trial_log.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            log = []
        log.append({"run_id": record.run_id, "seed": record.seed, "metrics": record.metrics})
        trial_log.write_text(json.dumps(_sanitize(log), indent=2, default=str))
    return p


def headline_sentence(ic_mean: float, ic_ir: float, period: str, cost_bps: float, concentration: str = "") -> str:
    """The one sentence a hiring quant repeats to a colleague. The net-of-cost
    number leads; the raw Sharpe does not appear anywhere here."""
    conc = f", with the edge concentrated in {concentration}" if concentration else ""
    return (
        f"Out-of-sample rank IC of {ic_mean:.3f} (IC IR {ic_ir:.2f}) across {period}, "
        f"net of {cost_bps:.0f}bp round-trip costs{conc}."
    )


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


__all__ = ["ExperimentRecord", "write_experiment_record", "headline_sentence", "config_hash", "load_config"]
