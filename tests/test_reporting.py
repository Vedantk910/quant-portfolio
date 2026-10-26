"""Reporting layer: experiment records are reproducible, the kill-criteria
verdict is computed from the pre-registration, and OOS numbers are the ones
the note leads with."""
from __future__ import annotations

import json
from pathlib import Path

from alpha.reporting.experiment import ExperimentRecord, config_hash
from alpha.reporting.note import render_research_note


def _cfg() -> dict:
    return {"seed": 1, "data": {"source": "synthetic"}, "n": 2}


def test_config_hash_is_deterministic() -> None:
    assert config_hash(_cfg()) == config_hash(_cfg())
    assert config_hash(_cfg()) != config_hash({"seed": 2})


def test_experiment_record_roundtrip(tmp_path: Path) -> None:
    out_dir = tmp_path / "exps"
    rec = ExperimentRecord(
        run_id="run_1", config_hash=config_hash(_cfg()), seed=1,
        data_vintage="abc123", metrics={"oos_ic": 0.031},
    )
    p = out_dir / "run_1.json"
    out_dir.mkdir(parents=True)
    rec.to_json(p)
    loaded = json.loads(p.read_text())
    assert loaded["run_id"] == "run_1"
    assert loaded["seed"] == 1
    assert loaded["metrics"]["oos_ic"] == 0.031


def test_trial_log_appends(tmp_path: Path) -> None:
    out_dir = tmp_path / "exps"
    out_dir.mkdir(parents=True)
    from alpha.reporting.experiment import write_experiment_record

    rec1 = ExperimentRecord("run_1", "h1", 1, "v1", {"ic": 0.1})
    rec2 = ExperimentRecord("run_2", "h2", 2, "v2", {"ic": 0.2})
    trial_log = out_dir / "trials.json"
    write_experiment_record(out_dir, rec1, trial_log)
    write_experiment_record(out_dir, rec2, trial_log)
    log = json.loads(trial_log.read_text())
    assert len(log) == 2
    assert [e["run_id"] for e in log] == ["run_1", "run_2"]


def test_note_renders_kill_verdict(tmp_path: Path) -> None:
    out = tmp_path / "note.md"
    pre = {"kill_criteria": {"oos_rank_ic_min": 0.02, "oos_ic_ir_min": 0.30, "net_sharpe_min": 0.50}}
    metrics = {"oos": {"primary": {"ic_mean": 0.031, "ic_ir": 0.45, "net_sharpe": 0.6, "turnover": 1.2}}}
    render_research_note(metrics, pre, out_path=out)
    text = out.read_text()
    assert "**PASS**" in text, "all kill criteria met -> PASS"
    assert "Out-of-sample rank IC" in text


def test_note_renders_negative_result(tmp_path: Path) -> None:
    out = tmp_path / "note.md"
    pre = {"kill_criteria": {"oos_rank_ic_min": 0.02, "oos_ic_ir_min": 0.30, "net_sharpe_min": 0.50}}
    metrics = {"oos": {"primary": {"ic_mean": 0.005, "ic_ir": 0.10, "net_sharpe": 0.2, "turnover": 1.2}}}
    render_research_note(metrics, pre, out_path=out)
    text = out.read_text()
    assert "KILL" in text, "a sub-kill-criteria result must be reported as killed"
    assert "negative result" in text
