# Reproducibility target: `make report` reproduces every number in the
# research note from the seeded dataset. `make test` is the full suite.
#
# Python is found via `py` (Windows launcher) by default; override with
# `make test PY=python3.11` on machines without the launcher.

PY ?= py -3

.PHONY: test lint report clean

## Run the whole test suite (PIT audit, guardrail, 20bp pin, costs, labels)
test:
	$(PY) -m pytest -q

## Lint (skips silently if ruff is not installed — not load-bearing)
lint:
	$(PY) -m ruff check src tests || echo "(ruff not installed — skipped)"

## Regenerate data/, out/experiments, out/figures, out/research_note_flagship.md
report:
	PYTHONPATH=src $(PY) -m alpha.run --config configs/default.yaml

## Drop generated artifacts and re-derive everything
clean:
	rm -rf out data .pytest_cache
	$(MAKE) report
