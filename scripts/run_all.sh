#!/usr/bin/env bash
# One-command reproducibility: test, then regenerate every artifact.
# Usage: bash scripts/run_all.sh [python-executable]
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${1:-$(command -v py >/dev/null 2>&1 && echo "py -3" || echo "python3")}"
echo "== pytest =="
$PY -m pytest -q
echo "== report =="
PYTHONPATH=src $PY -m alpha.run --config configs/default.yaml
echo "done."
