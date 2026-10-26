#!/usr/bin/env bash
# Regenerate every research artifact: data/, out/experiments/, out/figures/,
# out/research_note_flagship.md — one command, fully deterministic.
# Usage: bash scripts/make_report.sh [python-executable]
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${1:-$(command -v py >/dev/null 2>&1 && echo "py -3" || echo "python3")}"
PYTHONPATH=src $PY -m alpha.run --config configs/default.yaml
