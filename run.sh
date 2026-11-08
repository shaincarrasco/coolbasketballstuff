#!/usr/bin/env bash
# One-command launcher for the NBA projects. Usage: ./run.sh 1|2|3|all
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"

open_report() {
  local f="$1"
  echo "Report: $f"
  if [ "${NO_OPEN:-0}" = "1" ]; then return 0; fi
  if [ "$(uname)" = "Darwin" ]; then open "$f" 2>/dev/null || true
  elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$f" >/dev/null 2>&1 || true
  fi
}

run_one() {
  case "$1" in
    1) d="01-jumpshot-analyzer" ;;
    2) d="02-lineup-intelligence" ;;
    3) d="03-game-predictor" ;;
    *) echo "unknown project: $1"; exit 1 ;;
  esac
  echo "▶ $d"
  ( cd "$ROOT/$d" && "$PY" cli.py demo )
  [ -f "$ROOT/$d/output/report.html" ] && open_report "$ROOT/$d/output/report.html"
}

case "${1:-all}" in
  all) run_one 1; run_one 2; run_one 3 ;;
  *) run_one "$1" ;;
esac
