#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
PY=$(resolve_python)
action=${1:-help}
if [[ $# -gt 0 ]]; then shift; fi
case "$action" in
  download) "$PY" scripts/download_openseeker.py "$@" ;;
  prepare) "$PY" scripts/prepare_openseeker.py "$@" ;;
  plan|train|eval) "$PY" scripts/experiment.py "$action" "$@" ;;
  test) "$PY" -m pytest tests/test_openseeker.py "$@" ;;
  *) echo 'Usage: bash scripts/openseeker.sh {download|prepare|plan|train|eval|test} [args]'; exit 2 ;;
esac
