#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
PY=python3
command -v python3 >/dev/null 2>&1 || PY=python
"$PY" run.py "$@"
