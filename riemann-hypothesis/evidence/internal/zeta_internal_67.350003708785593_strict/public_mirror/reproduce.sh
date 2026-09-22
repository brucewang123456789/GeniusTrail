#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
python3 audit_constants.py
python3 verify_exact.py
python3 verify_strict.py
if [[ "${FULL:-0}" == "1" ]]; then exec ./run_strict.sh; fi
echo 'COMMITTED_EVIDENCE_AUDIT=PASS'
echo 'For a clean expensive replay: FULL=1 ./reproduce.sh'
