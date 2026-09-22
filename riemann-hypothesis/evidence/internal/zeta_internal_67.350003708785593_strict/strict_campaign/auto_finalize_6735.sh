#!/usr/bin/env bash
set -euo pipefail
BASE=/mnt/data/veriloop_zeta_finalize
CAMP="$BASE/high67_cert_v24"
echo "[$(date -u +%FT%TZ)] AUTO_FINALIZER_WAITING" > "$CAMP/auto_finalize.log"
while [[ ! -f "$CAMP/SUPERVISOR.status" ]]; do sleep 3; done
st="$(tr -d '\r\n' < "$CAMP/SUPERVISOR.status")"
echo "[$(date -u +%FT%TZ)] SUPERVISOR_STATUS=$st" >> "$CAMP/auto_finalize.log"
[[ "$st" == "PASS" ]] || { echo "ABORT_NONPASS" >> "$CAMP/auto_finalize.log"; exit 2; }
python3 "$BASE/build_final_6735.py" >> "$CAMP/auto_finalize.log" 2>&1
echo "[$(date -u +%FT%TZ)] AUTO_FINALIZER_DONE" >> "$CAMP/auto_finalize.log"
