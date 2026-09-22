#!/usr/bin/env bash
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
export OMP_NUM_THREADS=5
HC=0.000244140625
CAP=2000000000
WMIN=0.001
run_one(){
  name="$1"; s="$2"; eps="$3"
  echo "===== ${name} BEGIN $(date -u +%FT%TZ) =====" > "${name}.status"
  /usr/bin/time -f 'ELAPSED=%e RSS_KB=%M EXIT=%x' \
    "$HERE/bb24_strict_symmetry" "$s" "$eps" "$CAP" "$HC" 1 0 "$WMIN" \
    > "${name}.log" 2> "${name}.time"
  ec=$?
  cat "${name}.time" >> "${name}.status"
  tail -1 "${name}.log" >> "${name}.status"
  echo "EXIT_CODE=$ec" >> "${name}.status"
  echo "===== ${name} END $(date -u +%FT%TZ) =====" >> "${name}.status"
  [ "$ec" -eq 0 ] || exit "$ec"
}
: > CAMPAIGN.status
run_one s0.5 0.5 0.0067328
run_one s0.95 0.95 0.0079032
run_one s1 1 0.0080132
echo 'ALL_THREE_LOCAL_CERTIFICATES=PROVED' >> CAMPAIGN.status
sha256sum *.log *.status *.time bb24_strict_symmetry bb24_strict_symmetry.cpp cfg8.h ct_rig14_*.bin ft_rig18_*.bin 2>/dev/null > CAMPAIGN_SHA256.txt
echo "CAMPAIGN_DONE=$(date -u +%FT%TZ)" >> CAMPAIGN.status
