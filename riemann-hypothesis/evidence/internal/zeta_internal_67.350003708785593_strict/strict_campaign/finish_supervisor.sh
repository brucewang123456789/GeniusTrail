#!/usr/bin/env bash
set -u
D=/mnt/data/veriloop_zeta_finalize/high67_cert_v24
cd "$D"
log(){ printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a supervisor.log; }
fail(){ log "FAIL: $*"; echo FAIL > SUPERVISOR.status; exit 1; }
passlog(){ grep -Eq 'stack_left=0 HARD=0 .*result=PROVED' "$1"; }

log 'Waiting for s=19/20 bb26 full strict run (PID 6711)'
while kill -0 6711 2>/dev/null; do sleep 5; done
sleep 1
[[ -f s0.95_bb26.exit ]] || fail 's0.95 process ended without exit file'
[[ "$(cat s0.95_bb26.exit)" == "0" ]] || fail "s0.95 exit=$(cat s0.95_bb26.exit)"
passlog s0.95_bb26.log || fail 's0.95 missing fail-closed PROVED terminal'
cp -f s0.95_bb26.log s0.95_final.log
cp -f s0.95_bb26.time s0.95_final.time
{
  echo "===== s0.95 FINAL STRICT ====="
  cat s0.95_bb26.time
  grep -E 'slope=.*stack_left=.*HARD=.*result=' s0.95_bb26.log | tail -1
  echo "EXIT_CODE=0"
} > s0.95_final.status
log 's=19/20 PROVED; final evidence frozen'

# Old paused exploratory bb24 is no longer needed.
kill -KILL 6255 6254 2>/dev/null || true

# Resume the already-started shard 0, preserving its sunk work.
kill -CONT 7022 2>/dev/null || true
log 'Resumed s=1 shard 0; launching shards 1..4 on remaining cores'

pids=()
for sh in 1 2 3 4; do
  rm -f "s1_bb26_sh${sh}.log" "s1_bb26_sh${sh}.time" "s1_bb26_sh${sh}.exit"
  (
    export OMP_NUM_THREADS=1
    /usr/bin/time -f 'ELAPSED=%e RSS_KB=%M EXIT=%x' \
      ./bb26_strict_wells_symmetry 1 0.0080132 2000000000 0.000244140625 5 "$sh" 0.001 \
      > "s1_bb26_sh${sh}.log" 2> "s1_bb26_sh${sh}.time"
    rc=$?; echo "$rc" > "s1_bb26_sh${sh}.exit"; exit "$rc"
  ) &
  pids+=("$!")
done

# Wait for child shards 1..4.
for p in "${pids[@]}"; do wait "$p" || true; done
# Shard 0 belongs to the earlier wrapper; wait for its exit marker.
while kill -0 7022 2>/dev/null; do sleep 3; done
for i in $(seq 1 30); do [[ -f s1_bb26_sh0.exit ]] && break; sleep 1; done

for sh in 0 1 2 3 4; do
  [[ -f "s1_bb26_sh${sh}.exit" ]] || fail "s1 shard $sh missing exit file"
  [[ "$(cat s1_bb26_sh${sh}.exit)" == "0" ]] || fail "s1 shard $sh exit=$(cat s1_bb26_sh${sh}.exit)"
  passlog "s1_bb26_sh${sh}.log" || fail "s1 shard $sh missing fail-closed PROVED terminal"
  log "s=1 shard $sh PROVED"
done

python3 - <<'PY'
from pathlib import Path
import re
D=Path('/mnt/data/veriloop_zeta_finalize/high67_cert_v24')
parts=[]; total=0
for sh in range(5):
    p=D/f's1_bb26_sh{sh}.log'
    t=p.read_text(errors='replace')
    ms=re.findall(r'slope=1(?:\.0+)? eps=0\.0080132 nodes=(\d+) stack_left=(\d+) HARD=(\d+).*result=(PROVED|INCOMPLETE)',t)
    if not ms: raise SystemExit(f'missing terminal shard {sh}')
    n,st,hard,res=ms[-1]
    if (st,hard,res)!=('0','0','PROVED'): raise SystemExit(f'bad terminal shard {sh}: {ms[-1]}')
    total+=int(n); parts.append((sh,int(n)))
out=D/'s1_final.log'
with out.open('w') as f:
    f.write('===== AGGREGATED STRICT s=1 CERTIFICATE; 5/5 EXACT ROOT SHARDS =====\n')
    f.write('Shard partition is verifier-native NSH=5, SH=0..4; aggregation occurs only after each shard independently terminates stack_left=0 HARD=0 result=PROVED.\n')
    for sh,n in parts:
        f.write(f'\n===== SHARD {sh}/5 nodes={n} =====\n')
        f.write((D/f's1_bb26_sh{sh}.log').read_text(errors='replace'))
        if not f.tell(): pass
    f.write(f'\nslope=1 eps=0.0080132 nodes={total} stack_left=0 HARD=0 worst_lb=NA (aggregated 5/5 strict shards) result=PROVED\n')
(D/'s1_final.status').write_text('S1_SHARDS_PROVED=5/5\nTOTAL_NODES=%d\nstack_left=0\nHARD=0\nresult=PROVED\n' % total)
with (D/'s1_final.time').open('w') as f:
    for sh in range(5):
        f.write(f'SHARD_{sh}: '+(D/f's1_bb26_sh{sh}.time').read_text(errors='replace').strip()+'\n')
print('S1_TOTAL_NODES',total)
PY

# Freeze the already completed stronger s=1/2 proof (bb24; no well shortcut).
cp -f s0.5.log s0.5_final.log
cp -f s0.5.time s0.5_final.time
cp -f s0.5.status s0.5_final.status

python3 verify_strict.py > verify_strict_final.log 2>&1 || fail 'verify_strict.py failed'
grep -q 'RESULT: ALL STRICT 67.35 CHECKS PASS' verify_strict_final.log || fail 'verify_strict result not PASS'
log 'ALL THREE STRICT LOCAL CERTIFICATES + 327 WELLS + EXACT ASSEMBLY PASS'
echo PASS > SUPERVISOR.status
