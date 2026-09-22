#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
BUILD="${BUILD_DIR:-$ROOT/.strict_build}"
OUT="${OUT_DIR:-$ROOT/certificates/replay}"
mkdir -p "$BUILD" "$OUT"
CXX=${CXX:-g++}
FLAGS=(-O3 -std=c++17 -fopenmp -frounding-math -ffp-contract=off)
cp "$ROOT/verifier/"{cfg8.h,cfg8_rat.h,wells_generated.h,strict_core.hpp} "$BUILD/"
"$CXX" "${FLAGS[@]}" "$ROOT/verifier/strict_wells_batch_v2.cpp" -I"$ROOT/verifier" -o "$BUILD/strict_wells_batch_v2"
"$CXX" "${FLAGS[@]}" "$ROOT/verifier/bb24_strict_symmetry.cpp" -I"$ROOT/verifier" -o "$BUILD/bb24_strict_symmetry"
"$CXX" "${FLAGS[@]}" "$ROOT/verifier/bb26_strict_wells_symmetry.cpp" -I"$ROOT/verifier" -o "$BUILD/bb26_strict_wells_symmetry"
(cd "$ROOT" && python3 audit_constants.py) | tee "$OUT/audit_constants.log"
(cd "$BUILD" && OMP_NUM_THREADS="${OMP_NUM_THREADS:-5}" ./strict_wells_batch_v2) | tee "$OUT/wells_v2.log"
grep -q 'WELLS_V2_TOTAL=327 PROVED=327 FAILED=0 .*RESULT=PROVED' "$OUT/wells_v2.log"
rm -f "$BUILD"/ct_rig14_*.bin "$BUILD"/ft_rig18_*.bin
runone_direct(){
  local name=$1 slope=$2 eps=$3
  (cd "$BUILD" && OMP_NUM_THREADS="${OMP_NUM_THREADS:-5}" ./bb24_strict_symmetry "$slope" "$eps" 2000000000 0.000244140625 1 0 0.001) | tee "$OUT/${name}.log"
  grep -q 'stack_left=0 HARD=0 .*result=PROVED' "$OUT/${name}.log"
}
runone_wells(){
  local name=$1 slope=$2 eps=$3
  (cd "$BUILD" && OMP_NUM_THREADS="${OMP_NUM_THREADS:-5}" ./bb26_strict_wells_symmetry "$slope" "$eps" 2000000000 0.000244140625 1 0 0.001) | tee "$OUT/${name}.log"
  grep -q 'stack_left=0 HARD=0 .*result=PROVED' "$OUT/${name}.log"
}
runone_direct s0.5 0.5 0.0067328
runone_wells s0.95 0.95 0.0079032
runone_wells s1 1 0.0080132
(cd "$ROOT" && python3 verify_exact.py)
echo 'STRICT_67_350003708785593_REPLAY=PASS'
