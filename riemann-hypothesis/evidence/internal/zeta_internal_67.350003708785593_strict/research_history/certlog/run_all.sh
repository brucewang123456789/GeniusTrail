#!/bin/sh
# Replays all 48 shards. Exits nonzero unless every shard reports PROVED.
set -e
g++ -O2 -o bb9 bb9.cpp
fail=0
for spec in "0.5 0.0067328" "0.95 0.0079032" "1.0 0.0080132"; do
  set -- $spec
  for i in $(seq 0 15); do
    out=$(./bb9 $1 $2 600000000 0.000244140625 16 $i 1e-9 | tail -1)
    echo "s=$1 shard $i: $out"
    echo "$out" | grep -q "result=PROVED" || fail=1
  done
done
[ $fail -eq 0 ] && echo "ALL 48 SHARDS PROVED" || { echo "FAILURE: a shard did not prove"; exit 1; }
