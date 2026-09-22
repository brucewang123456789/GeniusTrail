#!/bin/sh
# Reproduces every unconditional claim in this repository.
set -e
echo "== exact rational chain (no floating point in any acceptance test) =="
python3 verify_exact.py
echo
echo "== Arb: lower bound for H, and positivity of the window =="
python3 arb_window.py
echo
echo "== refutation witness =="
python3 refute.py
echo
echo "All reproducible claims checked. Local certificates: run sh certificates/run_all.sh (48 shards)."
