# Strict local certificates

Author: Libo Wang.

| slope | epsilon | nodes | terminal state |
|---:|---:|---:|---|
| 1/2 | 526/78125 | 23,644,472 | PROVED; direct strict B&B; stack=0; HARD=0 |
| 19/20 | 9879/1250000 | 75,294,070 | PROVED; certified wells + complement; stack=0; HARD=0 |
| 1 | 20033/2500000 | 91,437,288 | PROVED; 5 exact root shards; certified wells + complement |
| **total** | | **190,375,830** | **3/3 PROVED** |

Before either well-aware global certificate may use a local-well shortcut, `strict_wells_batch_v2.cpp` independently proves all 327 frozen well boxes from the exact-rational configuration. The minimum well gap in the committed batch remains strictly positive.

Raw logs are under `certificates/strict/`; `run_strict.sh` rebuilds both checkers and replays the chain fail-closed.
