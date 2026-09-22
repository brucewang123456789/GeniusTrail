# Independent review checklist

This file is deliberately short: it identifies the trust boundary a reviewer should attack first.

1. **Exact target.** Run `python3 verify_exact.py`; all assembly is rational and the final inequality is strict.
2. **Constant direction.** Run `python3 audit_constants.py`; every coefficient/epsilon/slope enclosure must point in the conservative direction.
3. **Direct certificate.** Inspect/replay `s=1/2` with `bb24_strict_symmetry.cpp`; acceptance requires `stack_left=0`, `HARD=0`, `result=PROVED`.
4. **Well certificates.** Compile/run `strict_wells_batch_v2.cpp`; all 327 boxes must be proved independently. A listed well is not trusted merely because it appears in the header.
5. **Complement certificates.** Replay `s=19/20` and `s=1` with `bb26_strict_wells_symmetry.cpp`; well pruning is allowed only by full containment in the independently checked boxes.
6. **Coverage.** Root components are generated from rigorous one-dimensional exclusion bounds; reversal pruning is enabled only after the verifier's exact symmetry preflight. Shards use the verifier-native `NSH/SH` partition and must collectively cover every retained root ordinal.
7. **Fail closed.** Node-cap exhaustion, terminal unresolved boxes, symmetry failure, failed well checks, or a nonzero checker exit are rejection conditions.
8. **Fresh replay.** `FULL=1 ./reproduce.sh` rebuilds source and deletes derived lookup caches before the expensive replay.

The artifact's claim is the strict finite-dimensional certificate. The separate analytic/Lean attachment is not silently promoted to an accepted end-to-end theorem.
