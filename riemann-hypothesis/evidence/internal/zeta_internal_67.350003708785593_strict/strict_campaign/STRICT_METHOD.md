# Strict finite-dimensional certificate method for the 67.350003708785593% witness

This campaign replaces the historical `bb9` acceptance path with a fail-closed verifier whose
floating storage is explicitly outward-rounded and whose transcendental kernel is enclosed by
interval Taylor arithmetic rather than trusted libm point values.

The proof uses two independently checkable acceptance paths. The `s=1/2` certificate is a direct strict B&B run with no well shortcut. The two tighter targets (`s=19/20` and `s=1`) use the certified-well plus global-complement path below.

1. **Local convex wells.** 327 small boxes around the numerically difficult wells are proved
   directly by `strict_wells_batch_v2.cpp`.  It uses the exact rational configuration from
   `cfg8_rat.h`, interval sinc/sin/cos, a rigorous lower table for `W''`, interval LDL to prove a
   positive Hessian lower bound, and a strong-convexity lower bound.  The current frozen result is
   `327/327 PROVED`, with the smallest strict gap still positive.
2. **Global complement for the two tighter targets.** `bb26_strict_wells_symmetry.cpp` performs exhaustive interval
   branch-and-bound on the remaining domain. The direct `s=1/2` run is instead produced by `bb24_strict_symmetry.cpp`, which uses the same strict interval machinery but no well shortcut.  A box may be discarded only by a rigorous lower
   bound, by containment in one of the independently proved well boxes, or by reversal symmetry
   after an exact symmetry preflight.  Node-cap exhaustion, terminal hard boxes, or any unresolved
   root return nonzero/`INCOMPLETE`.

The local targets are

- `s = 1/2`, `epsilon = 526/78125 = 0.0067328`;
- `s = 19/20`, `epsilon = 9879/1250000 = 0.0079032`;
- `s = 1`, `epsilon = 20033/2500000 = 0.0080132`.

After all three local inequalities close, `verify_exact.py` performs the block-envelope assembly
in exact rational arithmetic and returns

`722547711262091300265625000000000 / 1072825050442925667061714615173641`

= `67.350003708785593%`.

## Trust boundary

The finite-dimensional certificate is a computer-assisted proof artifact, not a statement that
external peer review or the Lean challenge has already accepted the result.  The public release
therefore includes source, compiler flags, raw logs, exact-rational assembly, SHA-256 manifests,
and a clean replay script.  The analytic attachment to the unconditional Zeta23 baseline is
identified separately; an accepted Lean `Solution.lean` is not claimed here.
