# Internal archive — complete computational record
Author: Libo Wang.

The finite-dimensional layer of the q=8 witness `C = 67.350003708785593%` is closed:
three eight-dimensional local certificates, 48 shards, zero unresolved boxes.
This archive holds everything needed to re-derive, re-run or audit that, including
material excluded from the public repository.

## Layout
    public_mirror/  exact copy of the public repository
    scripts/        every script run, in the order run
    data/           configurations, witnesses, minimiser pools, checkpoints
    verifier/       the whole evolution of the verifier, bb2 -> bb9
    certlog/        shard logs, run record, replay script
    docs/           this file and the decision log

## The verifier lineage — each step was a diagnosis, not an addition of nodes
| file | what changed | measured effect |
|---|---|---|
| `bb2.cpp` | q=7 prototype: interval extension + mean-value + PSD-gated tangent | 40M nodes, INCOMPLETE |
| `bb8.cpp` | q=8 + separable span-1 root decomposition (1600 -> 203 roots) | 30M nodes, INCOMPLETE |
| `bb8c.cpp` | + per-node constraint-propagation contractor | still INCOMPLETE |
| `bb8d.cpp` | + instrumentation separating stalled boxes from the sweep | revealed the stall is not volume |
| `bb8e.cpp` | exact interval centre evaluation | correct but ~100x slower; abandoned |
| `bb8f.cpp` | second-order table interpolation with rigorous remainder | same speed, error 2.4e-4 -> ~1.5e-7 |
| `bb8g.cpp` | alpha-BB second-order bound replacing the PSD-gated tangent | **first PROVED**: s=1/2, 43.7M nodes, 198 s |
| `bb8h.cpp` | root sharding; box endpoints float -> double | removed a 1e-6 precision floor that faked unresolved cells |
| `bb9.cpp` | dual-resolution tables (coarse for ranges, 8x fine for points) + disk cache | all three certificates close |

## The three diagnoses that mattered
1. A table cell minimum carries error `h*|W'| ~ 2.4e-4`, five times the margin being
   certified. The bound could not certify at any node count. Fixed by second-order
   interpolation with a rigorous remainder, at the same table size and speed.
2. The Loewner lower Hessian from `min W''` over a box is almost never positive definite,
   so the tangent-plane test almost never fired. Replaced by an always-valid alpha-BB bound
   with a Gershgorin lower bound on the smallest eigenvalue.
3. `float` box endpoints impose a `~1e-6` floor on box width near coordinate values of 10.
   Boxes were being reported unresolved for a storage reason, not a mathematical one.

## Witness selection — proof cost, not decimals
The margin available to the certificates is capped by the 67.35% requirement:
`margin ~= (ceiling - 0.6735) / |dC/d(eps)|`, `|dC/d(eps)| ~ 0.6`. With the frozen q=8
ceiling `0.673522454` this gives `3.08e-5`, and a scan over slope subsets confirmed no
restructuring raises it (`sub2.py`). Two further attempts to raise the ceiling failed
(`opt9.py`: +2.8e-7; `opt9f.py`, freeing the window frequencies: +1.4e-7), so the q=8
configuration is at a genuine local optimum.

Dropping `s = 21/20` costs only `1.5e-7` of margin and removes an entire certificate, so the
delivered witness uses three slopes. `eps_{1/2}` sits at its own allowance limit `4.725e-5`,
which is exactly the value already proved — the first certificate was not wasted work.

## What a paper needs, and where it is
* object, imported interface, spectral identity, matrix theorem with proof, tightness and
  the two non-removable slacks, local certificates, span accounting, finite inequality,
  averaging, exact evaluation of `R`, open items: `public_mirror/THEORY.md` §0-7
* methodology, seed families, evaluator-noise rule, every route probed: `NUMERICS.md`
* the bandwidth ladder and the thin-sampling inflation warning: `LADDER.md`
* the two refuted candidates with runnable witnesses: `REFUTATION.md`
* certificate record, per-shard node counts, replay: `CERTIFICATES.md`, `certlog/`
