# A verified lower bound of 67.35% for the proportion of simple critical-line zeros

**Author: Libo Wang**

> All three eight-dimensional local certificates required by this witness are **closed**:
> 48 of 48 shards return `PROVED`, with zero unresolved boxes. What remains open is stated
> precisely in `STATUS.md` and is *not* hidden: the analytic interface D0 and the
> block-averaging step and the finite spectral inequality are now **proved** (`AUDIT.md`), and
> the analytic interface is identified as the public, unconditional, Lean-formalized 2026
> baseline rather than an unaudited import (`SUBMISSION.md`). What is *not* done is the Lean
> formalization required by the record ledger, above all the certificates as a verified
> computation. Read `STATUS.md` before quoting the number.

| | |
|---|---|
| value `C` | `722547711262091300265625000000000/1072825050442925667061714615173641` |
| decimal | `0.673500037087855885076` |
| percent | **67.350003708785593%** |
| witness | `q = 8` (`d = 16`, `r = 9`, `T = 9/8`), `m = 531`, `eta = 1/1` |
| window constant | `H > 672167187145431/1000000000000000`, certified in Arb at precision 300 |
| window positivity | `min v > 0.7508666999` on `[-1/2,1/2]`, certified in Arb |
| local certificates | 3 of 3 closed, 48/48 shards `PROVED`, 0 unresolved |
| previous rigorous record of this programme (`q = 6`) | `67.34574870891890%` |
| improvement | `+0.00425500` percentage points |

## The three local certificates

Each states: for all gap vectors `g in [0,inf)^8`,

    sum_r b_r g_r + s * sum_{0<=i<j<=8} a_ij W(y_j - y_i)  >=  eps_s,
    y_0 = 0,  y_j = g_1 + ... + g_j,  W(x) = (K(x)/K(0))^2.

| `s` | `eps_s` | hardened float minimum | margin | shards | nodes |
|---|---|---|---|---|---|
| `1/2` | `526/78125` | `0.0067800531` | `4.725e-05` | 16/16 PROVED | 43,685,299 (single run) |
| `19/20` | `9879/1250000` | `0.0079338933` | `3.069e-05` | 16/16 PROVED | 170,801,234 |
| `1` | `20033/2500000` | `0.0080438403` | `3.064e-05` | 16/16 PROVED | 213,519,829 |

A shard reports `PROVED` only when its stack empties with zero unresolved boxes and zero
terminal cells. Every bound is evaluated with outward rounding; a node-cap hit, a box that
cannot be split further, or an unresolved terminal cell all yield `INCOMPLETE`. The search
is therefore fail-closed with respect to its own arithmetic.

## How the certificates were made to close

Three things were needed; each was found by diagnosing a stall rather than by adding nodes.

1. **Second-order table interpolation with a rigorous remainder.** Reading a cell minimum
   from a table of step `h` costs `h*|W'| ~ 2.4e-4` of accuracy, which is *five times larger
   than the margin being certified*. No amount of branching can overcome that. Storing the
   cell-midpoint value and derivative together with a bound on `|W''|` and using
   `W(d) >= W(mid) + W'(mid)*delta - (1/2)|W''|max*delta^2` reduces the error to `~1e-7` at
   the same table size and the same speed.
2. **An alpha-BB second-order bound instead of a tangent plane gated on positive
   definiteness.** The Loewner lower Hessian assembled from `min W''` over a box is almost
   never positive definite, so the tangent test almost never fired. The bound
   `F(c) - sum_k |dF/dg_k(c)| r_k + (1/2) min(lambda_min,0) sum_k r_k^2`, with `lambda_min`
   bounded below by Gershgorin, is always valid and dispatches whole basins.
3. **Double-precision box storage.** With `float` endpoints, boxes near coordinate values
   of 10 cannot be narrowed below `~1e-6`; the search silently hit a precision floor and
   reported unresolved cells that were artefacts of storage, not of mathematics.

Two further reductions cut the domain before branching: a separable span-1 bound
`F(g) >= sum_r phi_r(g_r)` with `phi_r(g) = b_r g + s a_{r-1,r} W(g)`, which lifts every
coordinate off zero and cuts box volume by a factor of about `2e-3` while splitting the root
into 1024-4096 natural components, and a per-node constraint-propagation contractor.

## Reproduce

```
pip install -r requirements.txt
python3 verify_exact.py          # exact rational chain: span identities, B, R, C
python3 arb_window.py            # Arb: lower bound for H, positivity of the window
python3 refute.py                # re-evaluates a refutation witness
sh certificates/run_all.sh       # replays all 48 shards, fails closed
```

## Contents

| file | content |
|---|---|
| `THEORY.md` | the full finite-dimensional reduction, every step labelled |
| `AUDIT.md` | proofs of the two steps that used to be imported, plus the stress test |
| `SUBMISSION.md` | comparison against the published record and what a Lean submission needs |
| `lean/` | the submission contract as read from the repositories, the specification skeleton, and why porting the branch-and-bound proof to Lean is a dead end |
| `STATUS.md` | what may and may not be claimed |
| `CERTIFICATES.md` -> `certificates/` | shard record, logs, replay script |
| `NUMERICS.md`, `LADDER.md` | how the configuration was found; the bandwidth ladder |
| `REFUTATION.md`, `refute.py` | two earlier candidates that are false, with witnesses |
| `verifier/bb9.cpp` | the interval branch-and-bound verifier |
| `config/` | frozen configuration and witness, exact rationals |

## Licence

MIT. Author: Libo Wang. Released for scrutiny; if you can break a step, please do.