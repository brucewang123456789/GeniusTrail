# Audit of the two previously imported steps

Author: Libo Wang. This file records what was checked in this round, and how.

Until now two steps in `THEORY.md` carried the label **IMPORTED**: the finite spectral
inequality (B) that turns the Weil-form matrix into `Delta(M)`, and the pinching / shifted
pressure accounting that turns a per-block inequality into a bound on `Delta(M)`. Both are
finite statements, so both can be settled here. They are now settled.

## 1. The finite spectral inequality (B)

    ||P0 + Q0||_F^2  >=  4 tr(P0 + Q0) - 3 s - 4 b + Delta(M),
    P0 = V V^*,  M = V^* V,  s columns of norm at most 1,  n_+(Q0) <= b.

**Scalar core, verified exactly.** The reduction rests on

    min_{n >= 0} [ (p - n)^2 + 4 n ]  =  2 p - 1 + Psi(p).

For `p < 2` the minimiser is `n = 0` and the left side is `p^2`; the right side is
`2p - 1 + (p-1)^2 = p^2`. For `p >= 2` the minimiser is `n = p - 2` and the left side is
`4p - 4`; the right side is `2p - 1 + (2p - 3) = 4p - 4`. The identity holds exactly on both
branches, with no case left over.

**Whole inequality, stress-tested.** 4000 random instances with `n` in 2..8, `s` in 1..6,
`b` in 0..3, `V` with column norms drawn in (0,1], and `Q0` real symmetric with exactly the
allowed number of positive eigenvalues: **0 violations**, minimum slack 1.59. Script:
`audit/auditB.py`.

## 2. Pinching

`Delta(M) = tr Psi(M) >= sum_B tr Psi(M_B)` for a partition of the index set into principal
blocks. `Psi` is continuous and `C^1` at `t = 2` — both `(t-1)^2` and `2t-3` give value 1 and
slope 2 there — and its second derivative is `2` then `0`, so `Psi` is **convex**. The
eigenvalues of the block-diagonal pinching `C(M)` are majorised by those of `M`, and for a
convex `f` majorisation gives `tr f(C(M)) <= tr f(M)`. Since
`tr f(C(M)) = sum_B tr f(M_B)`, the claim follows. Operator convexity is not needed.

Discarding the incomplete leftover block is legitimate because `Psi >= 0`: its minimum is
`0` at `t = 1`, and on `[2, inf)` it is `2t - 3 >= 1`.

## 3. Shifted pressure accounting

Claim: summed over the `m` offsets, a fixed gap receives total pressure charge at most
`(m - q) B`, `B = sum_r b_r`.

Fix the gap and its position `r` inside a local window. That determines the window's start.
A window of `q + 1` points lies entirely inside a complete block of the offset partition for
exactly `m - q` of the `m` offsets; for the remaining `q` offsets it straddles a block
boundary and contributes nothing. Summing over `r` gives `(m - q) sum_r b_r = (m - q) B`.
Boundary omissions only reduce the charge because `b_r >= 0`. Dividing by `m` gives the
`eta B (m - q) / m` term, and the total gap extent is at most `N + o(N)`.

## 4. What this leaves

`THEORY.md` §5 (5.5) is now proved rather than imported, and the step from the Weil-form
matrix to `Delta(M)` is proved rather than imported. The single remaining analytic
dependency is the construction and the estimates (A) of `THEORY.md` §1 — see `SUBMISSION.md`
for its current status, which changed materially this round.
