# Where this sits against the published record, and what remains

Author: Libo Wang.

## The record, and the comparison

An independent, machine-checked ledger of unconditional lower bounds for the proportion of
nontrivial zeta zeros on the critical line is maintained at `riemannzeta.fun`, built on
Anthropic's 2026 article and its **Zeta23** Lean 4 formalization. Its state at the time of
writing:

| | |
|---|---|
| current kernel-verified record | `67.2500703679%`  (`kappa_0 = 2 - 1/cMT`) |
| verification | Lean kernel plus an independent nanoda replay |
| assumptions | **0** — the record does not assume the Riemann hypothesis |
| previous frontier | `41.6666%` (Pratt, Robles, Zaharescu, Zeindler, 2020; Levinson method) |

This repository's value, in exact rationals:

    kappa = 722547711262091300265625000000000 / 1072825050442925667061714615173641
          = 0.673500037087855885...
    kappa_0 = 672500703679 / 1000000000000
          = 0.672500703679

    kappa > kappa_0 :  strict, by +0.0999333409 percentage points.

**Compatibility of the counted quantity.** The ledger's `N_0*` counts *distinct* zeros on
`Re(s) = 1/2`; this repository bounds the count of zeros that are *simple and* on the
critical line. Simple implies distinct, so a lower bound for the latter is a lower bound for
the former. The comparison is therefore valid and conservative.

**Where the gain comes from.** The record constant is the optimum of the window functional
alone: `sup_{v >= 0 on [-1/2,1/2]} H(v) = 0.6725007...`, attained near `v(t) = cos(sqrt 2 t)`,
which is why `omega_0 = sqrt 2` appears in the window of this programme. The window used
here has `H = 0.672167187...`, deliberately **below** that optimum; the matrix/pressure
machinery of `THEORY.md` §2-§5 supplies a further `0.00133285`, and the net is `0.6735000...`.
The improvement is therefore not a better window — it is the machinery on top of the window.

## Status of the analytic interface D0

This changed materially this round. D0 was previously labelled an unaudited import with no
stated provenance. It is in fact the analytic layer of the published, **unconditional**,
Lean-formalized result that the ledger uses as its baseline. Two consequences:

1. The chain in this repository is **not conditional on RH**, provided the interface it
   consumes is the one that formalization establishes.
2. The remaining audit task is now bounded and concrete: check that the statement (A) of
   `THEORY.md` §1 — the matrix `A = P0 + Q0`, `n_+(Q0) <= b`, `s + 2b <= N'`, and
   `tr A = N + o(N)`, `||A||_F^2 = (2 - H(v)) N + o(N)` — is exactly what the Zeta23
   formalization provides, with the same normalisation of heights, gaps and multiplicities.
   That is a correspondence check against a public artefact, not an open analytic problem.

## What a submission would require

The ledger fixes the theorem; the only candidate-controlled value is the rational `kappa`:

    forall eps > 0, exists T0, forall T >= T0,
        (kappa - eps) * N(T, 2T)  <=  N_0*(T, 2T)

A submission must therefore be a Lean 4 proof of that statement for this `kappa`. Mapping the
present artefact onto it needs three things, in increasing size:

1. **The finite algebra.** `THEORY.md` §2, §3, §5 and §6, plus `AUDIT.md` §1-§3. All are
   finite, exact-rational or elementary-analytic statements. This is ordinary Lean work.
2. **The interface.** Import (A) from Zeta23 and discharge the correspondence check above.
3. **The three local certificates.** `sum_r b_r g_r + s sum a_ij W(y_j - y_i) >= eps_s` on
   `[0, inf)^8`. These are currently established by an exhaustive interval branch-and-bound
   outside Lean (48 shards, zero unresolved boxes, `CERTIFICATES.md`). Inside Lean they would
   have to become a verified computation — a reflective interval-arithmetic tactic replaying
   the same search, or a reformulation that removes the need for one. **This is the largest
   remaining item and it is not started.**

## Honest summary

The finite-dimensional mathematics is closed and independently reproducible. The analytic
interface is now identified, public and unconditional rather than unaudited. What is not
done is the Lean formalization — above all the certificates as a verified computation — and
until that exists this artefact is a mathematically complete argument outside the ledger's
verification pipeline, not an entry in it.
