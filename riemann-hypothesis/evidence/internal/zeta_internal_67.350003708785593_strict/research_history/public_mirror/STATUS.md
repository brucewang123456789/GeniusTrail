# Status ledger — read before quoting any number

Author: Libo Wang.

| number | status |
|---|---|
| `67.350003708785593%` | **this repository.** Matrix theorem proved for general `(d,r)`; `H`
  and window positivity certified in Arb; span identities, `R` and `C` exact over **Q**;
  **all three local certificates closed, 48/48 shards `PROVED`, zero unresolved boxes**. |
| `67.34574870891890%` | earlier `q = 6` line of the same programme; cited, not re-established here |
| `67.350352375073%`, `67.35006335392536%` | **refuted**, see `REFUTATION.md` |

The two steps that used to be imported — the finite spectral inequality (B) and the
pinching / shifted pressure accounting — are proved in `AUDIT.md`.

## What is closed

1. The matrix theorem for `d = 16`, `r = 9`, `T = 9/8` — proved here for general `(d, r)`.
2. `H > 672167187145431/10^15` and `min v > 0.7508666999` — Arb, precision 300, outward rounding.
3. Span capacities, `B = 93/23000`, the value of `R` as an exact minimum over the complete
   kink set, and the final fraction `C` — exact rational arithmetic, no floating point in any
   acceptance test.
4. The finite spectral inequality (B), the pinching bound and the pressure charge count —
   proved in `AUDIT.md`, with the scalar core checked exactly and the whole inequality
   stress-tested against 4000 random instances with zero violations.
5. The three eight-dimensional local inequalities — exhaustive interval branch-and-bound,
   48 shards, every shard fail-closed, zero unresolved boxes.

## What is open — three independent items

1. **Certificate arithmetic.** The verifier uses outward-rounded IEEE double together with
   rigorous second-order table remainders. Every operation widens outward and the search
   fails closed, but a submission-grade artefact should replay all 48 shards in directed
   MPFR or Arb. This is an engineering step, not a mathematical gap; nothing in the argument
   changes, only the width of the arithmetic.
2. **The analytic interface D0** (`THEORY.md` §1) is the analytic layer of the published,
   unconditional, Lean-formalized 2026 result that the `riemannzeta.fun` ledger uses as its
   baseline. It is therefore no longer an import of unknown provenance, and the chain here is
   **not conditional on the Riemann hypothesis**. What remains is a bounded correspondence
   check: that statement (A) is exactly what that formalization provides, in the same
   normalisation. See `SUBMISSION.md`.
3. **Lean formalization.** The ledger accepts only machine-checked proofs of a fixed theorem.
   Formalizing the finite algebra is ordinary work; formalizing the three local certificates
   as a verified computation is the largest remaining item and has not been started.

Items 1-3 are independent of each other and of the local certificates. Closing the local
certificates upgrades the finite-dimensional layer only. Acceptance by the mathematical
community is a matter of independent peer review and is not established by any computation
in this repository.