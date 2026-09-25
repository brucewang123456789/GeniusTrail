# Sector Connectivity and the Quantitative Modulus of Asymptotic Graviton Tomography

**Author:** Libo Wang · **Date:** 14 September 2026 · **License:** CC BY 4.0

> **Scope, stated first.** All results concern **free asymptotic radiative data** at future
> null infinity. There is no interaction, no dynamics and no evaporation model anywhere in
> this work. **The black-hole information paradox is not solved**, and nothing here should be
> read as if it were.

---

## What this is

Asymptotic tomography asks which quantum states of outgoing radiation can be reconstructed
from observables available at future null infinity. Two results were established earlier in
this line for free hard spin-2 radiation at fixed graviton number `N`:

- the established gravitational flux algebra has commutant exactly `W*(N_R)`;
- adjoining one bundle-covariant spin-4 Stokes coherence observable collapses it to scalars.

This repository establishes nine further results, in two parts.

**Part II — exact classification.** A single graph-theoretic criterion determines when a
sector-block algebra together with *any* added observable family is informationally complete,
computing the commutant exactly as the algebra of functions on the connected components of a
connectivity graph. The two established results are recovered as the empty-graph and
path-graph cases. Consequences: a resource bound showing the Stokes connector is
edge-optimal; the full Fock commutant computed exactly as `W*(N̂)`, which is *not* the
Stone–von Neumann statement; identification of the missing connector type as the soft BMS
charge, together with the precise point at which that identification fails; and a closed
dichotomy for the infrared memory layer that does not require adjudicating the contested
question of whether memory is superselected.

**Part III — quantitative completeness.** Every exact-zero criterion is replaced by a sharp
metric one. The modulus of completeness is the algebraic connectivity of a
*coherence-transfer Laplacian*, with equality on the Fiedler eigenvector. For the Stokes
connector this Laplacian is computed in closed form: the connector generates a hidden `su(2)`
whose adjoint Casimir gives spectrum `λ²k(k+1)`, so the spectral gap is exactly `2λ²`,
**independent of the graviton number**, and the least visible operator is exactly `N_R`. The
same gap turns the memory dichotomy into a continuous rate.

## The three claims worth defending

> The modulus of informational completeness is the algebraic connectivity of the
> coherence-transfer Laplacian, sharp on the Fiedler eigenvector.

> For the spin-4 Stokes connector that Laplacian is half an `su(2)` adjoint Casimir, so the
> spectral gap is exactly `2λ²` and does not degrade with graviton number.

> The two opposing positions on infrared memory superselection differ by an order of limits,
> and the spectral gap is the quantity that interpolates between them.

## Repository layout

```
docs/
  01_TECHNICAL_REPORT.md        complete statements and proofs
  02_DERIVATION_WALKTHROUGH.md  how the results were reached, including abandoned routes
  03_SCOPE_AND_LIMITATIONS.md   every limitation, and where to attack the work
  04_NOVELTY_AND_PRIOR_WORK.md  conservative novelty assessment with collision-risk table
  05_REPRODUCIBILITY.md         how to run and how to falsify
prompts/
  01_CORE_KNOWLEDGE_PROMPT.md      self-contained context-window form of the full result
  02_RESEARCH_FRONTIER_PROMPT.md   method, open problems, attack surface, output standards
  03_SCOPE_CALIBRATION_PROMPT.md   short scope-correctness reference
certificates/
  cert_01..06 + run_all.py      executable numerical certificates
results/
  theorem_status.json           machine-readable scope labels, enforced by cert_06
  run_log.txt                   reference transcript
```

## Reproduce

```bash
pip install -r certificates/requirements.txt
cd certificates && python3 run_all.py      # expects: SUITE: 6/6 PASS
```

Runtime under one minute. `certificates/cert_06_scope_guard.py` fails the build if any claim
label in `results/theorem_status.json` is weakened, making overclaiming a machine-detectable
error rather than an editorial slip.

## Results at a glance

| # | Result | Status |
|---|---|---|
| T1 | Sector Connectivity Theorem: commutant `= ℓ^∞(components)`; complete iff connected | Proved; novelty **not claimed** |
| T1.1 | `≥ N` off-diagonal blocks required; Stokes connector supplies exactly `N` | Proved |
| T2 | Fock commutant `= W*(N̂)`, not scalar — hence not Stone–von Neumann | Proved |
| T3 | One linear observable completes, no genericity hypothesis; failure locus `f_g ∉ H₁` | Proved |
| T4 | Memory dichotomy, both branches; Branch 1 gives a structural no-go | Proved as dichotomy |
| T5 | Commutant abelian; residue is classical superselection, not quantum degeneracy | Proved, conditional |
| T6 | Completeness modulus `= λ₂`, sharp on the Fiedler eigenvector | Proved |
| T7 | Dimension-free bound via resistance diameter | Proved |
| T8 | `spec(L) = λ²k(k+1)`; gap `= 2λ²` independent of `N`; extremal operator `= N_R` | Proved |
| T9 | Memory controversy is an order-of-limits distinction | Structure proved |

## What is not claimed

- The black-hole information paradox is **not** solved.
- No dynamics, interaction or evaporation is treated.
- Theorem 1 is **not** claimed as new mathematics; in finite dimensions it is near-folklore.
- Hypothesis (M1) on memory superselection is **not** adjudicated.
- The exponent `α` of Theorem 9 is imported from the literature, not derived.
- **No world priority is claimed.** Targeted searches found no direct prior statement of the
  combined results, which is not evidence of priority.

See `docs/03_SCOPE_AND_LIMITATIONS.md` for the complete list, including a ranked guide to the
load-bearing assumptions for anyone wishing to refute the work.

## Citing

See `CITATION.cff`.

## Acknowledgements

Parts of the derivation and all numerical verification were developed with AI assistance. All
theorem statements, scope decisions and claim calibrations are the author's responsibility.
