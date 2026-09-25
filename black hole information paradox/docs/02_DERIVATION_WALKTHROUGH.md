# Derivation Walkthrough

**Purpose.** The technical report states the results in final form. This document records how
they were reached: the order of reasoning, the questions that motivated each step, the routes
that were tried and abandoned, and the checks that caught errors. It is written so that a
reader can follow the derivation without first accepting the conclusions.

---

## Step 0 — The starting position and its weak point

The inherited position was two results about free hard spin-2 radiation at fixed graviton
number `N`:

- the established flux algebra has commutant exactly `W*(N_R)` (Proposition 1);
- one bundle-covariant spin-4 Stokes observable collapses that commutant (Proposition 2).

Proposition 2 came with a minimality statement of the form *"some `N_R`-changing observable is
necessary; one Stokes connector is sufficient."*

**The observation that started everything.** Necessary-plus-sufficient pairs of that shape are
usually shadows of a single iff criterion. If a proof only ever uses the fact that *some*
block is nonzero, then the proof does not know about adjacency, does not know about the
particular observable, and does not know how many blocks there are. All of that is free
information being discarded.

So the first question was not "what is the next physical layer?" but: **what is the most
general statement the existing proof already supports?**

---

## Step 1 — Extracting Theorem 1

Reading the Proposition 2 argument line by line, it uses exactly three facts:

1. `P_r` lies in the block algebra, so commutant elements are block diagonal;
2. `B(H_r)' = ℂ`, so the blocks are scalars;
3. compressing `[X,S] = 0` between two sectors gives `(c_{r'} − c_r)·P_{r'}SP_r = 0`.

Nothing else. In particular, nothing about helicity, nothing about `κ`, nothing about
adjacency, nothing about the index set being finite.

Fact 3 says: *a nonzero block is an edge, and an edge forces two coefficients to be equal.*
That is a graph statement. Once written that way the classification is forced — coefficients
are constant on connected components, and by the bicommutant theorem the algebra is the
direct sum over components.

**Check performed.** Certificate 1 tests the prediction `dim(commutant) = #components` against
a brute-force numerical commutant computation, across eleven configurations: adjacent chains,
single non-adjacent edges, two-component splits, crossing edges, empty graphs, and
multi-operator families where different operators supply different edges. All agree.

**What this immediately bought.** Corollary 1.1 fell out with no extra work: a connected graph
on `N+1` vertices needs `N` edges, so any completing family needs at least `N` nonzero blocks —
and the Stokes connector supplies exactly `N`. Proposition 2 was not just minimal in operator
count; it was optimal in block count. Certificate 2 confirms by exhaustive search over edge
sets that `0`, `1`, `2` blocks cannot complete at `N = 3` and that `3` can.

---

## Step 2 — Re-examining a layer that had been dismissed

The previous stopping point had explicitly declined the full Fock layer on the grounds that
"a number-changing operator plus fixed-`N` completeness gives standard CCR/Fock irreducibility".

Theorem 1 made it easy to test that claim, because Theorem 1 computes commutants rather than
asserting them. Applying it to the number grading with no added connector:

- all flux generators preserve total graviton number;
- the Stokes connector `dΓ(K_κ)` also preserves it;
- so the number graph has **no edges**, and the commutant is `ℓ^∞(ℕ) = W*(N̂)`.

That is not scalar. So the algebra is not irreducible, and the dismissal had conflated a
small physically distinguished algebra with the entire Weyl algebra. Stone–von Neumann is a
statement about the latter and says nothing about the former.

**Check performed.** Certificate 3 computes the commutant dimension in a truncation at
`N_max = 5` and returns `6 = N_max + 1`, exactly as predicted.

---

## Step 3 — Choosing the right connector, and why the obvious choice is worse

To close the number grading one needs an operator with nonzero blocks between adjacent
`N`-sectors. The two natural candidates are the Weyl unitary `e^{iΦ(f)}` and the field operator
`Φ(f) = a(f) + a†(f)` itself.

The Weyl unitary is bounded, which looks like an advantage. It is not. Its adjacent blocks
carry Hermite-type coefficients which can be made arbitrarily small and which can vanish at
isolated `(N, f)`. A proof routed through Weyl operators would therefore need a genericity
hypothesis.

The field operator needs none, because

```
    ‖a†(f)ψ‖² = ‖a(f)ψ‖² + ‖f‖²‖ψ‖² > 0
```

is an identity, not a generic statement. `a†(f)` is injective for every nonzero `f`, so every
adjacent block is nonzero for every `N`. The cost is unboundedness, which is handled by
Remark R3 of the technical report: commutant elements of the relevant form preserve the
finite-particle domain, so the block computation is legitimate there.

**Check performed.** Certificate 3 verifies the collapse for four choices of `f` including a
`10⁻⁶`-scale one, verifies that `f = 0` does *not* collapse (the control), measures the minimum
singular value of `a†` across adjacent blocks, and prints the Weyl adjacent-block magnitudes
for contrast.

**Choosing the harder-looking route because it removes a hypothesis was the right call.**
An unnecessary genericity assumption is exactly the kind of thing a referee finds first.

---

## Step 4 — Following the physics, and hitting the real wall

The BMS charge splits into a quadratic hard part and a *linear* soft part. Theorem 3 says the
missing connector must be linear in the field. So the grading structure and the hard/soft
split are the same decomposition seen twice.

This looked like a clean physical payoff until the obvious question: *is the physical soft
charge an admissible `f`?*

It is not. `Q_soft` is the `ω → 0` mode, so its smearing function is infrared
non-normalizable and lies outside `H₁`. Theorem 3 holds for every `f` in `H₁`, and the physical
soft charge sits at exactly the one boundary point where it fails.

**This was the moment the derivation stopped being bookkeeping.** The failure is not an
inconvenience to route around; it is a precise localization of where the asymptotic
information problem actually lives.

---

## Step 5 — The infrared layer, and a deliberate refusal

The natural next move was to assert that memory sectors are superselected and derive the
consequences. A literature check was run before doing so — and it showed the question is
actively contested. One line of work treats memory as a superselection label indexing
symplectic leaves; another argues explicitly that the vacuum is changed by soft quanta from
*any* finite-energy process, unlike genuine superselection sectors which no finite-energy
process can connect.

**Asserting either side would have made the whole result hostage to an unsettled premise.**

Theorem 1 offered a way out, because it classifies *any* graph, including the empty one. So
both branches were computed:

- if memory sectors are disjoint, the projections are central, the graph is empty, and no
  connector of the Stokes type can exist *inside* the algebra — a structural no-go, not a
  failure of ingenuity;
- if they are not, intertwiners exist, edges exist, and Theorem 1 classifies the outcome.

**Check performed.** Certificate 4 models three memory sectors as inequivalent `su(2)` irreps
of dimensions 2, 3, 4. The commutant comes out 3-dimensional and block-scalar to `10⁻¹⁵`,
confirming centrality. A contrast case with three *equivalent* copies returns a 9-dimensional
commutant containing off-diagonal intertwiners, confirming that inequivalence is what does
the work. The certificate also confirms `P_m ∈ π(𝔄)''` and that cross-sector expectation values
are phase-independent to `3×10⁻¹⁶`.

---

## Step 6 — Taking the strongest objection seriously

At this point the results formed a consistent classification, but every one of them was an
exact-zero statement. The sharpest available criticism was:

> Completeness in the von Neumann sense carries no physical content. At finite precision
> "nonzero" and "zero" are indistinguishable, and a theorem with no modulus says nothing,
> especially as `N` grows.

That objection is correct. Rather than defend against it, the criterion was rebuilt with a
modulus.

The route: the blocks of `[X,S]` are mutually Hilbert–Schmidt-orthogonal, so
`‖[X,S]‖²_HS` is *exactly* a weighted quadratic form — a graph Laplacian with weights
`w_{rr'} = ‖P_{r'}SP_r‖²_HS`. Then the standard variational bound gives the Fiedler value as the
modulus, with equality on the Fiedler eigenvector. Connectivity is recovered as `λ₂ > 0`.

A second, dimension-free route was added because Hilbert–Schmidt norms diverge in the
continuum: using operator norms and summing reciprocal edge weights along the cheapest path
gives a bound in terms of the resistance diameter, with no finiteness assumption anywhere.

**Check performed.** Certificate 5(E) verifies the bound and its exact saturation on random
multi-mode block families carrying no special structure — the Fiedler ratio comes out
`1.0000000000`.

---

## Step 7 — The computation that was not expected

With the modulus defined, the obvious question was how `λ₂` behaves for the actual Stokes
connector as `N` grows. The expectation was bad news: a path graph on `N+1` vertices with
uniform weights has `λ₂ ~ π²/N²`, which would mean completeness degrades as `N⁻²` and the whole
programme is physically weak at large graviton number.

The numerical scan returned `λ₂ = 2.000000` at every `N` tested.

That is not the kind of number one accepts without an explanation, so the next step was to
find the structure responsible. The full spectrum turned out to be `λ²·k(k+1)` — triangular in
`k`, which is the signature of an angular-momentum Casimir. That identified the mechanism:

- on the two-mode subspace, `𝒫_κ = λ(a_R†a_L + a_L†a_R) = 2λJ_x` in Schwinger bosons;
- at fixed `N` this is the spin-`N/2` irrep;
- the commutant of Proposition 1 is exactly the functions of `J_z`;
- the adjoint Casimir acts on the spin-`k` component of the adjoint decomposition by `k(k+1)`;
- for `J_z`-commuting `X` the residual `x`/`y` symmetry halves it.

So the coherence-transfer Laplacian *is* half the `su(2)` adjoint Casimir, and the spectrum is
forced. The weights `(r+1)(N−r)` grow at exactly the rate needed to cancel the `N⁻²` path
suppression, and the cancellation is not a coincidence — it is the representation theory.

**Independent corroboration.** The operator-norm route was computed separately: the resistance
diameter is `Σ 1/(λ√((r+1)(N−r)))`, a convergent sum tending to `π/λ`. Two different norms,
two different arguments, same conclusion — uniform in `N`.

**And a closure that was not planned.** The Fiedler eigenvector came out `c_r = r − N/2`, i.e.
proportional to `N_R`. The operator hardest for the connector to see is exactly the one
Proposition 1 had identified as the missing information.

---

## Step 8 — Returning to the contested question with a sharper tool

With a modulus in hand, the memory dichotomy could be revisited. At any finite infrared
regulator `μ`, cross-sector overlaps are nonzero and vanish as a power `μ^α`. So the weights
are positive, the gap is positive, and completeness holds — with a modulus diverging as
`μ^{−α}`. As `μ → 0` the gap closes and superselection is recovered.

**This dissolves the controversy without deciding it.** One camp is describing the finite-`μ`
statement; the other is describing the limit. Both are right. They differ in the order of
limits, and the spectral gap is the quantity neither had named.

The exponent `α` is imported from the dressing literature and is *not* derived here. That
limitation is stated in the technical report and in the scope guard.

---

## Routes tried and abandoned

**Asserting memory superselection and proceeding.** Abandoned at Step 5 after the literature
check showed the premise is contested. Delivering both branches is strictly better: the result
does not decay if the premise flips.

**Using Weyl unitaries for the Fock connector.** Abandoned at Step 3 because it requires a
genericity hypothesis that the field operator does not.

**Claiming Theorem 1 as new mathematics.** Abandoned on inspection. In finite dimensions it is
close to folklore — block-algebra structure theory and inclusion-graph connectedness. The
technical report concedes this in §3.8 rather than waiting to be corrected.

**Attempting an interacting or evaporating extension.** Not attempted. Nothing in this toolkit
bears on dynamics, and no amount of operator algebra will supply the missing physical input.

---

## How the checks were designed

Every certificate is built to be able to fail:

- **Predictions, not fits.** Certificate 1 compares a brute-force null-space computation against
  a combinatorial prediction made in advance.
- **Controls.** Certificate 3 includes `f = 0`, which must *not* collapse the commutant.
- **Contrasts.** Certificate 4 includes equivalent representations, which must produce
  intertwiners and a larger commutant.
- **Exhaustive search where feasible.** Certificate 2 enumerates all edge subsets rather than
  sampling.
- **Sharpness, not just validity.** Certificate 5(E) checks that the Fiedler eigenvector
  saturates the bound, not merely that the bound holds.
- **A scope guard.** Certificate 6 fails the build if any claim label is weakened, making
  overclaiming a machine-detectable error rather than an editorial slip.
