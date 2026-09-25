# Sector Connectivity and the Quantitative Modulus of Asymptotic Graviton Tomography

**Author:** Libo Wang
**Date:** 14 September 2026
**Status:** Technical report. Every theorem below carries an explicit scope. The black-hole
information paradox is **not** solved by this work, and no statement here should be read as
if it were.

---

## Abstract

Asymptotic tomography asks which quantum states of outgoing radiation can be reconstructed
from observables available at future null infinity. Earlier work in this line established
two results for free hard spin-2 radiation at fixed graviton number `N`: the established
gravitational energy, superrotation and helicity flux algebra has commutant exactly
`W*(N_R)` (Proposition 1), and adjoining one globally bundle-covariant spin-4 Stokes
coherence observable collapses that commutant to scalars (Proposition 2).

This report establishes nine further results. Part II proves an exact graph-theoretic
classification — the **Sector Connectivity Theorem** — of when a sector-block algebra
together with an arbitrary added observable family is informationally complete, computing
the commutant exactly as the algebra of functions on the connected components of a
connectivity graph. Propositions 1 and 2 are recovered as the empty-graph and path-graph
cases. The classification yields a resource lower bound showing the Stokes connector is
edge-optimal; an exact computation of the full Fock commutant as `W*(N̂)`, which is not the
Stone–von Neumann statement; an identification of the missing connector type as the soft
BMS charge together with the precise point at which that identification fails; and a closed
dichotomy for the infrared memory layer that does not require adjudicating the contested
question of whether memory is superselected.

Part III replaces every exact-zero criterion by a sharp metric one. The modulus of
informational completeness is shown to be the algebraic connectivity of a **coherence-transfer
Laplacian**, with equality on the Fiedler eigenvector. For the Stokes connector this
Laplacian is computed in closed form: the connector generates a hidden `su(2)` action whose
adjoint Casimir gives spectrum `λ²k(k+1)`, so the spectral gap is exactly `2λ²`, **independent
of the graviton number**, and the least visible operator is exactly `N_R`. The same gap turns
the memory dichotomy into a continuous rate, showing that the two opposing positions in the
current literature differ by an order of limits rather than by a fact.

All results are accompanied by executable numerical certificates.

---

## Contents

- §1 Setting and conventions
- §2 Established input (Propositions 1 and 2)
- §3 Theorem 1 — Sector Connectivity, with Corollaries 1.1 and 1.2
- §4 Theorems 2 and 3 — the Fock layer
- §5 Theorems 4 and 5 — the memory layer
- §6 Theorems 6 and 7 — quantitative completeness
- §7 Theorem 8 — the exact `su(2)` gap
- §8 Theorem 9 — the infrared rate
- §9 Consolidated statement of results
- §10 What is not proved
- §11 References

---

## 1. Setting and conventions

### 1.1 The physical arena

Work in asymptotically flat spacetime at future null infinity `𝓘⁺`, with free hard spin-2
radiative data. The one-graviton Hilbert space is the `L²` space of sections of the
two-helicity bundle over the positive massless momentum cone `𝒞 = (0,∞)_ω × S²`:

```
    H₁ = L²(𝒞, L_R) ⊕ L²(𝒞, L_L),        H_N = Sym^N H₁,        F = ⊕_{N≥0} H_N.
```

Fixed-`N` sectors decompose by right-helicity count:

```
    H_N = ⊕_{r=0}^{N} H_{r,N-r},          r = N_R,   N-r = N_L.            (1.1)
```

`P_r` denotes the orthogonal projection onto `H_{r,N-r}`; `P_N` the projection onto `H_N`
inside `F`.

### 1.2 The recovery criterion

Throughout, "informationally complete on `H`" means, for a set `A ⊂ B(H)`,

```
    A' = ℂ·I        ⟺ (bicommutant)        W*(A) = B(H).                   (1.2)
```

This is the operative definition. Every claim below is a commutant computation, never a
qualitative assertion that information is "at infinity".

### 1.3 Standing scope

**Free fields only.** There is no interaction, no dynamics, no evaporation model and no
map from initial data to outgoing data anywhere in this report. Every theorem is a statement
about the observable algebra of free asymptotic radiative data.

---

## 2. Established input

The following two results are taken as given. They are the starting point, not part of the
new content.

### Proposition 1 (fixed-`N` flux commutant)

Let `𝔄_flux` be the von Neumann algebra generated on `H_N` by the established gravitational
null-infinity flux operators: time-independent supertranslation/energy fluxes `𝒯_g`, smooth
superrotation flows `U_Y`, and time-independent helicity/superduality fluxes `𝒪_h`. Then

```
    (𝔄_flux)' = W*(N_R),        equivalently        𝔄_flux = ⊕_{r=0}^{N} B(H_{r,N-r}).   (2.1)
```

The mechanism: energy and helicity flux spectra jointly determine the colored atomic
configuration `{(ω_i, Ω_i, σ_i)}`, giving the multiplication algebra `L^∞(X_N)`; smooth
superrotations act transitively on the tangent directions of each fixed-`r` component; and
every generator commutes with total helicity `𝒪_1 = N_R - N_L = 2N_R - N`.

### Proposition 2 (minimal Stokes completion)

Same-direction opposite-helicity graviton coherence carries spin weight four, so `a_R† a_L`
is not a scalar. Let `κ` be any nonzero smooth section of `Hom(L_L, L_R)`, and set

```
    K_κ = [[0, κ], [κ†, 0]],        𝒫_κ = dΓ_N(K_κ),        ‖𝒫_κ‖ ≤ N‖K_κ‖_∞.      (2.2)
```

`𝒫_κ` is bounded and self-adjoint on all of `H_N` with no domain assumption, and

```
    ⟨r+1, N-r-1 | 𝒫_κ | r, N-r⟩ = λ √((r+1)(N-r)) ≠ 0,        λ = ‖κφ_L‖ > 0.      (2.3)
```

Consequently `W*(𝔄_flux, 𝒫_κ) = B(H_N)`.

---

# PART II — EXACT CLASSIFICATION

## 3. Theorem 1 — the Sector Connectivity Theorem

### 3.1 Setting

Let `I` be a countable index set, `H = ⊕_{r∈I} H_r` with every `H_r ≠ 0`, and

```
    𝔄₀ = ⊕_{r∈I} B(H_r)        (ℓ^∞-direct sum, a von Neumann algebra),
    P_r ∈ 𝔄₀,        𝔄₀' = W*({P_r}) = ℓ^∞(I).
```

This is exactly the output of Proposition 1 with `I = {0,…,N}`.

### 3.2 The connectivity graph

Let `S` be any family of operators on `H` — bounded, or self-adjoint with
`span{H_r} ⊆ D(S)` so that the blocks `P_{r'} S P_r` are defined. Define an undirected graph
`G(S)` on vertex set `I`:

```
    {r, r'} is an edge    ⟺    r ≠ r'  and  ∃ S ∈ S  with  P_{r'} S P_r ≠ 0.       (3.1)
```

For a connected component `C`, write `P_C = Σ_{r∈C} P_r` and `H_C = P_C H`.

### 3.3 Statement

> **Theorem 1.**
> ```
> (i)   W*(𝔄₀ ∪ S)'  =  { Σ_C c_C P_C : c ∈ ℓ^∞(comp G(S)) }  ≅  ℓ^∞(comp G(S))
> (ii)  W*(𝔄₀ ∪ S)   =  ⊕_{C ∈ comp G(S)} B(H_C)
> (iii) W*(𝔄₀ ∪ S)   =  B(H)      ⟺      G(S) is connected
> (iv)  |I| < ∞      ⟹      dim W*(𝔄₀ ∪ S)'  =  #comp(G(S)).
> ```

### 3.4 Proof

**Step 1 — reduce to block diagonal.** Let `X ∈ W*(𝔄₀ ∪ S)' = (𝔄₀ ∪ S)'`. Since `P_r ∈ 𝔄₀`,
`X` commutes with every `P_r`, hence `X = ⊕_r X_r` with `X_r ∈ B(H_r)`.

**Step 2 — reduce to scalars.** Each `X_r` commutes with all of `B(H_r)`, and `B(H_r)' = ℂ I_{H_r}`.
Therefore

```
    X = Σ_{r∈I} c_r P_r,        c ∈ ℓ^∞(I).                                        (3.2)
```

**Step 3 — edges force equality.** Fix `S ∈ S` and `r ≠ r'`. Compressing `[X,S] = 0`:

```
    0 = P_{r'} [X, S] P_r = (c_{r'} - c_r) · P_{r'} S P_r.                         (3.3)
```

If `{r,r'}` is an edge the block is nonzero, so `c_{r'} = c_r`. Equality propagates along
edges, so `c` is constant on every connected component.

**Step 4 — converse and bicommutant.** Conversely let `c` be constant on components and set
`X = Σ_C c_C P_C`. Then `X ∈ W*({P_r}) = 𝔄₀'`. For `S ∈ S`, decompose
`[X,S] = Σ_{r,r'} (c_{r'} - c_r) P_{r'} S P_r`. Each term vanishes: within a component because
`c_{r'} = c_r`, across components because `{r,r'}` is then not an edge so `P_{r'} S P_r = 0`.
Hence `X ∈ (𝔄₀ ∪ S)'`, proving (i). The algebra `ℓ^∞(comp G(S))` is generated by mutually
orthogonal projections summing to `I`, and its commutant is `⊕_C B(H_C)`; the bicommutant
theorem gives (ii). Parts (iii) and (iv) follow immediately. ∎

### 3.5 Remarks

**R1.** The condition involves generators only, since the commutant of a set equals the
commutant of the von Neumann algebra it generates. No closure argument is needed.

**R2.** Adjacency is never used. Any connected edge set suffices, including edge sets
containing no adjacent pair. Certificate 1 verifies configurations such as
`{(0,3),(1,2),(0,1)}`.

**R3.** Unbounded `S` is admissible when commutation is understood in the strong sense
(commutation with spectral projections). An `X` of the form (3.2) preserves `span{H_r}`, so
(3.3) is a valid computation on that domain. This is used in Theorem 3.

**R4.** The empty-edge case gives the maximal residual commutant `ℓ^∞(I)`. This is the memory
case of §5.

### 3.6 Corollary 1.1 — spanning-tree resource bound

> **Corollary 1.1.** Let `|I| = N+1 < ∞`. If `W*(𝔄₀ ∪ S) = B(H)` then `G(S)` is connected, hence
> has at least `N` edges. Therefore **any** completing family must exhibit at least `N` nonzero
> off-diagonal sector blocks.

The Stokes connector `𝒫_κ` of Proposition 2 produces exactly the `N` adjacent blocks (2.3)
and no others. It therefore **saturates the bound**: it is optimal not only in operator count
(one) but in block count (`N`). This upgrades the minimality statement of Proposition 2 from
a claim about the *type* of observable required to a claim about the *resource* required.

Certificate 2 exhibits, by exhaustive search over edge sets for `N = 3`, that `0`, `1` and `2`
blocks are impossible and `3` blocks are achievable.

### 3.7 Corollary 1.2 — Propositions 1 and 2 recovered

- `S = ∅`: no edges, `#comp = N+1`, commutant `ℓ^∞({0..N}) = W*(N_R)`. **Proposition 1.**
- `S = {𝒫_κ}`: path `0–1–⋯–N`, connected, commutant `ℂI`. **Proposition 2.**
- The necessity half of Proposition 2 is the case "every added observable commutes with `N_R`",
  which contributes no edges and leaves `N+1 ≥ 2` components.

### 3.8 Honest calibration

Theorem 1 is elementary. The proof is four short moves and a specialist will see it
immediately; in the finite-dimensional setting it is close to folklore, being the standard
structure theory of block-diagonal von Neumann algebras and the connectedness of an
inclusion graph. **No claim of mathematical novelty is made for it.** Its value is that it is
exact, that it subsumes Propositions 1 and 2 and Corollary 1.1 under one criterion, and that
it applies without modification to the Fock and infrared layers treated below, where the
previously available statements were partial.

---

## 4. Theorems 2 and 3 — the Fock layer

### 4.1 Theorem 2 — exact Fock commutant

Let `𝔄_F = W*({flux generators on F} ∪ {𝒫_κ = dΓ(K_κ)})` acting on `F = ⊕_N H_N`.

> **Theorem 2.**
> ```
>     𝔄_F' = W*(N̂),        equivalently        𝔄_F = ⊕_{N≥0} B(H_N).                (4.1)
> ```

**Proof.** The joint spectral analysis underlying Proposition 1 yields the multiplication
algebra `L^∞(X)` on the configuration space `X = ⨆_{N≥0} X_N`; the indicator `1_{X_N}` is `P_N`,
so `P_N ∈ 𝔄_F`. Proposition 2 gives `P_N 𝔄_F P_N = B(H_N)`. Every generator of `𝔄_F` preserves
total graviton number — the fluxes are number-preserving quadratic operators, and
`𝒫_κ = dΓ(K_κ)` is number-preserving by construction — so `G` has no edges. Theorem 1(i) gives
`𝔄_F' = ℓ^∞(ℕ) = W*(N̂)`. ∎

### 4.2 Why Theorem 2 is not the Stone–von Neumann statement

Stone–von Neumann and Fock irreducibility assert that the **entire** Weyl algebra `W(H₁)` acts
irreducibly on `F`. Theorem 2 concerns a strictly smaller, physically distinguished
generating set: the quadratic hard flux algebra plus one Stokes connector. Its commutant is
**not scalar**. Certificate 3 exhibits `dim 𝔄_F' = 6` in a truncation at `N_max = 5`, matching
`N_max + 1` exactly. Standard irreducibility says nothing about this algebra, and conflating
the two would be a category error.

### 4.3 Theorem 3 — completion by one linear observable

For `f ∈ H₁`, `f ≠ 0`, let `Φ(f) = a(f) + a†(f)`, essentially self-adjoint on the
finite-particle domain.

> **Theorem 3.** `W*(𝔄_F, Φ(f)) = B(F)` for every nonzero `f`.

**Proof.** `a†(f)` is injective on `F`:

```
    ‖a†(f)ψ‖² = ⟨ψ, a(f)a†(f)ψ⟩ = ⟨ψ, (a†(f)a(f) + ‖f‖²)ψ⟩ = ‖a(f)ψ‖² + ‖f‖²‖ψ‖² > 0   (4.2)
```

for `ψ ≠ 0`. Hence `P_{N+1} Φ(f) P_N = P_{N+1} a†(f) P_N ≠ 0` for every `N ≥ 0`, so `G` contains
the full ray `0–1–2–⋯`, which is connected. Apply Theorem 1(iii) with Remark R3. ∎

**Why `Φ(f)` and not the Weyl unitary.** Identity (4.2) is unconditional: no nonzero `f` can
fail. By contrast the adjacent blocks of `e^{iΦ(f)}` carry Hermite-type coefficients that can
be made arbitrarily small and can vanish at isolated `(N, f)`; a proof routed through Weyl
operators would require a genericity hypothesis. Certificate 3 records both the identity and
the contrast. Removing an unnecessary hypothesis is the point.

### 4.4 Physical identification, and the precise point where it fails

The BMS supertranslation charge at `𝓘⁺` splits as `Q[g] = Q_hard[g] + Q_soft[g]`, with `Q_hard`
quadratic in the radiative data and `Q_soft` **linear** in it. Comparing with Theorem 3:

```
    number-diagonal generators      ⟷    hard charges / fluxes
    number-off-diagonal connector   ⟷    soft charge                              (4.3)
```

So the residual obstruction of the hard flux algebra at the Fock level is exactly the degree
of freedom removed by the hard/soft split. This is a structural correspondence between an
operator-algebraic grading and a physical decomposition; it is not a new construction of
`Q_soft`.

**The failure locus.** `Q_soft[g]` is the `ω → 0` mode of the news, so its smearing function
`f_g` is **not** in `H₁`: it carries the standard infrared non-normalizability. Therefore

> Theorem 3 holds for every `f ∈ H₁`, and the physical soft charge sits precisely at the single
> boundary point where Theorem 3 fails.

This is not a technicality to be routed around. It names where the asymptotic information
problem actually resides in this framework, and it hands the question to §5.

---

## 5. Theorems 4 and 5 — the memory layer

Let `M` be the set of supertranslation memory values, `𝔄` the asymptotic radiative algebra,
`π_m` the representation labelled by `m ∈ M`, and `π = ⊕_{m∈M} π_m` (or the corresponding
direct integral).

### 5.1 The contested hypothesis, stated and not assumed

> **(M1)** For `m ≠ m'`, `π_m` and `π_{m'}` are factorial and mutually unitarily inequivalent
> representations of `𝔄`.

**The status of (M1) in the current literature is contested, and this report does not
adjudicate it.**

- **Supporting:** memory and electric flux treated as superselection labels indexing symplectic
  leaves [7]; superselection sectors in asymptotic quantization [9]; infrared-finite BMS
  scattering representations [8].
- **Opposing:** the argument that the vacuum is changed by soft quanta emitted in *any*
  finite-energy process, unlike ordinary superselection sectors which no finite-energy process
  can connect [10].

### 5.2 Lemma 4.1

Two factor representations are either quasi-equivalent or disjoint. Hence factorial together
with unitarily inequivalent implies **disjoint**: zero intertwiner space.

### 5.3 Theorem 4 — dichotomy

> **Theorem 4 (Branch 1: (M1) holds).** By Lemma 4.1, `Hom(π_m, π_{m'}) = 0` for `m ≠ m'`, so
> ```
>     π(𝔄)'  = ⊕_m π_m(𝔄)' = ⊕_m ℂ,        π(𝔄)'' = ⊕_m B(H_m).                   (5.1)
> ```
> The memory projections `P_m` are **central**, and every `A ∈ π(𝔄)''` satisfies `P_{m'} A P_m = 0`
> for `m ≠ m'`. In the language of Theorem 1, the memory graph is the **empty graph**, and
> Theorem 1(i) returns the maximal commutant `ℓ^∞(M)`.

> **Corollary 4.2 (structural no-go).** No connector of the Proposition 2 type exists *inside*
> the algebra. The impossibility is structural, not a failure of ingenuity. Any completion
> must enlarge the algebra beyond `π(𝔄)''` — for instance by memory-shifting dressing
> unitaries, which are automorphisms of `𝔄` but not elements of `π(𝔄)''`.

> **Theorem 4 (Branch 2: (M1) fails).** Nonzero intertwiners exist, so `G` has edges and
> Theorem 1(ii) returns `⊕_C B(H_C)` over the connected components of the memory graph. If that
> graph is connected, asymptotic tomography is complete across memory.

### 5.4 Why a dichotomy is the correct deliverable

Both branches are classified by the same criterion. The contested physics question is
thereby reduced to one sharp, checkable mathematical question:

```
    Is the memory connectivity graph connected, or empty?
```

Neither branch is assumed, and the result does not decay if the premise flips. §8 goes
further and shows the two branches are the endpoints of a single continuous quantity.

### 5.5 Theorem 5 — localization of the residue

Assume the sector-wise structure established above, so that `π_m(𝔄)'' = B(H_m)` by Theorems 1–3.

> **Theorem 5.**
> ```
> (i)   π(𝔄)' is ABELIAN, equal to ℓ^∞(comp G_memory).
> (ii)  Equivalently, π is MULTIPLICITY-FREE.
> (iii) The component projections P_C lie IN π(𝔄)''; the sector label is MEASURABLE.
> (iv)  Every state normal with respect to a single component is completely reconstructible.
> (v)   The unique irrecoverable datum is relative phase BETWEEN distinct components.
> ```

**Proof.** (i) is Theorem 1(i). (ii) is the standard equivalence: a representation is
multiplicity-free iff its commutant is abelian. (iii): `π(𝔄)'' = (ℓ^∞)' = ⊕_C B(H_C)` contains
each `P_C`. (iv): within a component the algebra is all of `B(H_C)`, which separates normal
states. (v): for `ψ_α = (v₁ + e^{iα}v₂)/√2` with `v₁, v₂` in distinct components,
`⟨ψ_α|A|ψ_α⟩` is independent of `α` for every `A ∈ π(𝔄)''`. Certificate 4 verifies the phase
independence to `3×10⁻¹⁶` and the membership `P_C ∈ π(𝔄)''` to `9×10⁻¹⁶`. ∎

### 5.6 Reading of Theorem 5 — flagged as interpretation

An abelian commutant carries no multiplicity space, so there is no hidden *quantum*
degeneracy: the unobserved datum is a classical random variable, and by (iii) that variable is
itself measurable — memory is, operationally, a permanent detector displacement.

> In the free asymptotic sector, the information-recovery problem reduces to a **classical
> labelling problem**, not a quantum one. What is not recoverable is coherence across distinct
> classical labels.

This is **consistent with** unitarity rather than in tension with it. It is **not** a proof of
unitarity, not an evaporation theorem, and the relation to independent decoherence arguments
is a consistency observation, not a derivation.

---

# PART III — QUANTITATIVE COMPLETENESS

## 6. Theorems 6 and 7 — the modulus

### 6.1 The objection this part answers

Parts I and II rest entirely on exact-zero criteria: `A' = ℂI`, `P_{r+1} S P_r ≠ 0`, "the graph
is connected". The strongest objection to the whole programme is:

> Informational completeness in the von Neumann sense carries no physical content. An algebra
> can be complete while the coherence it must detect is suppressed by an arbitrarily small
> factor. At finite detector precision, "nonzero" and "zero" are indistinguishable, and a
> completeness theorem with no modulus says nothing — especially as `N` grows.

The objection is correct as stated. Part III removes it.

### 6.2 The coherence-transfer Laplacian

Let `X = Σ_r c_r P_r` be an arbitrary element of the commutant of Proposition 1. Define

```
    w_{rr'} = ‖P_{r'} S P_r‖²_HS,        L = diag(Σ_{r'} w_{rr'}) − (w_{rr'}).      (6.1)
```

Since `[X,S] = Σ_{r≠r'} (c_{r'} − c_r) P_{r'} S P_r` has mutually Hilbert–Schmidt-orthogonal
blocks,

```
    ‖[X,S]‖²_HS = Σ_{r≠r'} |c_r − c_{r'}|² w_{rr'} = 2 c* L c.                     (6.2)
```

This identity is exact.

### 6.3 Theorem 6 — spectral-gap form

> **Theorem 6.** Let `λ₂(L)` be the algebraic connectivity (Fiedler value) of `L`. Then for every
> `X` in the commutant of Proposition 1,
> ```
>     ‖c − c̄‖  ≤  ‖[X,S]‖_HS / √(2 λ₂(L)),                                          (6.3)
> ```
> with **equality exactly on the Fiedler eigenvector**. Consequently
> ```
>     λ₂(L) > 0    ⟺    G(S) connected    ⟺    completeness in the sense of Theorem 1.
> ```

**Proof.** `ker L` is the constants, so on its orthogonal complement `c* L c ≥ λ₂‖c − c̄‖²`.
Substituting into (6.2) gives (6.3); equality holds iff `c − c̄` is a `λ₂`-eigenvector. ∎

**What changes.** Theorem 1's criterion is now the degenerate `λ₂ > 0` case of a metric
statement, and `λ₂` is its modulus. Completeness ceases to be a fragile yes/no property.
Certificate 5(E) verifies both the bound and its saturation — Fiedler ratio `1.0000000000` —
on random multi-mode block families carrying no special structure.

### 6.4 Theorem 7 — dimension-free version

Hilbert–Schmidt norms diverge in the continuum, so Theorem 6 requires a mode truncation. The
following avoids that. Set `w_e = ‖P_{r'} S P_r‖_op` on each edge and define the **resistance
diameter**

```
    D_R(G) = max_{r,r'}  min_{paths r→r'}  Σ_{e ∈ path} 1/w_e.                     (6.4)
```

> **Theorem 7.** `dist(X, ℂ·I)_op = min_c ‖X − cI‖ ≤ (1/2)·D_R(G)·‖[X,S]‖_op.`

**Proof.** `‖[X,S]‖ ≥ ‖P_{r'}[X,S]P_r‖ = |c_{r'} − c_r|·w_e`, so each edge gives
`|c_{r'} − c_r| ≤ ‖[X,S]‖/w_e`. Summing along the cheapest path bounds `|c_r − c_{r'}|` by
`‖[X,S]‖·D_R`. Finally `min_c max_r |c_r − c| ≤ (1/2) max_{r,r'}|c_r − c_{r'}|`, and
`‖X − cI‖ = max_r |c_r − c|`. ∎

---

## 7. Theorem 8 — the exact `su(2)` gap

This is the principal quantitative result. It rests on a structure not visible at the level
of Proposition 2.

### 7.1 The hidden `su(2)`

Restrict `𝒫_κ` to the two-mode subspace generated by the wavepackets
`χ_R = κφ_L/‖κφ_L‖` and `φ_L` used in (2.3), with `λ = ‖κφ_L‖ > 0`. There

```
    𝒫_κ = λ(a_R† a_L + a_L† a_R) = 2λ J_x,                                         (7.1)
```

where `(J_x, J_y, J_z)` is the Schwinger-boson `su(2)` built from the two modes. On the
fixed-`N` sector this is the **spin-`N/2` irreducible representation**, with

```
    J_z = (N_R − N_L)/2,        ⟨r+1|J_x|r⟩ = (1/2)√((r+1)(N−r)).                  (7.2)
```

So the Stokes connector is `2λJ_x`, the obstruction `N_R` is `J_z + N/2`, and **the commutant
of Proposition 1 is exactly the algebra of functions of `J_z`**.

### 7.2 The Laplacian is the adjoint Casimir

The adjoint action of `su(2)` on the fixed-`N` operator space decomposes into spins
`k = 0,1,…,N`, and the adjoint Casimir `ad²_{J_x} + ad²_{J_y} + ad²_{J_z}` acts on the spin-`k`
component by `k(k+1)`. For `X` in the commutant of Proposition 1 we have `[J_z, X] = 0`, and the
residual `x`/`y` symmetry gives `⟨X, ad²_{J_x} X⟩ = ⟨X, ad²_{J_y} X⟩`. Hence on the spin-`k`
component

```
    ‖[X, 2λJ_x]‖²_HS = 2λ² · k(k+1) · ‖X‖²_HS.                                     (7.3)
```

### 7.3 Statement

> **Theorem 8.** For the Stokes connector of Proposition 2,
> ```
>     spec(L) = { λ²·k(k+1) : k = 0,1,…,N },                                       (7.4)
>     λ₂(L)  = 2λ²        EXACTLY, FOR EVERY N,                                    (7.5)
> ```
> and therefore
> ```
>     ‖X − X̄·I‖_HS  ≤  ‖[X, 𝒫_κ]‖_HS / (2λ)        uniformly in N.                 (7.6)
> ```
> The extremal (least visible) operator is the Fiedler eigenvector `c_r = r − N/2`, that is
> ```
>     X  ∝  N_R  —  the obstruction identified in Proposition 1.                    (7.7)
> ```

### 7.4 Why the `N`-independence is not trivial

A path graph on `N+1` vertices with uniform weights has `λ₂ ~ π²/N²`, so the naive expectation
is that completeness degrades as `N⁻²`. The weights `(r+1)(N−r)` grow exactly fast enough to
cancel this. **That cancellation is the `su(2)` structure.** It is a fact about the spin-2
Stokes connector, not an artefact of the model.

The operator-norm route agrees independently. Since `w_r ≥ λ√((r+1)(N−r)) ≥ λ√N`,

```
    D_R ≤ Σ_{r=0}^{N-1} 1/(λ√((r+1)(N−r)))  →  π/λ        as N → ∞,                (7.8)
```

a convergent sum. Certificate 5(F) records `D_R = 3.0960` at `N = 4096` against the limit
`π/λ = 3.1416`. So the dimension-free bound of Theorem 7 is also uniform in `N`.

### 7.5 Certificates

- 5(A): `max|spec(L) − λ²k(k+1)| < 10⁻¹²` for `N` up to 40.
- 5(B): `λ₂ = 2.00000000` at `N = 5, 12, 30, 100`; Fiedler vector matches `r − N/2` to `10⁻¹⁴`.
- 5(C): identity (7.3) verified for `k = 1, 2, 3`.
- 5(D): bound (7.6) verified over random `X` at `N = 4, 10, 25, 60`.

### 7.6 Operational reading

Suppose the connector channel is measured to precision `ε`, i.e. operators `X` with
`‖[X, 𝒫_κ]‖_HS ≤ ε` are not resolved. By (7.6) the residual ambiguity is at most `ε/(2λ)`.

> To resolve cross-helicity coherence to accuracy `δ`, one needs `ε ~ 2λδ` — **independent of the
> number of gravitons**.

### 7.7 Why this closes the chain

Proposition 1 identified `N_R` as exactly the missing information. Theorem 8 shows `N_R` is
also exactly the *hardest* operator for the Stokes connector to see: it is the slowest mode of
the coherence transfer. Its visibility is `2λ`, with no `N` suppression. The obstruction and
the extremal direction coincide.

---

## 8. Theorem 9 — the infrared rate

Theorem 4 delivered the memory question as a binary dichotomy because the literature is
split. Theorem 9 shows the two positions are not contradictory: they are the two ends of one
continuous quantity.

Introduce an infrared regulator `μ` — a graviton mass, or the dressing-cloud regulator used in
the Wilson-line dressing literature. At finite `μ`, memory sectors are **not** disjoint:
overlaps between differently dressed sectors are nonzero and vanish as a power `μ^α` as the
regulator is removed, where `α` depends on the relative rapidity and dressing geometry [11].

> **Theorem 9.**
> ```
>     For every μ > 0 :   w_memory ~ μ^{2α} > 0   ⟹   the memory graph is CONNECTED,
>                         completeness HOLDS, with modulus diverging as D_R(μ) ~ μ^{−α}.
>     As μ → 0        :   λ₂(μ) → 0, the gap closes, superselection is RECOVERED.
> ```

**Proof.** Immediate from Theorems 6 and 7 applied with the regulated weights. ∎

### 8.1 The reconciliation

- The position that memory is **not** superselected, because any finite-energy process emits
  soft quanta connecting the vacua [10], is the statement `λ₂(μ) > 0` for every `μ > 0`. **Correct.**
- The position that memory **is** a superselection label [7,9] is the statement `λ₂(μ) → 0` as
  `μ → 0`. **Also correct.**

The disagreement is not about a fact. It is about an **order of limits**, and the spectral gap
of the coherence-transfer Laplacian is the quantity that interpolates between them at a
computable rate.

### 8.2 Scope of Theorem 9

The **structure** is proved: that the gap is the interpolating quantity, that a positive gap at
finite `μ` implies completeness with modulus `μ^{−α}`, and that the gap closing is exactly
superselection. The **exponent `α` is not derived here**; it is taken as an input from the
dressing literature, and different dressing geometries give different `α`. Anyone quoting a
specific numerical rate must derive `α` for their own setup.

---

## 9. Consolidated statement of results

| # | Result | Status |
|---|--------|--------|
| **T1** | Sector Connectivity Theorem: commutant `= ℓ^∞(comp G(S))`; algebra `= ⊕_C B(H_C)`; complete iff `G(S)` connected | Proved; mathematical novelty **not claimed** |
| **T1.1** | Spanning-tree bound: `≥ N` off-diagonal blocks required; the Stokes connector supplies exactly `N` | Proved |
| **T1.2** | Propositions 1 and 2 recovered as the empty-graph and path-graph cases | Proved |
| **T2** | Exact Fock commutant `𝔄_F' = W*(N̂)`, not scalar — hence not Stone–von Neumann | Proved |
| **T3** | `W*(𝔄_F, Φ(f)) = B(F)` for every nonzero `f ∈ H₁`, with no genericity hypothesis; soft charge identified as the missing connector type; failure locus `f_g ∉ H₁` | Proved |
| **T4** | Memory dichotomy: both branches classified; Branch 1 gives a structural no-go | Proved as dichotomy; (M1) not adjudicated |
| **T5** | Commutant abelian; representation multiplicity-free; sector label measurable; only cross-component phase irrecoverable | Proved, conditional on sector-wise structure |
| **T6** | Completeness modulus `= λ₂` of the coherence-transfer Laplacian; sharp on the Fiedler eigenvector | Proved |
| **T7** | Dimension-free bound via resistance diameter | Proved |
| **T8** | `spec(L) = λ²k(k+1)`; `λ₂ = 2λ²` exactly, independent of `N`; extremal operator `= N_R`; `D_R → π/λ` | Proved |
| **T9** | `λ₂(μ) > 0` at finite regulator, `λ₂(μ) → 0` in the limit: the memory controversy is an order-of-limits distinction | Structure proved; exponent imported |

---

## 10. What is not proved

- **The black-hole information paradox is not solved.** No part of it is.
- **No dynamics anywhere.** Everything is free asymptotic radiative data. There is no
  interaction, no evaporation model, and no initial-to-out map.
- **Hypothesis (M1) is not resolved.** Theorem 4 deliberately refuses to adjudicate it.
- **The exponent `α` of Theorem 9 is imported, not derived.**
- **No new construction of `Q_soft`.** §4.4 is a structural correspondence; §4.4's closing
  paragraph records the failure locus.
- **Theorem 1 is not claimed as new mathematics.** See §3.8.
- **The exact value `2λ²` in Theorem 8** uses the two-mode `su(2)` reduction of `𝒫_κ`. For a
  general `κ` acting across many modes, Theorem 6 applies with the corresponding `λ₂`, and the
  bound `D_R ≤ π/λ` of Theorem 7 still holds because block operator norms only increase when
  more modes are available. The exact value should be quoted as the `su(2)`-reduction value.
- **No priority claim.** Fiedler bounds, Schwinger bosons, block-algebra structure theory and
  the factor dichotomy are all standard. What is offered is their combination in this setting
  and the exact `N`-independence it produces.

---

## 11. References

[1] Donnelly, W. *Quantum gravity tomography.* arXiv:1806.05643 (2018).
[2] Ball, A.; Himwich, E.; Narayanan, S. A.; Pasterski, S.; Strominger, A. *Uplifting AdS₃/CFT₂ to flat space holography.* arXiv:1905.09809 (2019).
[3] Liu, W.-B.; Long, J. *Symmetry group at future null infinity III: gravitational theory.* arXiv:2307.01068 (2023).
[4] Liu, W.-B.; Long, J.; Zhou, X.-H. *Quantum flux operators in higher spin theories.* arXiv:2311.11361 (2023).
[5] Gubitosi, G.; Magueijo, J. *Correlation between opposite-helicity gravitons.* arXiv:1610.05702 (2016).
[6] Prabhu, K.; Satishchandran, G.; Wald, R. M. *Infrared finite scattering theory in quantum field theory and quantum gravity.* arXiv:2203.14334 (2022).
[7] Riello, A.; Schiavina, M. *Null Hamiltonian Yang–Mills theory: soft symmetries and memory as superselection.* Ann. Henri Poincaré (2024). arXiv:2303.03531.
[8] Prabhu, K.; Satishchandran, G. *Infrared finite scattering theory: scattering states and representations of the BMS group.* JHEP 08 (2024) 055. arXiv:2402.00102.
[9] Dominguez, A.; Kozameh, C. *Superselection sectors in asymptotic quantization of gravity.* arXiv:gr-qc/9609071.
[10] Donnay, L.; Herfray, Y. *Infrared physics of QED and gravity from representation theory.* arXiv:2603.06297.
[11] Semenoff, G. W.; Waterfield, C. *The Wilson-line-dressed charged sector of scalar QED: superselection and the infraparticle.* arXiv:2609.06224 (2026).
[12] Fiedler, M. *Algebraic connectivity of graphs.* Czechoslovak Math. J. 23 (1973) 298–305.
