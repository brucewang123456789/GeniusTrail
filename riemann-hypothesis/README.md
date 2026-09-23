# VeriLoop E2: 67.350003708785593% for Simple Critical-Line Zeta Zeros

**A strict finite-dimensional computer-assisted certificate built on Anthropic's public 67.25% analytic foundation**

| Item | Value |
|---|---|
| Research system | **VeriLoop E2** |
| Author / maintainer | **Libo Wang** |
| Frozen result | **67.350003708785593%** |
| Public starting point | **Anthropic: 67.2500703679%** |
| Improvement | **+0.0999333409 percentage points** |
| Certificate status | **Strict finite-dimensional computer-assisted certificate** |
| Formalization status | **End-to-end Lean verification in progress** |
| Reproducibility | **Verifier, configuration, certificate logs, exact assembly, and audit material released** |

---

## Overview

This repository releases the derivation, numerical construction, strict verifier, raw certificate logs, exact-rational assembly, audit material, and formalization roadmap for the following lower-bound witness:

```text
κ ≥ 0.67350003708785593...
  = 67.350003708785593%
```

Here `κ` denotes the asymptotic lower proportion of nontrivial zeros of the Riemann zeta function that are both:

1. on the critical line `Re(s) = 1/2`; and
2. simple.

The work starts from Anthropic's public 2026 analytic framework and its public 67.25% result. VeriLoop E2 does **not** relabel that foundation as original work. The contribution developed here is a stricter finite-dimensional spectral/block construction that retains additional matrix information and converts it into a certified positive correction.

The resulting frozen witness is:

```text
722547711262091300265625000000000
──────────────────────────────────
1072825050442925667061714615173641

= 0.67350003708785593...
= 67.350003708785593%
```

The improvement over the public Anthropic baseline is:

```text
67.350003708785593%
− 67.2500703679%
──────────────────
≈ 0.0999333409 percentage points
```

> **Scientific boundary**
>
> This repository does **not** prove the Riemann Hypothesis and does **not** claim that “67.35% of RH” has been solved.  
> The claim is a lower-bound witness for the proportion of zeta zeros satisfying the specific property **simple and on the critical line**.

---

## 1. Quantity being bounded

Let `N(T)` be the number of nontrivial zeros

```text
ρ = β + iγ,    0 < γ ≤ T
```

counted with multiplicity.

Let `S(T)` be the number of those zeros that are both simple and satisfy

```text
β = 1/2.
```

The object studied in this repository is

```text
κ = lim inf_{T→∞} S(T) / N(T).
```

The frozen VeriLoop E2 certificate establishes the witness

```text
κ ≥ 67.350003708785593%.
```

A simple critical-line zero is, in particular, a distinct critical-line zero; the quantity therefore fits naturally into the public comparison framework used for critical-line zero proportions.

---

## 2. Starting point: Anthropic's public 67.25% result

The analytic starting point is Anthropic's public work:

**More Than Two Thirds of the Zeros of the Riemann Zeta Function Lie on the Critical Line**

Paper:

https://www-cdn.anthropic.com/564f962e60643842f5fcb4a17c9dbc8f608f1c37.pdf

Lean 4 formalization:

https://github.com/anthropics/zeta-23-lean

The public optimized baseline used here is

```text
κ₀ = 0.672500703679...
   = 67.2500703679...%.
```

At a high level, the public analytic interface can be written in the form

```text
S ≥ H(v) N + Δ(M) − o(N),
```

where:

- `H(v)` is the contribution from the chosen window;
- `M` is a positive-semidefinite Gram-type matrix;
- `Δ(M)` is a nonnegative finite-dimensional spectral correction.

The baseline argument may retain the optimized window contribution while discarding the additional nonnegative correction.

The VeriLoop E2 construction takes a different route:

> **Rather than claiming a larger window value, it certifies a quantitative lower bound for the finite-dimensional correction `Δ(M)`.**

That distinction is the central mechanism behind the move from approximately `67.25%` to `67.35%`.

---

## 3. Derivation of the 67.350003708785593% witness

### 3.1 Window contribution

For an even nonnegative window `v` supported on `[-1/2, 1/2]`, define

```text
I₁ = ∫ v

I₂ = ∫ v²

J  = ∬ |s − t| v(s) v(t) ds dt
```

and

```text
H(v) = 2 − (I₂ + J) / I₁².
```

The frozen strict witness uses the rigorous lower bound

```text
H = 672167187145431 / 10¹⁵
  = 0.672167187145431.
```

This value is deliberately **below** the best window-only value.

The construction therefore does not obtain its gain by improving `H(v)`. It accepts a small loss in the window term in exchange for a larger certified gain from finite-dimensional matrix structure.

---

### 3.2 Finite-dimensional spectral correction

Define the scalar function

```text
Ψ(t) = (t − 1)²    for 0 ≤ t ≤ 2
       2t − 3      for t ≥ 2
```

and the spectral functional

```text
Δ(G) = tr Ψ(G).
```

For

```text
X = G − I
U = (G − 2I)₊,
```

the finite spectral identity used in this release is

```text
Δ(G) = ||X − U||_F² + 2 tr(U).
```

Now retain matrix interactions within bandwidth `q` and define

```text
E = 2 Σ |Gᵢⱼ|²
```

over pairs satisfying

```text
1 ≤ j − i ≤ q.
```

For maximum degree

```text
d = 2q,
```

colouring number

```text
r = q + 1,
```

and threshold

```text
T = (q + 1) / q,
```

the finite-dimensional matrix bound takes the form

```text
Δ(G) ≥ h(E),
```

with

```text
h(E) = E                                      for 0 ≤ E ≤ T

h(E) = E − [d/(d+1)] (√E − √T)²             for E ≥ T.
```

The frozen strict configuration uses

```text
q = 8
d = 16
r = 9
T = 9/8.
```

The theorem itself is formulated at the finite-dimensional level; moving to `q = 8` changes the certification burden rather than changing the logical form of the spectral argument.

---

### 3.3 Eight-gap local certificates

Let

```text
W(x) = [K(x) / K(0)]²,
```

where `K` is the Fourier transform of the chosen window.

For eight nonnegative gaps

```text
g₁, g₂, ..., g₈ ≥ 0,
```

define cumulative positions

```text
y₀ = 0
yⱼ = g₁ + ... + gⱼ.
```

The local pressure and pair-energy terms are

```text
P_loc(g) = Σ bᵣ gᵣ

Q_loc(g) = Σ aᵢⱼ W(yⱼ − yᵢ).
```

The nonnegative pair weights are chosen to satisfy exact span-capacity identities:

```text
Σ aᵢ,ᵢ₊ₛ = 2,    s = 1, ..., 8.
```

The local certification target is

```text
P_loc(g) + s Q_loc(g) ≥ εₛ
```

for every point in the eight-dimensional nonnegative gap domain.

The frozen strict release certifies the following three slopes:

| Slope `s` | Certified `εₛ` | Strict B&B nodes | Status |
|---:|---:|---:|:---:|
| `1/2` | `526/78125 = 0.0067328` | `23,644,472` | **PROVED** |
| `19/20` | `9879/1250000 = 0.0079032` | `75,294,070` | **PROVED** |
| `1` | `20033/2500000 = 0.0080132` | `91,437,288` | **PROVED** |
| **Total** |  | **190,375,830** | **3/3 PROVED** |

Each committed strict run is fail-closed and terminates with

```text
stack_left=0
HARD=0
result=PROVED
```

The `s = 1/2` certificate is closed by direct strict branch-and-bound.

For `s = 19/20` and `s = 1`, the difficult local minima are isolated into a separately certified well layer. Before those wells are allowed to participate in the global proof, an independent directed-interval checker verifies every frozen well box:

```text
WELLS_V2_TOTAL=327
PROVED=327
FAILED=0
```

This separation is intentional:

> Numerical search is allowed to discover difficult regions, but discovery is never treated as proof.

---

### 3.4 From local certificates to a block inequality

Take blocks of length `m` and write

```text
n = m − q.
```

The frozen configuration is

```text
m = 531
q = 8
n = 523.
```

Summing the translated local certificates and using the exact span-capacity identities gives

```text
P + sE ≥ n εₛ.
```

Define the pressure envelope

```text
p(E) = max(0, maxₛ(n εₛ − sE)).
```

Hence

```text
P ≥ p(E).
```

For `η ≥ 0`, define

```text
R = inf_{E≥0} [ h(E) + η p(E) ].
```

The frozen witness uses

```text
η = 1
```

and the exact rational lower bound

```text
R =
1113172768314043426732876281699
───────────────────────────────
265625000000000000000000000000

= 4.190768068946987...
```

Because `p(E)` is piecewise linear, the committed evaluation reduces to a finite collection of relevant kinks and endpoints. The frozen lower bound for `R` is therefore checked in exact rational arithmetic.

The pressure-weight sum is

```text
B = Σ bᵣ = 93/23000.
```

Block averaging and pinching then give

```text
Δ(M) ≥ (R/m) S − [η B (m−q) / m] N.
```

The finite spectral inequality, the pinching step, and the shifted pressure accounting are separately audited in this release.

---

### 3.5 Exact final assembly

Return to the analytic interface

```text
S ≥ HN + Δ(M) − o(N).
```

Insert the certified block bound

```text
Δ(M) ≥ (R/m) S − [η B (m−q) / m] N.
```

After rearrangement,

```text
S [1 − R/m]
≥
N [H − η B (m−q)/m].
```

Therefore

```text
κ ≥ C
```

with

```text
C = [mH − η B (m−q)] / [m − R].
```

The frozen exact parameters are

```text
q   = 8
m   = 531
η   = 1
B   = 93/23000

H   = 672167187145431 / 10¹⁵

R   =
      1113172768314043426732876281699
      ───────────────────────────────
      265625000000000000000000000000
```

The exact assembly produces

```text
C =
722547711262091300265625000000000
──────────────────────────────────
1072825050442925667061714615173641
```

and therefore

```text
κ ≥ 0.67350003708785593...
  = 67.350003708785593%.
```

The committed exact checker verifies:

```text
PASS exact C equals frozen fraction
PASS exact C > 0.6735
```

The final acceptance condition is therefore not a rounded floating-point comparison.

---

## 4. Why this is more than a numerical candidate

The release separates three logically different activities.

### Discovery

Floating-point numerical search is used to propose:

- candidate windows;
- pressure weights;
- pair weights;
- slopes;
- difficult local regions;
- parameter combinations worth certifying.

Search output is never accepted as proof.

### Certification

Accepted local statements are replayed through a strict fail-closed verification chain using:

- interval branch-and-bound;
- outward-safe interval storage;
- directed widening at arithmetic boundaries;
- interval Taylor enclosures for the transcendental kernel;
- rigorous first- and second-derivative bounds;
- symmetry preflight;
- an independently certified well layer;
- zero unresolved terminal boxes.

### Exact assembly

The final span identities, pressure sum, pressure-envelope evaluation, `R`, and the final lower bound `C` are checked with exact rational arithmetic.

The committed evidence audit reports:

```text
PASS R lock is a valid rational lower bound of the complete kink minimum
PASS exact C equals frozen fraction
PASS exact C > 0.6735
PASS 327/327 strict well boxes proved
PASS constant/rational enclosure audit passed
TOTAL_STRICT_BB_NODES=190375830
RESULT: ALL STRICT 67.35 CHECKS PASS
```

This is why the frozen value is described as a **strict finite-dimensional computer-assisted certificate**, rather than as a floating-point optimization result.

---

## 5. Development path

The project retained a lower-cost certificate tier while pursuing the stricter 67.35% witness.

| Stage | Certified value | Role |
|---|---:|---|
| Anthropic public baseline | **67.2500703679%** | Public analytic/formal starting point |
| VeriLoop E2 short tier | **67.275055959117140%** | Lower-cost reproducible certificate; `793,374` B&B nodes |
| VeriLoop E2 strict tier | **67.350003708785593%** | Primary frozen strict finite-dimensional certificate |

The short tier is methodologically useful because it shows that certification cost is margin-driven. The project therefore does not optimize decimal digits in isolation: it first chooses the threshold that must be exceeded, then works backward to the loosest certificate margins that still close rigorously.

The main frozen result of this repository is the strict tier:

```text
67.350003708785593%.
```

---

## 6. What has been achieved

The following statements describe the current frozen release.

### Exact frozen witness

```text
67.350003708785593%
```

### Strict local certification

All three frozen local inequalities terminate fail-closed with

```text
result=PROVED.
```

### Independent difficult-well certification

```text
327 / 327 proved
0 failed
```

### Zero unresolved terminal boxes

Committed strict runs terminate with

```text
stack_left=0
HARD=0.
```

### Exact-rational final assembly

The final `R` and `C` checks do not depend on decimal rounding.

### Audited finite-dimensional bridge

The release explicitly audits:

- the finite spectral inequality;
- the pinching step;
- shifted pressure accounting;
- safe-direction numerical constants;
- the final rational assembly.

### Public reproducibility

The repository releases the verifier source, frozen configuration, committed certificate logs, exact assembly checks, audit material, and reproduction commands required to inspect the finite-dimensional result.

### Explicit provenance

The Anthropic/Zeta23 analytic foundation is identified as the upstream starting point. The VeriLoop E2 contribution is presented as the finite-dimensional extension and strict certificate built on top of that public foundation.

---

## 7. What is still in progress

The finite-dimensional certificate is frozen. The remaining work is the **end-to-end formal bridge**.

The team is actively working on the following items.

### 7.1 End-to-end correspondence with the upstream analytic interface

The remaining formal layer must connect the finite-dimensional certificate to the upstream theorem statements with all normalizations fixed explicitly, including:

- zero heights;
- gap normalization;
- multiplicities;
- matrix normalization;
- the precise attachment of the strengthened `Δ(M)` estimate.

### 7.2 Kernel-friendly certificate compression

The current strict branch-and-bound computation is strong external evidence but is far too large to replay naively inside the Lean kernel.

A major objective is therefore to replace the enormous search tree with a much smaller formal object.

One candidate route is:

1. derive rigorous polynomial lower bounds for the kernel `W` over the already reduced domain;
2. reduce the required inequalities to polynomial nonnegativity;
3. certify them using a Positivstellensatz / sum-of-squares object;
4. replay the resulting bounded rational certificate in Lean.

### 7.3 Lean formalization

The team is developing an end-to-end Lean formalization compatible with the upstream theorem interface and the frozen finite-dimensional witness.

The intended endpoint is not merely a successful external numerical replay, but a proof object that can survive independent kernel checking.

### 7.4 Independent review and formal submission

Independent mathematical review and formal submission remain in progress.

The repository is being published now precisely so that the derivation and certificate can be challenged before that process is complete.

---

## 8. What this repository does not claim

For clarity, the following claims are **not** being made:

- this is not a proof of the Riemann Hypothesis;
- this is not a statement that “67.35% of RH” has been solved;
- this is not yet a completed end-to-end Lean theorem;
- this is not yet an accepted formal challenge or ledger entry;
- this is not yet a claim of completed independent peer review;
- this repository does not self-declare mathematical-community acceptance.

The strongest intended wording at the present stage is:

> **A strict, independently reproducible finite-dimensional computer-assisted certificate for the 67.350003708785593% witness, built on Anthropic's public analytic foundation and released for independent audit, formalization, and review.**

---

## 9. Reproducing the frozen evidence

### Quick committed-evidence audit

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt

./reproduce.sh
```

Expected evidence includes:

```text
PASS R lock is a valid rational lower bound of the complete kink minimum
PASS exact C equals frozen fraction
PASS exact C > 0.6735
PASS 327/327 strict well boxes proved
TOTAL_STRICT_BB_NODES=190375830
RESULT: ALL STRICT 67.35 CHECKS PASS
COMMITTED_EVIDENCE_AUDIT=PASS
```

### Full strict replay

```bash
FULL=1 ./reproduce.sh
```

The full mode rebuilds and replays the expensive strict verification path from source.

Derived lookup caches are not treated as trusted evidence; they are reproducible artifacts.

---

## 10. Repository structure

| Path | Purpose |
|---|---|
| `README.md` | Result, derivation, boundaries, and reproduction |
| `THEORY.md` | Finite-dimensional reduction and mathematical argument |
| `NUMERICS.md` | Numerical search methodology and design decisions |
| `CERTIFICATES.md` | Strict certificate summary |
| `AUDIT.md` | Spectral, pinching, pressure-accounting, and numerical audit |
| `STATUS.md` | Evidence-status ledger |
| `SUBMISSION.md` | Comparison with the public baseline and remaining formal boundary |
| `PROVENANCE.md` | Provenance of the frozen witness |
| `REFUTATION.md` | Rejected provisional candidates and failure analysis |
| `TIERS.md` | Short-tier and strict-tier certificate definitions |
| `verify_exact.py` | Exact-rational final assembly |
| `verify_strict.py` | Strict committed-certificate audit |
| `audit_constants.py` | Constant/enclosure safety audit |
| `arb_window.py` | Rigorous window evaluation |
| `verifier/` | Strict interval verifier implementation |
| `certificates/strict/` | Raw committed strict logs |
| `config/strictlock.json` | Frozen strict witness configuration |
| `lean/LEAN_ROADMAP.md` | Formalization roadmap |
| `lean/Solution.skeleton.lean` | Lean theorem contract / proof skeleton |
| `reproduce.sh` | One-command audit and optional full replay |

---

## 11. Research transparency

Computer-assisted mathematics is useful only if failed candidates are allowed to fail visibly.

During development, higher provisional numerical candidates were rejected when explicit counterexamples exposed insufficiently converged local minima. Those candidates are not silently retained as achievements; their failure modes are documented in `REFUTATION.md`.

The frozen 67.350003708785593% release was rebuilt around a stricter acceptance chain:

1. separate discovery from proof;
2. freeze exact rational coefficients;
3. certify difficult wells independently;
4. prove the global complement fail-closed;
5. audit numerical constant directions and interval storage;
6. assemble the final result exactly over the rationals.

That evidence discipline is part of the result.

---

## 12. Invitation to reproduce, audit, and formalize

The full finite-dimensional calculation is being released because it should be independently tested.

Contributions are especially welcome from researchers working in:

- analytic number theory;
- rigorous interval arithmetic;
- Lean / Mathlib;
- formalized mathematics;
- exact computational proof systems;
- SOS / Positivstellensatz certificates.

Useful contributions include:

- independent reproduction of the frozen certificate;
- attempts to falsify or tighten any local inequality;
- verification of the block accounting;
- independent reimplementation of the exact assembly;
- compression of the large branch-and-bound certificate;
- construction of a kernel-checkable algebraic certificate;
- completion of the Lean bridge to the upstream analytic theorem.

VeriLoop E2 is deliberately used here as a research system willing to explore difficult directions, including directions that can fail.

The standard is not that every proposal succeeds.

The standard is that the final public claim is separated from rejected candidates and survives increasingly strict layers of evidence.

We welcome independent formalization and hope the public release helps move the result toward a successful end-to-end formal submission.

---

## 13. References and attribution

### Anthropic foundation

**Anthropic / Claude (2026)**  
*More Than Two Thirds of the Zeros of the Riemann Zeta Function Lie on the Critical Line*

https://www-cdn.anthropic.com/564f962e60643842f5fcb4a17c9dbc8f608f1c37.pdf

**Zeta23 — Lean 4 formalization**

https://github.com/anthropics/zeta-23-lean

The present VeriLoop E2 work explicitly builds on this public analytic and formal foundation.

### VeriLoop E2 release

**VeriLoop E2 + Libo Wang**

*Strict 67.350003708785593% finite-dimensional computer-assisted certificate for simple critical-line zeta zeros*

Author / maintainer: **Libo Wang**

---

## License

See `LICENSE`.

---

## Citation

If you use this artifact, verifier, configuration, or certificate chain, please distinguish clearly between:

1. the Anthropic/Zeta23 analytic foundation;
2. the VeriLoop E2 finite-dimensional extension and strict certificate;
3. any subsequent independent formalization, modification, or verification.
