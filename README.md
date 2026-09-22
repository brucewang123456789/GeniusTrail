# VeriLoop E2: 67.350003708785593% of Simple Critical-Line Zeta Zeros

| Field | Value |
|---|---|
| **Research system** | VeriLoop E2 |
| **Author / maintainer** | Libo Wang |
| **Status** | Strict finite-dimensional computer-assisted certificate; independent reproduction and formal verification invited |
| **Frozen result** | 2026-09-17 |

This repository releases the derivation, verifier, raw certificate logs, exact-rational assembly, audit material, and formalization roadmap for a **67.350003708785593% lower-bound witness** for the asymptotic proportion of nontrivial zeros of the Riemann zeta function that are **simple and lie on the critical line**.

The work starts from the public analytic framework released by Anthropic in 2026, including the paper *More Than Two Thirds of the Zeros of the Riemann Zeta Function Lie on the Critical Line* and the accompanying Lean 4 formalization **Zeta23**. Anthropic's optimized public baseline is

$$
\kappa_0
=
0.672500703679\ldots
=
67.2500703679\ldots\%.

$$

The frozen VeriLoop E2 witness is

$$
\boxed{
\kappa
\ge
\frac{
722547711262091300265625000000000
}{
1072825050442925667061714615173641
}
=
0.67350003708785593\ldots
}

$$

or

$$
\boxed{67.350003708785593\%}.

$$

The numerical improvement over the Anthropic baseline is therefore

$$
+0.0999333409\ \text{percentage points}.

$$

The important point is **where the gain comes from**. It is not obtained by claiming a better value of Anthropic's window functional. The frozen VeriLoop E2 window deliberately has

$$
H(v)\ge 0.672167187145431,

$$

which is below the Anthropic optimized window-only value. The improvement comes from retaining and certifying additional finite-dimensional matrix structure that the baseline estimate does not exploit.

---

## 1. What is being bounded

Let $N(T)$ denote the number of nontrivial zeros

$$
\rho=\beta+i\gamma,\qquad 0<\gamma\le T,

$$

counted with multiplicity, and let $S(T)$ denote the number of those zeros that are both

1. on the critical line $\beta=\tfrac12$, and
2. simple.

The quantity studied here is

$$
\kappa
=
\liminf_{T\to\infty}
\frac{S(T)}{N(T)}.

$$

This distinction is essential.

> [!IMPORTANT]
> **This repository does not prove the Riemann Hypothesis.**  
> It does not claim that 67.35% of RH has been proved.  
> It gives a lower-bound witness for the proportion of nontrivial zeta zeros that are **simple and lie on the critical line**.

Because a simple critical-line zero is in particular a distinct critical-line zero, this quantity is compatible with the distinct-critical-line count appearing in the public comparison framework.

---

## 2. Starting point: Anthropic's public 67.25% result

Anthropic's 2026 work gives an unconditional critical-line/simple-zero proportion based on a finite compression of Weil's Hermitian form together with the Montgomery–Taylor optimized test family.

The public paper and formalization are:

- Paper: https://www-cdn.anthropic.com/564f962e60643842f5fcb4a17c9dbc8f608f1c37.pdf
- Lean 4 formalization: https://github.com/anthropics/zeta-23-lean

At the optimized window, the public baseline is

$$
\kappa_0
=
2-\frac{1}{c_{\mathrm{MT}}}
=
0.672500703679\ldots.

$$

The VeriLoop E2 programme **uses this public analytic foundation as its starting point**. It does not present the Anthropic analysis as our own work.

The additional contribution developed here is a finite-dimensional spectral/block mechanism that retains information discarded by the window-only lower bound.

At a high level, the Anthropic interface supplies a bound of the form

$$
S
\ge
H(v)N+\Delta(M)-o(N),

$$

where $M$ is a positive-semidefinite Gram-type matrix, $H(v)$ is the window contribution, and $\Delta(M)\ge0$ is a spectral correction.

The baseline can discard the nonnegative correction and keep only the optimized $H(v)$.  
The VeriLoop E2 construction instead develops a quantitative, certified lower bound for that correction.

That is the source of the 67.25% $\rightarrow$ 67.35% gain.

---

## 3. Derivation: from the analytic interface to 67.350003708785593%

### 3.1 Window term

For an even nonnegative window $v$ supported on $[-1/2,1/2]$, define

$$
I_1=\int v,\qquad
I_2=\int v^2,\qquad
J=\iint |s-t|\,v(s)v(t)\,ds\,dt,

$$

and

$$
H(v)=2-\frac{I_2+J}{I_1^2}.

$$

The frozen strict witness uses the rigorous lower bound

$$
H
=
\frac{672167187145431}{10^{15}}
=
0.672167187145431.

$$

This is deliberately **below** the best window-only value. The construction trades a small loss in $H$ for a larger gain from finite-dimensional structure.

---

### 3.2 Spectral correction

Define

$$
\Psi(t)=
\begin{cases}
(t-1)^2,&0\le t\le2,\$$4pt]
2t-3,&t\ge2,
\end{cases}

$$

and

$$
\Delta(G)=\operatorname{tr}\Psi(G).

$$

For $X=G-I$ and $U=(G-2I)_+$, the finite spectral identity proved in this release is

$$
\Delta(G)
=
\|X-U\|_F^2
+
2\operatorname{tr}U.

$$

For retained bandwidth $q$, define

$$
E
=
2\sum_{1\le j-i\le q}|G_{ij}|^2.

$$

With maximum degree $d=2q$, colouring number $r=q+1$, and

$$
T=\frac{q+1}{q},

$$

the matrix theorem gives

$$
\Delta(G)\ge h(E),

$$

where

$$
h(E)=E,\qquad 0\le E\le T,

$$

and

$$
h(E)
=
E-\frac{d}{d+1}
\left(\sqrt E-\sqrt T\right)^2,
\qquad E\ge T.

$$

For the frozen witness,

$$
q=8,\qquad
d=16,\qquad
r=9,\qquad
T=\frac98.

$$

The theorem is proved for general $(d,r)$; moving to $q=8$ therefore changes the certification burden, not the underlying matrix theorem.

---

### 3.3 Local eight-gap inequalities

Let

$$
W(x)=\left(\frac{K(x)}{K(0)}\right)^2,

$$

where $K$ is the Fourier transform of the chosen window.

For eight nonnegative gaps $g_1,\ldots,g_8$, define cumulative positions

$$
y_0=0,\qquad
y_j=g_1+\cdots+g_j,

$$

and introduce

$$
P_{\mathrm{loc}}(g)
=
\sum_{r=1}^{8} b_r g_r,

$$

$$
Q_{\mathrm{loc}}(g)
=
\sum_{0\le i<j\le8}
a_{ij}W(y_j-y_i).

$$

The nonnegative pair weights satisfy the exact span-capacity identities

$$
\sum_{i=0}^{8-s}a_{i,i+s}=2,
\qquad
s=1,\ldots,8.

$$

The core local statement is

$$
P_{\mathrm{loc}}(g)+sQ_{\mathrm{loc}}(g)
\ge
\varepsilon_s
\qquad
\text{for all }g\in[0,\infty)^8.

$$

The frozen strict release certifies three slopes:

| slope $s$ | certified $\varepsilon_s$ | strict B&B nodes | terminal state |
|---:|---:|---:|---|
| $1/2$ | $526/78125=0.0067328$ | 23,644,472 | **PROVED** |
| $19/20$ | $9879/1250000=0.0079032$ | 75,294,070 | **PROVED** |
| $1$ | $20033/2500000=0.0080132$ | 91,437,288 | **PROVED** |
| **total** |  | **190,375,830** | **3/3 PROVED** |

Every committed local run terminates fail-closed with

```text
stack_left=0
HARD=0
result=PROVED
```

The $s=\tfrac12$ inequality is closed by direct strict branch-and-bound.

For $s=\tfrac{19}{20}$ and $s=1$, the difficult local minima are isolated into an independently checked well layer. Before the global verifier may use those wells, a separate directed-interval checker proves all frozen well boxes:

```text
WELLS_V2_TOTAL=327
PROVED=327
FAILED=0
```

Only then may the global complement proof prune against them.

This separation is deliberate: numerical search may discover difficult regions, but discovery is not trusted as proof.

---

### 3.4 From local inequalities to a block inequality

Take blocks of length $m$ and let

$$
n=m-q.

$$

For the frozen witness,

$$
m=531,\qquad
q=8,\qquad
n=523.

$$

Summing the local certificates over translated windows and using the span-capacity identities gives

$$
P+sE\ge n\varepsilon_s.

$$

Therefore define the piecewise-linear pressure envelope

$$
p(E)
=
\max\left(
0,\,
\max_s(n\varepsilon_s-sE)
\right).

$$

Thus

$$
P\ge p(E).

$$

For $\eta\ge0$,

$$
R
=
\inf_{E\ge0}
\left[h(E)+\eta p(E)\right].

$$

The frozen strict witness uses

$$
\eta=1

$$

and the exact rational lower bound

$$
R
=
\frac{
1113172768314043426732876281699
}{
265625000000000000000000000000
}
=
4.190768068946987\ldots.

$$

Because $p(E)$ is piecewise linear and the relevant minimization reduces to a finite set of kinks/endpoints, the committed value of $R$ is evaluated in exact rational arithmetic.

The pressure-weight sum is

$$
B=\sum_r b_r=\frac{93}{23000}.

$$

The block averaging and pinching argument then yields

$$
\Delta(M)
\ge
\frac{R}{m}S
-
\frac{\eta B(m-q)}{m}N.

$$

The finite spectral inequality, pinching step, and shifted pressure accounting are audited explicitly in `AUDIT.md`.

---

### 3.5 Exact assembly

Insert the block inequality into

$$
S\ge HN+\Delta(M)-o(N).

$$

Ignoring only the asymptotically vanishing $o(N)$ term and rearranging,

$$
S
\left(
1-\frac{R}{m}
\right)
\ge
N
\left(
H-\frac{\eta B(m-q)}{m}
\right).

$$

Hence

$$
\kappa
\ge
C
=
\frac{
mH-\eta B(m-q)
}{
m-R
}.

$$

For the frozen strict parameters

$$
q=8,\quad
m=531,\quad
\eta=1,\quad
B=\frac{93}{23000},

$$

$$
H=
\frac{672167187145431}{10^{15}},

$$

$$
R=
\frac{
1113172768314043426732876281699
}{
265625000000000000000000000000
},

$$

the exact assembly gives

$$
C
=
\frac{
722547711262091300265625000000000
}{
1072825050442925667061714615173641
}.

$$

Therefore

$$
\boxed{
\kappa
\ge
0.67350003708785593\ldots
=
67.350003708785593\%.
}

$$

The final exact checker verifies both:

```text
PASS exact C equals frozen fraction
PASS exact C > 0.6735
```

No ordinary floating-point decimal is used as the final acceptance criterion for $C$.

---

## 4. Why this is stronger than “just another numerical candidate”

The repository separates **discovery**, **certification**, and **exact assembly**.

### Discovery

Floating-point optimization is used to locate promising windows, pressure weights, pair weights, slopes, and difficult local regions.

Those numerical searches are treated as **proposal mechanisms only**.

### Certification

The accepted local inequalities are replayed by fail-closed interval branch-and-bound with:

- outward-safe storage of interval tables;
- directed widening at arithmetic boundaries;
- interval Taylor enclosures for the transcendental kernel;
- rigorous first/second-derivative bounds;
- symmetry preflight;
- independently certified difficult wells;
- zero unresolved terminal boxes.

### Exact assembly

The final span identities, pressure sum, envelope evaluation, $R$, and final $C$ are checked over exact rationals.

The committed strict evidence audit reports:

```text
PASS R lock is a valid rational lower bound of the complete kink minimum
PASS exact C equals frozen fraction
PASS exact C > 0.6735
PASS 327/327 strict well boxes proved
PASS constant/rational enclosure audit passed
TOTAL_STRICT_BB_NODES=190375830
RESULT: ALL STRICT 67.35 CHECKS PASS
```

This is why the frozen result is presented as a **strict finite-dimensional computer-assisted certificate**, rather than as a floating-point optimization result.

---

## 5. Development path: 67.25% → 67.275% → 67.350%

The programme deliberately kept a lower-cost certification tier while pursuing the higher-value witness.

| stage | value | role |
|---|---:|---|
| Anthropic public baseline | **67.2500703679%** | public analytic/formal starting point |
| VeriLoop E2 short tier | **67.275055959117140%** | low-cost reproducible certificate; 793,374 B&B nodes |
| VeriLoop E2 strict tier | **67.350003708785593%** | primary high-value strict finite-dimensional certificate |

The short tier is important methodologically: it demonstrates that certification cost is margin-driven. We therefore do not optimize decimals for their own sake; the target threshold is chosen first, and certificate margins are then relaxed as far as exact improvement permits.

The present repository freezes the 67.350003708785593% strict tier as the main result.

---

## 6. Claim boundary — what has been achieved

The following statements are the intended public claims of this repository.

### Achieved

- **Exact frozen witness:**  

$$
  67.350003708785593\%.

$$

- **Strict local finite-dimensional certification:**  
  all three frozen inequalities are fail-closed **PROVED**.

- **Independent difficult-well certification:**  
  **327/327** frozen well boxes are proved, with zero failures.

- **Zero unresolved global boxes:**  
  committed runs terminate with `stack_left=0` and `HARD=0`.

- **Exact-rational assembly:**  
  the final $R$ and $C$ checks do not rely on a rounded decimal comparison.

- **Finite spectral and block-accounting audit:**  
  the spectral inequality, pinching step, and shifted pressure accounting are explicitly audited in this release.

- **Public reproducibility:**  
  verifier source, exact configuration, certificate logs, audit scripts, and reproduction commands are included.

- **Explicit provenance:**  
  the analytic starting point is attributed to Anthropic's public result and Zeta23 formalization rather than being relabelled as original work.

### In progress

The team is actively working on the remaining formal-verification boundary:

1. **End-to-end correspondence with the upstream Zeta23 analytic interface**, including normalization of heights, gaps, multiplicities, and the exact attachment of the strengthened $\Delta(M)$ estimate.
2. **A kernel-friendly short certificate** suitable for Lean replay. The current large branch-and-bound computation is excellent external evidence but is not an efficient object to replay directly inside the Lean kernel.
3. **Lean formalization of the full candidate theorem** under the challenge's fixed theorem statements and permitted axioms.
4. **Independent mathematical review and formal submission** once the end-to-end formal chain is complete.

The current formalization strategy is documented in `lean/LEAN_ROADMAP.md`.

A promising route is to replace the enormous interval tree by a much shorter algebraic object:

- rigorous polynomial lower bounds for the kernel $W$ on the already reduced domain; then
- a Positivstellensatz / sum-of-squares certificate for the resulting polynomial inequalities.

Such an object could potentially be checked by the Lean kernel as a bounded rational computation.

---

## 7. What has **not** been claimed

To keep the scientific boundary explicit:

- this is **not a proof of the Riemann Hypothesis**;
- this is **not** a claim that 67.35% of RH has been solved;
- this is **not yet** an accepted Lean challenge / ledger entry;
- this is **not yet** a claim of completed independent peer review;
- this repository does **not** self-declare mathematical-community acceptance;
- the present strict computer-assisted certificate and an eventual end-to-end Lean theorem are related, but they are **not the same verification object**.

The strongest correct wording at this stage is:

> **A strict, independently reproducible finite-dimensional computer-assisted certificate for the 67.350003708785593% witness, built on Anthropic's public analytic foundation and released for independent audit, formalization, and review.**

---

## 8. Reproduce the result

### Quick committed-evidence audit

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt

./reproduce.sh
```

The quick audit checks the committed evidence, exact constants, exact assembly, terminal certificate states, and the independent well layer.

Expected terminal evidence includes:

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

The full mode rebuilds the strict verifier path and replays the expensive certificate chain from source.

Derived lookup caches are intentionally not treated as trusted evidence; they can be rebuilt.

---

## 9. Repository map

| path | purpose |
|---|---|
| `README.md` | high-level result, derivation, boundaries, reproduction |
| `THEORY.md` | full finite-dimensional reduction and proofs |
| `NUMERICS.md` | search methodology, design decisions, rejected directions |
| `CERTIFICATES.md` | strict certificate summary |
| `AUDIT.md` | finite spectral inequality, pinching, pressure-accounting audit |
| `STATUS.md` | evidence-status ledger |
| `SUBMISSION.md` | comparison with the public baseline and remaining formal boundary |
| `PROVENANCE.md` | provenance of the frozen 67.350003708785593% witness |
| `REFUTATION.md` | explicitly rejected provisional candidates |
| `TIERS.md` | short tier vs strict high-value tier |
| `verify_exact.py` | exact-rational assembly checks |
| `verify_strict.py` | committed strict-certificate audit |
| `audit_constants.py` | safe-direction constant/enclosure audit |
| `arb_window.py` | rigorous window evaluation |
| `verifier/` | strict C++ interval verifiers |
| `certificates/strict/` | raw committed strict logs |
| `config/strictlock.json` | frozen exact strict witness |
| `lean/LEAN_ROADMAP.md` | end-to-end Lean strategy and current formal boundary |
| `lean/Solution.skeleton.lean` | formal theorem contract / proof skeleton |
| `reproduce.sh` | one-command evidence audit and optional full replay |

---

## 10. Research transparency

Computer-assisted mathematics is valuable only if failed candidates are allowed to fail.

During development, higher provisional numerical candidates were rejected after explicit counterexamples exposed insufficiently converged local minima. Those candidates are not silently retained as results; the failure modes are documented in `REFUTATION.md`.

The strict 67.350003708785593% release was rebuilt around a stronger acceptance path:

1. separate numerical discovery from proof;
2. use exact-rational frozen coefficients;
3. certify difficult wells independently;
4. prove the global complement fail-closed;
5. audit constant directions and interval storage;
6. perform final assembly exactly over $\mathbb{Q}$.

This is the evidence discipline behind the frozen number.

---

## 11. Invitation to reproduce, audit, and formalize

Everything required to inspect the finite-dimensional calculation is being released because the result should be challenged, not protected from challenge.

We especially welcome help from researchers working in:

- analytic number theory;
- rigorous interval arithmetic;
- Lean / Mathlib;
- formalized mathematics;
- SOS / Positivstellensatz certificates;
- exact computational proof infrastructure.

If you can shorten the current certificate, formalize the analytic attachment, construct a kernel-checkable algebraic certificate, or independently reproduce/refute any step, please do so.

The next major objective is not another prettier decimal. It is an **end-to-end formal proof object** that can survive independent kernel replay and external mathematical scrutiny.

VeriLoop E2 is intentionally being used here as a research system willing to explore difficult mathematical directions, including directions that may fail. The standard is not whether a hypothesis was ambitious; the standard is whether the final public claim survives evidence, reproduction, and formal checking.

We would be delighted to see independent researchers take this repository, reproduce the result, attempt the formalization, and help push the work toward a successful formal submission.

---

## 12. References and attribution

### Anthropic foundation

**Claude / Anthropic (2026).**  
*More Than Two Thirds of the Zeros of the Riemann Zeta Function Lie on the Critical Line.*  
https://www-cdn.anthropic.com/564f962e60643842f5fcb4a17c9dbc8f608f1c37.pdf

**Anthropic — Zeta23 Lean 4 formalization.**  
https://github.com/anthropics/zeta-23-lean

The present project explicitly builds on this public analytic and formal foundation.

### This release

**VeriLoop E2 + Libo Wang.**  
*Strict 67.350003708785593% finite-dimensional computer-assisted certificate for simple critical-line zeta zeros.*

Author / maintainer: **Libo Wang**

---

## License

See `LICENSE`.

---

### Citation

If you use this artifact, verifier, configuration, or certificate chain, please cite the repository and clearly distinguish:

1. the Anthropic/Zeta23 analytic foundation;
2. the VeriLoop E2 finite-dimensional extension and strict certificate;
3. any subsequent independent formalization, modification, or verification.