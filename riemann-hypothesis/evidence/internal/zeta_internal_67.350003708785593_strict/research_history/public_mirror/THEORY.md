# The finite-dimensional reduction, stated in full

Author: Libo Wang.

Every step below carries one of five labels.

| label | meaning |
|---|---|
| **IMPORTED** | stated here, not reproved in this repository, and not audited |
| **PROVED** | proved in this document |
| **CERTIFIED** | established by rigorous interval arithmetic (Arb, outward rounding) |
| **EXACT** | exact arithmetic over **Q**; no floating point enters the acceptance test |
| **FLOAT** | numerical evidence only; never sufficient to assert a statement is true |

---

## 0. Object

Let `N(T)` count nontrivial zeros of the Riemann zeta function with `0 < Im rho <= T`,
and let `S(T)` count those that are **simple and on the critical line**. The quantity
being bounded is

    kappa = liminf_{T -> inf} S(T) / N(T).

This is a proportion of zeros, **not** a percentage of the Riemann Hypothesis being
proved. Nothing in this repository bears on the truth of RH.

---

## 1. The imported analytic interface (D0)   — **IMPORTED**

Fix an even window `v` supported on `[-1/2, 1/2]` with `v >= 0`, and set

    K(x) = integral_{-1/2}^{1/2} v(t) cos(2 pi x t) dt        (Fourier transform of v)
    I1   = integral v,      I2 = integral v^2,
    J    = double integral |s - t| v(s) v(t) ds dt
    H(v) = 2 - (I2 + J) / I1^2 .

D0 asserts the existence, for each height `T`, of a Hermitian positive semidefinite
matrix `M = M(T)` of size `N = N(T)` with unit diagonal, whose entries are asymptotically

    M_{ij} ~ K(x_i - x_j) / K(0)

for a point configuration `x_1 <= ... <= x_N` on the rescaled zero line, such that

    S  >=  H(v) * N  +  Delta(M)  -  o(N),                                   (D0-1)

where `Delta` is the spectral functional of section 2. D0 is the pair-correlation /
explicit-formula layer of the programme. **It is stated, not reproved, and not audited
here.** Every result below is conditional on it.

Sanity check on D0-1: for `v = 1` one computes `I1 = I2 = 1`, `J = 1/3`, hence
`H = 2 - 4/3 = 2/3`. Dropping `Delta(M) >= 0` recovers the classical `kappa >= 2/3`.
So the whole point of the machinery below is to extract a strictly positive linear-in-`N`
lower bound for `Delta(M)`.

---

## 2. The spectral functional   — **PROVED**

    Psi(t) = (t - 1)^2      for 0 <= t <= 2
    Psi(t) = 2t - 3         for t >= 2
    Delta(G) = trace Psi(G) = sum over eigenvalues.

**Lemma 2.1.** For Hermitian `G` with `X = G - I` and `U = (G - 2I)_+`
(the positive part of `G - 2I`),

    Delta(G) = || X - U ||_F^2 + 2 trace U .

*Proof.* Both sides are spectral. For an eigenvalue `lambda <= 2` the right side is
`(lambda-1)^2`. For `lambda > 2` it is `(lambda - 1 - (lambda - 2))^2 + 2(lambda - 2)
= 1 + 2 lambda - 4 = 2 lambda - 3`. These match `Psi`. []

---

## 3. The matrix theorem   — **PROVED**, general `(d, r)`

Fix a retained bandwidth `q >= 1`. On the index set `{1, ..., m}` let `Gamma` be the graph
with `i ~ j` iff `0 < |i - j| <= q`. Then

* max degree `d = 2q`;
* residues modulo `q + 1` give a proper colouring, so the chromatic number is at most
  `r = q + 1` (and the clique number is exactly `q + 1`, so this is optimal);
* set `T = r / (r - 1) = (q + 1) / q`.

For `G` positive semidefinite with unit diagonal define the retained energy

    E = 2 * sum over 1 <= j - i <= q of |G_{ij}|^2 .

**Theorem 3.1.**

    Delta(G)  >=  h(E),     where
    h(E) = E                                        for 0 <= E <= T,
    h(E) = E - (d / (d + 1)) * (sqrt E - sqrt T)^2  for E >= T.

*Proof.* Write `z_i = U_{ii} >= 0`, `D = sum_i z_i^2`, `tau = trace U = sum_i z_i`, and
`u = || (X - U) restricted to the retained band ||` measured as below. Since `X` has zero
diagonal and `U` is positive semidefinite, Lemma 2.1 gives

    Delta(G) >= D + (sqrt E - u)^2 + 2 tau                                    (3.1)

where `u` is the Frobenius norm of the retained band of `U`. Two bounds on `u`:

(i) **Degree bound.** Each off-diagonal entry of a positive semidefinite matrix obeys
`|U_{ij}|^2 <= U_{ii} U_{jj} = z_i z_j`, so
`u^2 <= sum_{i ~ j} z_i z_j <= sum_i deg(i) z_i^2 <= d * D`, i.e. `D >= u^2 / d`.

(ii) **Colouring bound.** With colour classes `V_1, ..., V_r` and masses
`c_a = sum_{i in V_a} z_i`, no edge lies inside a class, so
`u^2 <= sum_{a != b} c_a c_b = tau^2 - sum_a c_a^2 <= tau^2 (1 - 1/r) = tau^2 / T`,
i.e. `tau >= sqrt T * u`. (The bound `sum_a c_a^2 >= tau^2 / r` is Cauchy-Schwarz; it is
attained exactly when `z` is uniform on a maximum clique, which is why `r = q + 1` is the
right constant.)

Substituting both into (3.1),

    Delta(G) >= E - 2 sqrt E u + u^2 + u^2/d + 2 sqrt T u
             = E + (1 + 1/d) u^2 - 2 (sqrt E - sqrt T) u .

Minimising the right side over `u >= 0` gives `u* = (d/(d+1)) (sqrt E - sqrt T)_+` and the
stated value. For `E <= T` the minimiser is `u* = 0` and the bound is `Delta >= E`. []

**Remark 3.2 (tightness, and why improving `h` is worthless at the present witness).**
For `E <= T` the bound `Delta(G) >= E` is attained: take `G = I + X` with `X` supported on
the retained band, `||X||_op <= 1`, so all eigenvalues lie in `[0, 2]`, `U = 0`, and
`Delta = ||X||_F^2 = E`. In every witness produced by this programme the minimising `E`
lies **strictly below `T`**. Consequently sharpening `h` on `E > T` changes nothing, and
effort spent there is wasted. This was verified numerically for the `q = 6`, `q = 7` and `q = 8` lines.

**Remark 3.3 (the slack in (i) and (ii) cannot both be removed).**
On a maximum clique with uniform `z`, bound (ii) is tight while (i) is loose by a factor
of exactly 2. A joint bound tight on cliques is *false* for spread-out `z` (uniform `z` on
`k` consecutive indices satisfies `u^2 -> d D` as `k -> inf`), so the pair (i)+(ii) is the
correct compromise. This closes an otherwise tempting line of attack.

---

## 4. Local certificates

Let `W(x) = (K(x) / K(0))^2`, so `W(0) = 1` and `|G_{ij}|^2 -> W(x_j - x_i)`.

Fix nonnegative **pressure weights** `b_1, ..., b_q` and **pair weights**
`a_{ij} >= 0` for `0 <= i < j <= q`, subject to the **span-capacity identity**

    for every span s = 1, ..., q:    sum_{i=0}^{q-s} a_{i, i+s} = 2 .          (SC)

For gaps `g_1, ..., g_q >= 0` put `y_0 = 0`, `y_j = g_1 + ... + g_j`, and

    P_loc(g) = sum_{r=1}^{q} b_r g_r ,
    Q_loc(g) = sum_{0 <= i < j <= q} a_{ij} W(y_j - y_i) .

**Certificate C(s).** For a slope `s > 0`, the statement to be proved is

    P_loc(g) + s * Q_loc(g)  >=  epsilon_s      for all g in [0, inf)^q .      (C(s))

This is a `q`-dimensional global minimisation over an unbounded orthant. It is decidable by
exhaustive interval branch-and-bound because the domain reduces: since `Q_loc >= 0`,
any `g` with `g_r > epsilon_s / b_r` already satisfies `P_loc >= epsilon_s`. Hence the
search region is contained in the simplex `{ sum_r b_r g_r <= epsilon_s }`.

**In this repository the three `epsilon_s` of the witness are CERTIFIED.** Each was proved by
exhaustive interval branch-and-bound over 16 disjoint root shards, every shard fail-closed
with zero unresolved boxes; see `CERTIFICATES.md`. `REFUTATION.md` records two previously
circulated candidates whose `epsilon_s` were false, and why.

---

## 5. Block assembly   — **PROVED** modulo the inherited averaging step

Partition (with offsets) the index line into blocks of length `m`. A block contains
`n = m - q` translated local windows. Let `P` be the total pressure accumulated in the
block and `E` the retained energy of its Gram submatrix `G`.

**Span accounting.** Summing `Q_loc` over the `n` translates, the pair `(i, j)` at span
`s = j - i` receives total weight `sum_i a_{i,i+s} = 2` by (SC), which is exactly its
weight in `E = 2 * sum_{1 <= j-i <= q} |G_{ij}|^2`. Boundary windows can only lose terms,
so

    sum over translates of Q_loc  <=  E .                                      (5.1)

Summing `C(s)` over the `n` translates and applying (5.1),

    P + s E  >=  n * epsilon_s        for every certified slope s.             (5.2)

Therefore, with `p(E) = max(0, max_s ( n epsilon_s - s E ))`,

    P >= p(E),    and p is convex, nonincreasing, piecewise linear in E.       (5.3)

**Finite inequality.** For any multiplier `eta >= 0` define

    R = inf_{E >= 0} [ h(E) + eta * p(E) ] .                                   (5.4)

Then `Delta(G) + eta P >= R` for every block. Because `h` is concave and `p` is piecewise
linear, `h + eta p` is concave on each linear piece of `p`; the infimum is therefore
attained at a kink of `p`, at `E = T`, at `E = 0`, or where `p` first vanishes. This is a
**finite** set, so (5.4) is evaluated exactly. — **EXACT**

**Averaging (inherited).** Summing the finite inequality over blocks and offsets, using
that the total pressure charge attributable to a single gap is at most `(m - q) B` with
`B = sum_r b_r`, and that total gap length is at most `N`, the framework yields

    Delta(M)  >=  (R / m) * S  -  (eta * B * (m - q) / m) * N .                (5.5)

The combinatorial averaging in (5.5) is inherited from the `q = 6` predecessor and is
**IMPORTED** here; it is unchanged by the move to `q = 7`.

**Conclusion.** Combining (D0-1) with (5.5),

    S >= H N + (R/m) S - (eta B (m-q)/m) N
    => S (1 - R/m) >= N (H - eta B (m-q)/m)
    => kappa >= C := ( m H - eta B (m - q) ) / ( m - R ),      provided m > R. (5.6)

---

## 6. Exact evaluation of `R`   — **EXACT**

`h` is irrational for `E > T`. Writing `d = 2q` and `T = (q+1)/q`,

    h(E) = E/(d+1) + (2d/(d+1)) sqrt(E T) - (d/(d+1)) T ,

so a rational lower bound for `h(E)` follows from a rational lower bound for `sqrt(E T)`,
obtained by integer square root at denominator `10^30`. Equivalently, `h(E) >= Dv` holds
iff `E T > L^2` with `L = ((d+1) Dv - E + d T) / (2d)` whenever `L > 0`. For `q = 8` (`d = 16`), the witness of this repository, this is `32 sqrt(E T) >= 17 Dv - E + 16 T`; for `q = 7` (`d = 14`),
`28 sqrt(E T) >= 15 Dv - E + 14 T`; for `q = 6` (`d = 12`), `24 sqrt(E T) >= 13 Dv - E + 12 T`,
which reproduces the predecessor's test. The theorem of section 3 is proved for general
`(d, r)`, so raising `q` requires no new proof — only harder certificates.

No floating-point value enters any acceptance decision in `verify_exact.py`.

---

## 7. What is open

1. The three certificates of section 4 are closed, but their arithmetic is outward-rounded
   IEEE double with rigorous second-order table remainders. A submission-grade artefact should
   replay all 48 shards in directed MPFR or Arb. The argument is unchanged by this; only the
   width of the arithmetic is.
2. D0 (section 1) is **IMPORTED**, but now with identified public provenance: it is the
   analytic layer of the unconditional, Lean-formalized 2026 result. See `SUBMISSION.md`.
3. The averaging step (5.5) and the spectral inequality (B) are **PROVED** in `AUDIT.md`.

Items 1-3 are independent. The finite-dimensional layer is now closed; 2 and 3 are not.
