# Scope and Limitations

This document exists so that no reader has to infer the boundaries of the work. Everything
below is a limitation the author states voluntarily.

## 1. The headline restriction

**All results concern free asymptotic radiative data.** There is no interaction, no dynamics,
no evaporation model, and no map from initial data to outgoing data anywhere in this work.
Every theorem is a statement about the observable algebra of free asymptotic radiation at
future null infinity.

## 2. The black-hole information paradox is not solved

No part of it is. The word "paradox" appears in this repository only in statements of what has
*not* been achieved. Specifically:

- No initial-to-out faithfulness is proved.
- No statement about Hawking radiation is made or implied.
- No statement about unitarity of evaporation is made or implied.
- Theorem 5 is *consistent with* unitarity. Consistency is not proof, and the distinction is
  not rhetorical.

## 3. Theorem-by-theorem limitations

**Theorem 1.** Mathematically elementary. In the finite-dimensional setting it is close to
folklore: block-diagonal von Neumann algebra structure theory together with connectedness of
an inclusion graph. No claim of mathematical novelty is made. Its value is exactness and
unification.

**Theorem 2.** Depends on Proposition 1 and Proposition 2, which are inputs rather than
results of this work.

**Theorem 3.** Holds for every `f ∈ H₁`. The physical soft charge has `f_g ∉ H₁`, so the theorem
does **not** apply to it. This is stated as a feature — it localizes the obstruction — but it
means no completion of the physical soft sector is achieved.

**Theorem 4.** Hypothesis (M1) is contested in the current literature and is deliberately not
adjudicated. Both branches are proved. A reader who believes one branch should read only that
branch; the other is then irrelevant to them.

**Theorem 5.** Conditional on the sector-wise structure established by Theorems 1–3. The
direct-sum treatment of the memory label is an idealization; the physical memory space is a
continuum and the general statement requires a direct integral with a standard Borel
structure on the memory space. The discrete version is what is proved.

**Theorem 6.** Uses Hilbert–Schmidt norms and therefore requires a mode truncation. Theorem 7
exists precisely to cover the continuum case.

**Theorem 8.** The exact value `λ₂ = 2λ²` uses the two-mode `su(2)` reduction of the Stokes
connector. For a general `κ` acting across many modes, Theorem 6 applies with the
corresponding `λ₂`, and the bound `D_R ≤ π/λ` of Theorem 7 still holds because block operator
norms only increase when more modes are available. **The exact value should always be quoted
as the `su(2)`-reduction value**, never as a universal constant.

**Theorem 9.** Only the structure is proved. The exponent `α` is imported from the dressing
literature and depends on dressing geometry. Any specific numerical rate must be derived
separately for the setup in question.

## 4. On numerical certificates

The certificates verify finite-dimensional models and truncations. They are designed to be
able to fail — with controls, contrasts, exhaustive searches, and sharpness checks — but they
are **not** proofs. Every theorem in the technical report carries an analytic proof; the
certificates exist to catch errors in those proofs, not to substitute for them.

Passing the certificate suite is not external validation. It means the internal consistency
checks the author could devise did not detect an error.

## 5. On priority

**No claim of world priority is made.** The following are all standard and are used, not
claimed: the commutant/bicommutant criterion, block-algebra structure theory, the factor
dichotomy (quasi-equivalent or disjoint), the multiplicity-free/abelian-commutant equivalence,
Fiedler's algebraic connectivity bound, and the Schwinger boson realization of `su(2)`.

Targeted literature searches did not find a direct prior statement of the combined results.
**That is not evidence of priority.** It is evidence that the searches did not find one. The
relevant literature is large, active, and partly contemporaneous.

## 6. How to criticize this work efficiently

For a reader who wants to attack it, the load-bearing points are:

1. **Propositions 1 and 2 are inputs.** If either fails, most of Part II fails with it.
2. **Theorem 2 assumes `P_N` lies in the algebra**, via the configuration-space multiplication
   algebra. If that spectral argument is wrong, Theorem 2 is wrong.
3. **Theorem 5's direct-sum idealization.** The continuum version needs measurability structure
   that is asserted rather than constructed here.
4. **Theorem 8's two-mode reduction.** If the physically relevant `κ` cannot be reduced this
   way, the exact constant does not apply, though the general bounds still do.
5. **Theorem 9's exponent.** Imported. If the dressing-literature behaviour is different, the
   rate changes — though the structural statement does not.

These are the places to look. They are listed here rather than left to be discovered.
