# VeriLoop E2 — Black-Hole Information-Paradox Research Artifact

## Sector Connectivity and the Quantitative Modulus of Asymptotic Graviton Tomography

This directory publishes the **asymptotic-graviton-tomography** research artifact used in the VeriLoop E2 scientific-reasoning program. The public package is organized so that the analytic derivation, executable checks, reference computation log, machine-readable theorem status, reproducibility instructions, and scope controls can be audited independently.

> **Current status.** The artifact establishes an exact and quantitative tomography framework for **free asymptotic radiative graviton data at future null infinity**, within the assumptions stated in the technical report. The shipped certificate suite records **6/6 PASS**.
>
> **Boundary.** This release does **not** solve the black-hole information paradox. It does not yet contain interacting or evaporating dynamics, an initial-to-out faithfulness theorem, a Hawking-radiation calculation, or a proof of evaporation unitarity. The ongoing research program is directed at the missing dynamical layer and the other explicitly listed open problems without weakening these scope constraints.

---

## What has been established

The present artifact closes a substantial **free-asymptotic tomography** chain at both the exact and quantitative levels.

### Exact classification

| Result | Established statement | Status |
|---|---|---|
| **T1 — Sector Connectivity Theorem** | The commutant is the algebra of functions on the connected components of the connectivity graph; informational completeness holds exactly when the graph is connected. | **Proved**; mathematical novelty not claimed |
| **T1.1 — Spanning-tree resource bound** | Any completing family requires at least `N` nonzero off-diagonal blocks; the Stokes connector supplies exactly `N`. | **Proved** |
| **T2 — Exact Fock commutant** | The hard-Fock commutant is `W*(N̂)`, not scalar. | **Proved** |
| **T3 — Linear-connector completion** | For every nonzero `f ∈ H₁`, adjoining `Φ(f)` completes the Fock algebra, with no genericity hypothesis; the physical obstruction is localized at `f_g ∉ H₁`. | **Proved** |
| **T4 — Memory dichotomy** | Both branches of the contested memory-superselection question are classified under one criterion. | **Proved as a dichotomy**; hypothesis (M1) not adjudicated |
| **T5 — Localization of the residue** | Under the stated sector-wise structure, the residual commutant is abelian and the representation is multiplicity-free. | **Proved conditionally** |

### Quantitative completeness

| Result | Established statement | Status |
|---|---|---|
| **T6 — Spectral-gap modulus** | The modulus of informational completeness is controlled by the algebraic connectivity `λ₂` of the coherence-transfer Laplacian, with sharpness on the Fiedler eigenvector. | **Proved** |
| **T7 — Dimension-free bound** | A resistance-diameter bound remains meaningful beyond Hilbert–Schmidt mode truncations. | **Proved** |
| **T8 — Exact `su(2)` spectral gap** | In the two-mode `su(2)` reduction, `spec(L) = λ² k(k+1)` and the gap is exactly `2λ²`, independent of graviton number `N`; the extremal operator is `N_R`. | **Proved** |
| **T9 — Infrared rate reconciliation** | At finite regulator the gap is positive and tends to zero in the infrared limit, giving an order-of-limits structure for the memory controversy. | **Structure proved**; exponent `α` imported, not derived |

The strongest quantitative result is the **dimension-independent gap in T8**: the asymptotic-tomography modulus in the stated two-mode reduction does not degrade with graviton number. The certificate suite checks the exact spectrum, the `N`-independent gap, the Fiedler extremal direction, the adjoint-Casimir identity, the general spectral-gap inequality, and the dimension-free resistance-diameter behavior.

---

## Evidence chain

The release is intentionally organized as an auditable chain rather than as a single narrative document.

```text
Analytic statements and proofs
        ↓
Derivation walkthrough and abandoned routes
        ↓
Executable numerical certificates
        ↓
Reference computation transcript
        ↓
Machine-readable theorem/scope status
        ↓
Cryptographic integrity manifest
```

| Evidence layer | File / directory | Purpose |
|---|---|---|
| **Technical report** | [`research-artifact/docs/01_TECHNICAL_REPORT.md`](research-artifact/docs/01_TECHNICAL_REPORT.md) | Full theorem statements, assumptions, analytic proofs, consolidated result table |
| **Derivation process** | [`research-artifact/docs/02_DERIVATION_WALKTHROUGH.md`](research-artifact/docs/02_DERIVATION_WALKTHROUGH.md) | Step-by-step route to the results, including routes tried and abandoned |
| **Scope and limitations** | [`research-artifact/docs/03_SCOPE_AND_LIMITATIONS.md`](research-artifact/docs/03_SCOPE_AND_LIMITATIONS.md) | Complete claim boundary and load-bearing assumptions |
| **Novelty calibration** | [`research-artifact/docs/04_NOVELTY_AND_PRIOR_WORK.md`](research-artifact/docs/04_NOVELTY_AND_PRIOR_WORK.md) | Conservative novelty assessment and collision-risk table |
| **Reproducibility** | [`research-artifact/docs/05_REPRODUCIBILITY.md`](research-artifact/docs/05_REPRODUCIBILITY.md) | Reproduction levels, determinism, falsifiability, certificate interpretation |
| **Executable certificates** | [`research-artifact/certificates/`](research-artifact/certificates/) | Six independent numerical / structural checks |
| **Reference run** | [`research-artifact/results/run_log.txt`](research-artifact/results/run_log.txt) | Complete reference transcript ending in `SUITE: 6/6 PASS` |
| **Machine-readable status** | [`research-artifact/results/theorem_status.json`](research-artifact/results/theorem_status.json) | Theorem statuses and immutable scope labels |
| **Integrity manifest** | [`research-artifact/SHA256SUMS.txt`](research-artifact/SHA256SUMS.txt) | Byte-level integrity check for the preserved source artifact |
| **Research frontier** | [`research-artifact/prompts/02_RESEARCH_FRONTIER_PROMPT.md`](research-artifact/prompts/02_RESEARCH_FRONTIER_PROMPT.md) | Ranked open problems, attack surface, and standards for extensions |

### Certificate coverage

The six executable certificates are designed to be able to fail rather than merely reproduce expected plots:

1. **Sector connectivity** — brute-force commutant dimensions against the graph-theoretic prediction.
2. **Spanning-tree bound** — exhaustive edge-subset search for the resource lower bound.
3. **Fock commutant and connector completion** — exact commutant dimension, connector collapse, zero-control, injectivity, and contrast.
4. **Memory dichotomy** — disjoint-sector no-go, equivalent-representation contrast, central projections, and cross-sector phase invisibility.
5. **Spectral gap** — exact `λ²k(k+1)` spectrum, `2λ²` gap, `N_R` extremal direction, adjoint-Casimir identity, sharpness tests, and resistance-diameter behavior.
6. **Scope guard** — fails if the machine-readable scope labels are weakened.

The certificates are **verification aids, not substitutes for proof**. The analytic proofs are in the technical report; the executable suite exists to detect inconsistencies and numerical counterexamples to the stated structure.

---

## Reproduce the evidence chain

The original artifact is preserved under `research-artifact/`.

### 1. Integrity

```bash
cd research-artifact
sha256sum -c SHA256SUMS.txt
```

Every shipped source file should report `OK`.

### 2. Certificates

```bash
python3 -m pip install -r certificates/requirements.txt
cd certificates
python3 run_all.py
```

Expected terminal result:

```text
SUITE: 6/6 PASS
```

The reference transcript is stored in [`research-artifact/results/run_log.txt`](research-artifact/results/run_log.txt).

### 3. Analytic audit

Read the complete proofs in [`research-artifact/docs/01_TECHNICAL_REPORT.md`](research-artifact/docs/01_TECHNICAL_REPORT.md) and compare each theorem against the executable checks.

### 4. Adversarial audit

The artifact explicitly documents falsification routes. A counterexample to the graph criterion, the spanning-tree lower bound, the Fock commutant, the connector injectivity statement, or the exact spectral-gap structure would directly challenge the corresponding theorem.

---

## Boundary: what this result does and does not establish

### What is established now

The present work establishes a **closed exact-and-quantitative tomography framework for free asymptotic radiative data** in the stated setting. In particular, it identifies:

- exactly when sector-wise observables become informationally complete;
- the exact hard-Fock residual commutant;
- a connector criterion without a genericity assumption;
- a two-branch classification of the memory layer;
- a sharp spectral modulus for approximate completeness;
- an exact `su(2)` spectral gap `2λ²` independent of graviton number in the stated reduction;
- and an order-of-limits structure for the infrared-memory question.

The complete analytic, computational, machine-readable, and integrity evidence for these statements is included in this repository.

### What is not established yet

The present artifact does **not** establish:

- a solution of the black-hole information paradox;
- initial-to-out faithfulness;
- interacting or evaporating dynamics;
- a Hawking-radiation derivation;
- unitarity of black-hole evaporation;
- a resolution of the contested memory-superselection hypothesis (M1);
- a first-principles derivation of the infrared exponent `α`;
- a universal `2λ²` gap for arbitrary multi-mode connectors;
- or world priority for the results.

These are substantive boundaries, not editorial caveats.

### What is being pursued next

The ongoing research program is focused on the missing layers identified by the artifact itself:

1. **Derive the infrared exponent `α` from first principles** for a specified gravitational dressing geometry.
2. **Generalize the spectral-gap analysis to genuinely multi-mode connectors** and determine whether a mode-count-independent lower bound survives.
3. **Replace the discrete memory label by the physical direct-integral continuum formulation.**
4. **Extend the Laplacian framework beyond exactly block-diagonal base algebras**, a prerequisite for an interacting setting.
5. **Develop a dynamical analogue of the completeness modulus** capable of addressing information transfer from initial data to outgoing radiation.

The last item is the genuine bridge that would be required before claims about interacting/evaporating black holes or the information paradox itself become justified. Until that bridge is established, the current claims remain deliberately restricted to free asymptotic radiative data.

---

## Integrity and source preservation

The directory [`research-artifact/`](research-artifact/) is a **byte-for-byte preserved copy of the supplied `asymptotic-graviton-tomography.zip` payload**. The publication process does not rewrite the technical report, derivation walkthrough, scope document, novelty assessment, reproducibility document, prompts, certificates, reference run, theorem-status record, citation metadata, license, or original README.

The top-level README you are reading is a publication index and scope summary. It does not replace the source artifact.

---

## Citation

The preserved citation metadata is available at:

[`research-artifact/CITATION.cff`](research-artifact/CITATION.cff)

**Author:** Libo Wang  
**Original artifact title:** *Sector Connectivity and the Quantitative Modulus of Asymptotic Graviton Tomography*  
**Release date:** 14 September 2026  
**License:** CC BY 4.0

---

## Release principle

**What is proved is stated positively. What is not yet proved is stated explicitly. The next research phase begins exactly at that boundary.**
