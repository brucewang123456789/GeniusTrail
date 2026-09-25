# VeriLoop E2 — Black-Hole Information-Paradox Research Track

## Asymptotic Graviton Tomography: Exact Classification, Quantitative Completeness, and Reproducible Evidence

This directory publishes the **asymptotic-graviton-tomography** research artifact from the VeriLoop E2 scientific-reasoning program.

The purpose of this release is to make the completed work independently inspectable as a single evidence chain: **analytic statements → derivation process → executable certificates → computation transcript → machine-readable theorem status → explicit scope controls**.

> **Scope first.** All established results in this artifact concern **free asymptotic radiative graviton data at future null infinity**. There is no interaction, no dynamical evaporation model, and no initial-to-out map in the present work. **The black-hole information paradox is not solved.**
>
> Within that boundary, the artifact establishes a closed exact-and-quantitative tomography framework with nine stated results and a six-certificate verification suite.

---

## What has been achieved

The work advances asymptotic tomography in two layers.

### 1. Exact classification

The artifact gives an exact graph-theoretic criterion for informational completeness of a sector-block observable algebra augmented by connector observables. It then applies that structure to the hard-Fock layer and the infrared-memory layer.

| Result | Established result | Status |
|---|---|---|
| **T1** | Sector Connectivity Theorem: the commutant is the algebra of functions on connected components; informational completeness holds iff the connectivity graph is connected. | **PROVED**; mathematical novelty not claimed |
| **T1.1** | Any completing family requires at least `N` nonzero off-diagonal blocks; the Stokes connector supplies exactly `N`. | **PROVED** |
| **T2** | Exact hard-Fock commutant `= W*(N̂)`, not scalar. | **PROVED** |
| **T3** | For every nonzero `f ∈ H₁`, one linear observable completes the Fock algebra; no genericity hypothesis is required. | **PROVED** |
| **T4** | Both branches of the infrared-memory superselection question are classified without adjudicating the contested hypothesis (M1). | **PROVED AS DICHOTOMY** |
| **T5** | Under the stated sector-wise structure, the residual commutant is abelian and the representation is multiplicity-free. | **PROVED CONDITIONALLY** |

### 2. Quantitative completeness

The exact zero/nonzero criterion is upgraded to a quantitative modulus.

| Result | Established result | Status |
|---|---|---|
| **T6** | The modulus of informational completeness is controlled by the algebraic connectivity `λ₂` of the coherence-transfer Laplacian, sharp on the Fiedler eigenvector. | **PROVED** |
| **T7** | A resistance-diameter operator-norm bound gives a dimension-free continuum-compatible control. | **PROVED** |
| **T8** | In the stated two-mode `su(2)` reduction, `spec(L)=λ²k(k+1)` and the spectral gap is exactly `2λ²`, independent of graviton number `N`; the extremal operator is `N_R`. | **PROVED** |
| **T9** | At finite infrared regulator the gap is positive and tends to zero in the infrared limit, giving an order-of-limits structure for the memory controversy. | **STRUCTURE PROVED**; exponent `α` imported, not derived |

### Three central results

The original artifact identifies three claims as the main technical statements worth defending:

> The modulus of informational completeness is the algebraic connectivity of the coherence-transfer Laplacian, sharp on the Fiedler eigenvector.

> For the spin-4 Stokes connector that Laplacian is half an `su(2)` adjoint Casimir, so the spectral gap is exactly `2λ²` and does not degrade with graviton number.

> The two opposing positions on infrared memory superselection differ by an order of limits, and the spectral gap is the quantity that interpolates between them.

---

## Complete evidence chain

No derivation, computation, certificate, result record, or scope document is rewritten for this GitHub release. The original artifact is preserved byte-for-byte under [`research-artifact/`](research-artifact/).

```text
research-artifact/
├── README.md
├── CITATION.cff
├── LICENSE
├── SHA256SUMS.txt
├── docs/
│   ├── 01_TECHNICAL_REPORT.md
│   ├── 02_DERIVATION_WALKTHROUGH.md
│   ├── 03_SCOPE_AND_LIMITATIONS.md
│   ├── 04_NOVELTY_AND_PRIOR_WORK.md
│   └── 05_REPRODUCIBILITY.md
├── certificates/
│   ├── cert_01_sector_connectivity.py
│   ├── cert_02_spanning_tree_bound.py
│   ├── cert_03_fock_commutant.py
│   ├── cert_04_memory_dichotomy.py
│   ├── cert_05_spectral_gap.py
│   ├── cert_06_scope_guard.py
│   ├── requirements.txt
│   └── run_all.py
├── results/
│   ├── run_log.txt
│   └── theorem_status.json
└── prompts/
    ├── 01_CORE_KNOWLEDGE_PROMPT.md
    ├── 02_RESEARCH_FRONTIER_PROMPT.md
    └── 03_SCOPE_CALIBRATION_PROMPT.md
```

### Evidence map

| Evidence layer | File | What it contains |
|---|---|---|
| **Complete analytic result** | [`docs/01_TECHNICAL_REPORT.md`](research-artifact/docs/01_TECHNICAL_REPORT.md) | Full statements, assumptions, proofs, and result consolidation |
| **Derivation process** | [`docs/02_DERIVATION_WALKTHROUGH.md`](research-artifact/docs/02_DERIVATION_WALKTHROUGH.md) | Step-by-step derivation, including routes attempted and abandoned |
| **Claim boundary** | [`docs/03_SCOPE_AND_LIMITATIONS.md`](research-artifact/docs/03_SCOPE_AND_LIMITATIONS.md) | Complete limitations and load-bearing assumptions |
| **Novelty calibration** | [`docs/04_NOVELTY_AND_PRIOR_WORK.md`](research-artifact/docs/04_NOVELTY_AND_PRIOR_WORK.md) | Conservative novelty assessment and collision-risk analysis |
| **Reproducibility** | [`docs/05_REPRODUCIBILITY.md`](research-artifact/docs/05_REPRODUCIBILITY.md) | How to reproduce, audit, and falsify the result chain |
| **Executable evidence** | [`certificates/`](research-artifact/certificates/) | Six numerical / structural certificates designed to be able to fail |
| **Computation transcript** | [`results/run_log.txt`](research-artifact/results/run_log.txt) | Reference execution transcript |
| **Machine-readable result state** | [`results/theorem_status.json`](research-artifact/results/theorem_status.json) | T1–T9 status, scope labels, and explicit non-claims |
| **Integrity record** | [`SHA256SUMS.txt`](research-artifact/SHA256SUMS.txt) | Byte-level integrity of the preserved artifact |
| **Research frontier** | [`prompts/02_RESEARCH_FRONTIER_PROMPT.md`](research-artifact/prompts/02_RESEARCH_FRONTIER_PROMPT.md) | Open problems, attack surface, and standards for any extension |

---

## Computational and certificate closure

The artifact ships six executable certificates covering the main structural and quantitative claims.

1. **Sector connectivity** — compares explicit commutant dimensions against the graph prediction.
2. **Spanning-tree resource bound** — checks the minimum off-diagonal resource needed for completion.
3. **Fock commutant and connector completion** — checks the exact commutant and connector collapse structure.
4. **Memory dichotomy** — checks both sector branches rather than assuming one side of the contested premise.
5. **Spectral-gap structure** — checks the exact `λ²k(k+1)` spectrum, the `2λ²` gap, the `N`-independent behavior, the Fiedler extremal direction, and the dimension-free resistance-diameter behavior.
6. **Scope guard** — fails if the machine-readable claim labels are weakened.

The preserved reference transcript reports:

```text
SUITE: 6/6 PASS
```

The numerical certificates are **verification aids, not substitutes for the analytic proofs**. Passing the suite means the supplied internal consistency checks did not detect an error; it is not represented as external validation.

---

## Reproduce

From the preserved artifact:

```bash
cd research-artifact
sha256sum -c SHA256SUMS.txt

python3 -m pip install -r certificates/requirements.txt
cd certificates
python3 run_all.py
```

Expected suite result:

```text
SUITE: 6/6 PASS
```

The full reference transcript is available at [`results/run_log.txt`](research-artifact/results/run_log.txt).

---

## Boundary and current research direction

The scope boundary is part of the result, not an editorial disclaimer.

### Established now

The present artifact establishes, within free asymptotic radiative data:

- an exact connectivity criterion for sector-wise informational completeness;
- an exact hard-Fock commutant;
- a connector-completion theorem without a genericity hypothesis;
- a two-branch classification of the contested memory-superselection layer;
- a sharp spectral modulus for approximate informational completeness;
- an exact `2λ²` spectral gap independent of graviton number in the stated two-mode `su(2)` reduction;
- and an order-of-limits structure for the infrared-memory question.

### Not established in this release

The present artifact does **not** establish:

- a solution, partial solution, or proof of the black-hole information paradox;
- initial-to-out faithfulness;
- interacting or evaporating black-hole dynamics;
- a Hawking-radiation derivation;
- unitarity of black-hole evaporation;
- a resolution of the contested memory-superselection hypothesis (M1);
- a first-principles derivation of the infrared exponent `α`;
- a universal `2λ²` gap for arbitrary multi-mode connectors;
- or a world-priority claim.

### Current work

The next research stage is focused exactly on the gaps identified by the preserved research-frontier document:

1. **Deriving the infrared exponent `α` from first principles** for a specified gravitational dressing geometry.
2. **Extending the spectral-gap analysis to genuinely multi-mode connectors**, including whether a mode-count-independent lower bound survives.
3. **Replacing the discrete memory label with the physical direct-integral continuum formulation.**
4. **Extending the Laplacian framework beyond exactly block-diagonal base algebras**, which the artifact identifies as a prerequisite for an interacting setting.
5. **Developing a dynamical analogue of the completeness modulus**, asking whether a scattering map can connect initial data to outgoing radiation through a comparable quantitative structure.

The fifth item is the substantive bridge that would be required before the present kinematical framework could support claims about dynamical information recovery in an evaporating black-hole setting.

---

## Integrity policy

This GitHub publication follows a strict preservation rule:

- the source ZIP is read-only;
- the extracted research artifact is copied without editing its contents;
- all source files are checked against the artifact's own `SHA256SUMS.txt`;
- the staged `research-artifact/` tree is compared byte-for-byte against the extracted source tree;
- the publication README is the only new narrative layer;
- no original derivation, proof, computation, certificate, result record, prompt, citation file, license, or source README is rewritten.

---

## Citation

See [`research-artifact/CITATION.cff`](research-artifact/CITATION.cff).

**Author:** Libo Wang  
**Artifact:** *Sector Connectivity and the Quantitative Modulus of Asymptotic Graviton Tomography*  
**License:** CC BY 4.0

---

**The completed result is presented positively where the evidence closes. The boundary is stated exactly where the evidence stops. The next research stage begins at that boundary.**
