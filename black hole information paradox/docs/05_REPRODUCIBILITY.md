# Reproducibility

## Requirements

Python 3.10 or later, with `numpy` and `scipy`.

```bash
pip install -r certificates/requirements.txt
```

## Running the suite

```bash
cd certificates
python3 run_all.py
```

Expected: `SUITE: 6/6 PASS`, exit code `0`. Total runtime under one minute on a laptop. A
reference transcript is stored at `results/run_log.txt`.

## What each certificate checks

| File | Checks |
|---|---|
| `cert_01_sector_connectivity.py` | Theorem 1: `dim(commutant) = #components` against brute-force null-space computation, over eleven configurations including non-adjacent edges, empty graphs and multi-operator families |
| `cert_02_spanning_tree_bound.py` | Corollary 1.1: exhaustive search over all edge subsets confirms `< N` blocks cannot complete and `≥ N` can; the adjacent chain saturates |
| `cert_03_fock_commutant.py` | Theorem 2: commutant dimension `= N_max + 1`. Theorem 3: collapse for four choices of `f`, the `f = 0` control, the injectivity constant of `a†(f)`, and the Weyl-unitary contrast |
| `cert_04_memory_dichotomy.py` | Theorem 4 branch 1: commutant dimension equals sector count, block-scalar to `10⁻¹⁵`, projections central. Contrast with equivalent representations. Theorem 5: `P_C` membership and phase-independence of cross-sector expectation values |
| `cert_05_spectral_gap.py` | Theorem 8: exact spectrum `λ²k(k+1)` to `10⁻¹²`; gap `= 2λ²` at `N = 5,12,30,100`; Fiedler vector `= r − N/2`; adjoint-Casimir identity for `k = 1,2,3`; uniform bound over random `X`. Theorem 6: bound and Fiedler saturation on random multi-mode families. Theorem 7: resistance diameter convergence to `π/λ` |
| `cert_06_scope_guard.py` | Fails the build if any scope label in `results/theorem_status.json` has been weakened |

## Levels of reproduction

**Level 0 — integrity.** Verify `SHA256SUMS.txt` against the shipped files.

```bash
sha256sum -c SHA256SUMS.txt
```

**Level 1 — certificates.** Run the suite as above and compare against `results/run_log.txt`.

**Level 2 — analytic audit.** Read `docs/01_TECHNICAL_REPORT.md` and check each proof by hand.
The certificates cannot substitute for this; they exist to catch errors in the proofs.

**Level 3 — adversarial.** `docs/03_SCOPE_AND_LIMITATIONS.md` §6 lists the load-bearing
assumptions in order of fragility. That is the efficient place to attack.

## Determinism

All certificates use fixed random seeds. Results are reproducible bit-for-bit on the same
numpy/LAPACK build. Minor last-digit variation across BLAS implementations is expected and
does not affect any pass/fail threshold, all of which are set several orders of magnitude
above numerical noise.

## Falsifiability

Each theorem can be refuted by a single counterexample:

- **Theorem 1** — a block algebra plus added family where the commutant dimension differs from
  the component count.
- **Corollary 1.1** — a completing family with fewer than `N` nonzero blocks.
- **Theorem 2** — an element of the commutant that is not a function of `N̂`.
- **Theorem 3** — a nonzero `f ∈ H₁` and an `N` with `P_{N+1}Φ(f)P_N = 0`.
- **Theorem 8** — an eigenvalue of the coherence-transfer Laplacian outside `λ²k(k+1)`, or a
  spectral gap depending on `N`.

Constructing any of these would be a decisive refutation, and the certificates are written to
make such a counterexample easy to test.
