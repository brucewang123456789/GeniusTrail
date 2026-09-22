# VeriLoop zeta — strict 67.350003708785593% certificate artifact

Author: **Libo Wang**. Frozen 2026-09-17.

This release contains a fail-closed finite-dimensional computer-assisted certificate for the frozen q=8 witness

`C = 722547711262091300265625000000000 / 1072825050442925667061714615173641`

`= 67.350003708785593%`.

The historical float-storage concern is removed from the acceptance path: lower/upper float tables are explicitly rounded outward, the transcendental kernel uses interval Taylor enclosures, and the difficult local wells are separately proved by an directed interval kernel initialized from exact-rational coefficients before the well-aware B&B for `s=19/20` and `s=1` is allowed to prune them. The `s=1/2` committed certificate is the stronger direct strict B&B and does not use the well shortcut.

## Evidence gates

- 327/327 convex well boxes: **PROVED**, 0 failed.
- `s=1/2`, `s=19/20`, `s=1`: **3/3 fail-closed PROVED**, zero HARD/unresolved boxes.
- Total global B&B nodes in the committed strict logs: **190,375,830**.
- Exact span identities, envelope `R`, and final `C`: exact rational arithmetic, PASS.
- Constant-direction audit: rational coefficients are enclosed in the safe direction, PASS.

Run `./reproduce.sh` for the quick committed-evidence audit. Run `FULL=1 ./reproduce.sh` for a clean rebuild and full strict replay. Derived lookup caches are intentionally absent from the public package.

## Claim boundary

This is a public, independently reproducible **finite-dimensional computer-assisted proof artifact**. It does not claim that external peer review has already accepted it, and it is not an accepted Lean leaderboard entry. The upstream analytic attachment and formal submission status are documented in `STATUS.md` and `SUBMISSION.md`.
