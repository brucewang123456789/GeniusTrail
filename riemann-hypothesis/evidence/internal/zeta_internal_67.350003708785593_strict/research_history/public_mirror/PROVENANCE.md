# Provenance

## Lineage of the numbers

| stage | value | certification state | artefacts in this repository? |
|---|---|---|---|
| `q = 6` predecessor | `67.34574870891890%` | local certificates completed by an exhaustive interval verifier in earlier work | **no** — only the value is carried forward |
| `q = 7`, six-slope | `67.350352375073%` | **refuted** | refutation only |
| `q = 7`, Reduced-3 | `67.35006335392536%` | **refuted** | refutation only |
| `q = 7`, this work | `67.348184885262%` | candidate; matrix theorem proved, `H` certified, assembly exact, `epsilon_s` uncertified | yes, in full |

## What is *not* here

The raw certificate artefacts of the `q = 6` run (shard logs, node counts, MPFR tables)
were produced in earlier work and are not part of this repository. The `q = 6` value is
therefore cited, not re-established here.

## Integrity

`MANIFEST.sha256` lists the SHA-256 of every tracked file. Regenerate and compare with

    sha256sum -c MANIFEST.sha256

Only files authored and tested as part of this repository are included. Any file not listed
in the manifest is not part of the release.
