# VeriLoop E2 — Riemann Zeta Evidence Artifacts

This directory contains the frozen computational evidence accompanying the
VeriLoop E2 `67.350003708785593%` release.

The publication layout separates the public evidence package, the fuller
internal evidence package, and independently generated integrity manifests.

## Layout

```text
evidence/
├── public/
└── internal/

manifests/
├── SOURCE_ARCHIVES.sha256
├── PUBLIC_FILES.sha256
├── INTERNAL_FILES.sha256
└── RELEASE_INVENTORY.txt
```

## Public evidence

`evidence/public/` is a byte-preserving extraction of:

```text
zeta_public_67.350003708785593_strict.zip
```

It contains the externally inspectable release material.

## Internal / full evidence

`evidence/internal/` is a byte-preserving extraction of:

```text
zeta_internal_67.350003708785593_strict.zip
```

It preserves the fuller computational artifact set supplied with the frozen
release.

## Integrity policy

The publication process does not rewrite, reformat, rename, recompute, or
semantically normalize any extracted evidence file.

Repository organization is an external packaging layer only.

SHA-256 manifests are generated independently for every extracted regular file:

```text
manifests/PUBLIC_FILES.sha256
manifests/INTERNAL_FILES.sha256
```

The original source archives are also fingerprinted in:

```text
manifests/SOURCE_ARCHIVES.sha256
```

## Relationship to the main README

The repository-level `README.md` states the mathematical claim, the Anthropic
public starting point, the VeriLoop E2 finite-dimensional extension, the exact
assembly, the current verification boundary, and the reproduction policy.

This artifact index only organizes the frozen computational evidence.

## Verification boundary

Publication of these artifacts does not by itself constitute an end-to-end
Lean proof.

The strict finite-dimensional computer-assisted certificate and its exact
assembly are released for independent reproduction, audit, attempted
refutation, and formalization.

## Attribution

**Research system:** VeriLoop E2  
**Author / maintainer:** Libo Wang
