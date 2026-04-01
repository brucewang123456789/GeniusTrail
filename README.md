# VeriLoop Coder

**Evidence-governed harness engineering for code-capable foundation models.**

> **Project status**
>
> **VeriLoop Coder is still under construction.**
>
> This branch should be understood primarily as a **research and contribution branch for publicly obtainable coding-harness materials, reference implementations, and architecture-aligned notes**, not as the final released VeriLoop Coder product.
>
> At the current stage, this repository mainly serves four purposes:
>
> 1. to preserve and organize publicly accessible coding-agent and harness-related reference materials;
> 2. to study production-grade coding-harness patterns that became visible through public Claude Code materials and the widely circulated source-map-derived code incident;
> 3. to incorporate lessons from the MIT-licensed **shareAI-lab / learn-claude-code** harness-engineering pedagogy;
> 4. to prepare the future engineering substrate for **VeriLoop Coder** as a distinct, original, evidence-governed coding system.

---

## What This Repository Is Right Now

This repository is **not yet the final VeriLoop Coder runtime**.

At present, it should be understood as a **curated research workbench** for future VeriLoop coding infrastructure. The current branch contains a mixture of:

- third-party public reference materials,
- publicly circulated source-map-derived code resources collected from the open internet for research and comparative study,
- MIT-licensed harness-engineering learning materials,
- repository-local documentation, tests, web assets, and scaffolding that align these resources with VeriLoop's future architecture.

In other words, **the current value of this branch is not that VeriLoop Coder is already complete**, but that it establishes a serious, architecture-aware starting point for building it correctly.

---

## What VeriLoop Coder Is Intended To Become

VeriLoop Coder is the future coding product line within the VeriLoop family.

Its long-term goal is **not** to become a superficial clone of Claude Code, a prompt wrapper around a base model, or a generic automation shell. Its goal is to become a **VeriLoop-native coding system** in which a strong code-capable model operates inside a disciplined harness and is governed by an evidence-aware runtime.

In practical terms, the future VeriLoop Coder is intended to support:

- code generation under explicit control boundaries,
- repository understanding across large codebases,
- testing and CI-driven repair loops,
- tool-mediated coding work rather than prompt-only behavior,
- auditable execution traces,
- rollback-aware correction,
- evidence-linked coding conclusions,
- integration with a broader VeriLoop control architecture.

That future system is still being built. This branch is part of the preparation layer.

---

## Why Claude Code Matters Here

Claude Code matters to this repository because it provides one of the clearest publicly visible examples of a **terminal-native coding harness** for a strong model. It demonstrates a practical coding-agent runtime with repository awareness, file editing, command execution, context handling, and developer-facing workflow design.

For VeriLoop, the importance of Claude Code is not ideological imitation. It is architectural reference.

The most valuable lesson is that a serious coding system is not just “a powerful model with a prompt.” It is a runtime environment in which the model can inspect code, act through tools, coordinate state, and remain productive over multi-step engineering work.

That lesson aligns directly with VeriLoop's own view that:

- the **model is the agent**,
- the **harness is the runtime world**,
- and the **control layer must regulate how that world is used**.

---

## Why the Source-Map-Derived Materials Matter Here

A major reason this branch exists in its current form is that a large amount of Claude Code implementation detail became publicly visible through the 2026 source-map exposure and the subsequent circulation of mirror or derivative repositories on the public internet.

That event matters to VeriLoop Coder for one reason: it exposed a rare, large-scale view of how a production-grade coding harness is actually assembled in practice.

For future VeriLoop Coder work, these materials are being treated as:

- **publicly visible reference inputs**,
- **comparative engineering study material**,
- **harness design evidence**,
- **not** as the final identity of VeriLoop Coder.

This distinction is critical.

The future VeriLoop Coder should take inspiration where useful, discard what is not aligned, and recompose the runtime in VeriLoop's own design language. The goal is **learning and synthesis**, not brand substitution or source appropriation masquerading as originality.

---

## Why shareAI-lab / learn-claude-code Matters Here

This repository also draws directly from **shareAI-lab / learn-claude-code**, which is especially valuable because it explains harness engineering as a first-class discipline rather than treating it as scattered implementation trivia.

That project is important for VeriLoop Coder because it makes explicit several truths that also matter to VeriLoop:

- the model is the real agent;
- the harness is what gives the model a usable environment;
- planning, tools, memory shaping, task systems, delegation, and isolation are runtime mechanisms, not “the intelligence itself”;
- a good harness can make a strong model far more useful without pretending that the harness itself is the intelligence.

This is one of the strongest conceptual overlaps between the repository's upstream references and VeriLoop's own long-term direction.

---

## VeriLoop's Distinct Position

Even though this branch currently contains third-party reference material, the architectural stance of VeriLoop remains distinct.

VeriLoop does **not** define a coding agent as “just a workflow graph,” “just a bigger context window,” or “just a tool loop.” It aims to add a stricter governing layer around coding behavior.

The future VeriLoop Coder is intended to stand on three layers:

1. **Backbone model capability**  
   A strong code-capable base model or family of models.

2. **Harness runtime capability**  
   Files, shell, repository inspection, context shaping, tasks, subagents, teams, worktree isolation, browser/tool surfaces, and execution interfaces.

3. **VeriLoop control discipline**  
   A governing runtime that constrains execution through explicit state, uncertainty handling, evidence discipline, budget-aware routing, rollback logic, and auditable traces.

That third layer is what keeps VeriLoop Coder from collapsing into “just another coding wrapper.”

---

## Harness Engineering Is the Mainline, Not Context Inflation

This repository should be read through the lens of **Harness Engineering**, not old-style context inflation.

VeriLoop has already moved away from treating long prompt assembly as the primary way to create agent behavior. In this project, the direction is instead:

- stateful runtime objects,
- explicit tool surfaces,
- session-state discipline,
- controlled memory packets,
- structured receipts and artifacts,
- validator-aware execution,
- bounded rollback and revision.

This matters for two reasons.

First, it means the future VeriLoop Coder is being designed as a **runtime system**, not a long-prompt trick.

Second, it explains why the Claude Code source-map-derived materials and the shareAI harness pedagogy are relevant at all: both expose practical evidence that modern code agents succeed when the harness is well-designed.

---

## What This Branch Contains

At the current stage, this branch should be understood as containing several content categories:

### 1. Third-party reference lines

These include:

- public Claude Code materials,
- publicly circulated source-map-derived code resources,
- shareAI-lab harness-engineering teaching materials.

### 2. Repository-local organization layers

These include:

- repository-local documentation,
- tests,
- skills,
- web assets,
- branch-level structure for future VeriLoop Coder work.

### 3. Future-facing VeriLoop scaffolding

These are the parts that begin to connect external reference materials with VeriLoop's own direction, while the full product is still under construction.

---

## How To Interpret This Repository Correctly

Please interpret this repository according to the following rules.

### Rule 1 — Do not confuse this branch with the finished product

The current repository is a **build-stage research branch**, not the final release of VeriLoop Coder.

### Rule 2 — Do not confuse reference code with VeriLoop-original architecture

Third-party public materials may be preserved here for study, comparison, or contribution purposes. They are not, by themselves, the full substance of VeriLoop Coder.

### Rule 3 — Do not confuse public source-map exposure with official open-sourcing

Some materials in the public ecosystem arose because implementation details became publicly exposed and were then mirrored or reconstructed by third parties. That is different from an official project deciding to open-source a codebase under a formal release process.

### Rule 4 — Do not assume that the future VeriLoop Coder will simply replicate upstream code

The long-term intent is to build an original VeriLoop coding runtime that is informed by public evidence, not defined by it.

---

## Third-Party Provenance Notice

This repository may include, reference, discuss, or structurally learn from third-party materials drawn from multiple origins. These may include:

- official public Claude Code documentation or repositories published by Anthropic;
- publicly circulated source-map-derived Claude Code materials made accessible by third parties after a public exposure event;
- educational harness-engineering materials from **shareAI-lab / learn-claude-code**;
- repository-local notes and adaptations created by the maintainer of this repository.

Where possible, provenance should be made explicit at the file level or directory level.

### Provenance categories

#### A. Official public materials
These are materials that were intentionally published on official public channels by their original publisher.

#### B. Publicly circulated source-map-derived materials
These are materials that became visible through public exposure and were then mirrored, reconstructed, or circulated by third parties in public repositories or archives.

#### C. MIT-licensed educational harness materials
These are materials drawn from repositories such as **shareAI-lab / learn-claude-code**, where the upstream license permits reuse subject to retention of notices and license terms.

#### D. VeriLoop-original materials
These are repository-local documents, architectural notes, tests, and future implementation layers authored specifically for VeriLoop.

This repository should therefore be read as a **mixed-provenance research and build branch**, not as a homogeneous codebase of single-origin materials.

---

## Third-Party Disclaimer

### No affiliation or endorsement
This repository is **not affiliated with, endorsed by, sponsored by, or officially connected to Anthropic, Claude Code, or shareAI-lab**, unless explicitly stated otherwise in a specific file with supporting evidence.

### No claim of official source status
Where third-party materials are preserved or referenced, this repository does **not** claim that those materials constitute an official release by the original vendor unless that status can be independently verified from the original publisher.

### Respect for upstream rights
All third-party names, product names, trademarks, and original copyrights remain the property of their respective owners. Users of this repository are responsible for reviewing the terms attached to any upstream material they choose to use.

### Research and interoperability purpose
At this stage, this branch is being shared primarily for **research, learning, engineering comparison, interoperability study, and future architecture preparation**, while VeriLoop Coder itself is still under active construction.

### No transfer of third-party ownership
The presence of third-party-derived or third-party-referenced materials in this repository does not convert them into VeriLoop-owned original work.

### Correction / removal policy
If a file is incorrectly attributed, should carry additional upstream notice, or should be removed for documented rights reasons, the maintainer should be notified with specific evidence so the repository can be corrected in a timely and responsible manner.

### Informational notice only
This README is an engineering and provenance notice. It is **not legal advice**.

---

## License Handling Expectations

This repository should preserve and respect upstream license obligations wherever applicable.

- If material comes from an MIT-licensed source such as **shareAI-lab / learn-claude-code**, the relevant copyright and license notice should be preserved.
- If material comes from official public Claude Code repositories or documentation, it should be handled according to the terms attached to those sources.
- If material comes from third-party source-map-derived public mirrors, users should independently review provenance, rights status, and downstream use implications before reuse.
- VeriLoop-original material should be clearly distinguishable from third-party material wherever possible.

A clean future state for this repository is one in which provenance is not vague, but explicit.

---

## The Long-Term VeriLoop Coder Vision

The long-term goal is to develop a coding system that is:

- **model-centric**, not workflow-centric;
- **harness-first**, not prompt-fragile;
- **evidence-governed**, not assertion-driven;
- **rollback-capable**, not one-pass brittle;
- **stateful and auditable**, not opaque;
- **productizable**, not merely demonstrative.

This means the future VeriLoop Coder is expected to move beyond simply preserving public materials. It should become a full runtime for code-capable foundation models that can support repository understanding, tool use, execution, correction, evaluation, and governance under a VeriLoop-native architecture.

That future system is still being built.

This branch is part of the groundwork.

---

## Repository Reading Guide

If you are new here, the correct reading order is:

1. understand that **VeriLoop Coder is under construction**;
2. understand that the current branch is primarily a **public-reference and research branch**;
3. understand that **Harness Engineering is the mainline methodology**;
4. understand that future VeriLoop-original implementation is intended to sit **on top of**, and **beyond**, the current mixed-provenance materials;
5. understand that the purpose of this branch is to contribute useful public reference structure now while the larger VeriLoop Coder system is still being built.

---

## Final Statement

VeriLoop Coder is not presenting itself as a finished coding product today.

It is presenting itself honestly as:

- a serious coding-harness research branch,
- a contribution branch for publicly obtainable reference materials,
- a legally cautious, provenance-aware repository,
- and a forward-looking foundation for the future VeriLoop Coder system.

The immediate branch value is **shared public code and harness study**.  
The long-term value is **a distinct VeriLoop-native coding runtime still under construction**.

That distinction is intentional, and this README exists to make it explicit.
