# VeriLoop Coder

**Evidence-governed harness engineering for code-capable foundation models.**

VeriLoop Coder is a coding-model repository built around a simple premise: **the model is the agent, and the code around it is the harness**. This project does not treat intelligence as a prompt graph, a no-code workflow, or a pile of orchestration glue. Instead, it treats the base model as the decision-making core and builds a disciplined runtime around it: tools, context management, permissions, execution interfaces, memory, task persistence, and evidence-governed control.

This repository combines three distinct layers:

1. **Anthropic public Claude Code materials** as a reference for a production-grade terminal-native coding workflow, public command patterns, and developer-facing integration style.
2. **shareAI-lab's harness-engineering pedagogy** as a reference for progressive agent runtime construction, especially the transition from a minimal tool loop to planning, task graphs, subagents, teams, and worktree isolation.
3. **VeriLoop's original control-plane architecture** as the governing layer that adds evidence contracts, uncertainty-aware routing, budgeted action selection, rollback discipline, and auditable execution.

The result is not a clone of Claude Code, not a template pack, and not a prompt wrapper. It is a **VeriLoop-native coding harness** designed to support the development of robust code models, code agents, and coding-oriented foundation-model products.

---

## Design Position

### 1. The model is the agent
A coding agent is fundamentally a trained model operating in an environment. The harness does not create intelligence; it provides the environment in which intelligence can perceive, reason, and act.

### 2. The harness is the runtime world
For coding tasks, the runtime world includes file operations, shell access, repository state, diagnostics, tool invocation, task persistence, worktree isolation, and permission boundaries.

### 3. VeriLoop adds evidence-governed control
Where a conventional coding harness focuses on capability exposure, VeriLoop adds a stricter control layer:

- **state-centered execution** rather than prompt-only drift,
- **claim-and-evidence binding** rather than free-form assertion,
- **budget-aware action routing** rather than uncontrolled tool cascades,
- **rollback and correction** rather than one-pass completion,
- **auditable loop logs** rather than opaque agent traces.

This is the differentiator of VeriLoop Coder: it does not stop at making a model act; it constrains the action process so that coding behavior can become more inspectable, reproducible, and trustworthy.

---

## What This Repository Is For

VeriLoop Coder is intended to serve as:

- a **clean harness foundation** for code-capable models,
- a **runtime control shell** for future VeriLoop coding products,
- a **development and evaluation platform** for tool-using code agents,
- a **bridge** between publicly visible coding-agent practices and VeriLoop's evidence-driven architecture,
- a **training and post-training environment** for capturing high-value coding trajectories.

This repository is therefore aimed at engineers who are not merely prompting a model, but building the runtime conditions under which a model can become a reliable coding agent.

---

## Architectural Thesis

```text
Backbone Model
    ↓
Harness Runtime
    ↓
VeriLoop Control Plane
    ↓
Task / Team / Worktree Execution
    ↓
Evidence, Memory, Rollback, Evaluation
```

### Layer 1 — Backbone Model
The base model may be a proprietary API model or an open-weight backbone. It is the inference core that interprets goals, proposes actions, consumes tool results, and generates responses.

### Layer 2 — Harness Runtime
The harness runtime provides the environment:

- file read / write / edit,
- shell and process execution,
- repository inspection,
- search and retrieval,
- structured task state,
- subagent or teammate delegation,
- worktree or sandbox isolation,
- approval and permission controls.

### Layer 3 — VeriLoop Control Plane
VeriLoop overlays the harness with an evidence-driven loop. In this repository, that means the coding runtime is governed by the following principles:

- **State (`St`)** is explicit rather than implicit.
- **Goals (`Q`)** are maintained as controlled convergence targets.
- **Uncertainty (`U`)** informs whether to answer, retrieve, inspect, or execute.
- **Budget (`B`)** constrains tool use, expansion, and iterative correction.
- **Evidence (`E`)** binds conclusions to inspectable traces.
- **Claims (`C`)** can be decomposed, checked, revised, or rolled back.
- **Logs (`L`)** preserve auditable loop history.

### Layer 4 — Execution Fabric
The execution layer is where practical coding work happens:

- single-turn tool execution,
- planning and task breakdown,
- background jobs,
- teammate coordination,
- asynchronous messaging,
- isolated work directories,
- structured review and verification.

### Layer 5 — Evaluation and Improvement
The repository is also a data-generation and evaluation environment. Each coding trace can become:

- a harness debug record,
- an audit trail,
- an evaluation case,
- a post-training trajectory,
- a failure-analysis sample for future model refinement.

---

## Why Harness Engineering Matters Here

Harness engineering is not secondary infrastructure. In practice, it determines whether a coding model behaves like a fragile demo or a deployable engineering system.

This repository adopts harness engineering as a first-class discipline for five reasons:

1. **Capability exposure must be controlled.** A model with unrestricted tools is not automatically useful.
2. **Context must be organized, not merely accumulated.** Long coding sessions require compaction, decomposition, and selective recall.
3. **Execution must be structured.** Planning, task graphs, and worktree isolation reduce interference and improve completion quality.
4. **Permission boundaries must be explicit.** Safe coding requires controlled action surfaces.
5. **Reasoning must be grounded in evidence.** VeriLoop is designed to prevent unsupported coding assertions from passing unchecked.

For this reason, VeriLoop Coder treats harness engineering not as an accessory to the model, but as the operational substrate that lets the model become a usable coding system.

---

## Upstream Provenance and File Attribution

This repository draws from two upstream reference lines and one original line.

### A. Files derived from Anthropic public Claude Code materials
These are files or patterns originating from Anthropic's public Claude Code repository, public examples, or official public-facing developer materials.

They should be identified as one of the following in file headers, notices, or repository documentation:

- `Source: Anthropic public Claude Code materials`
- `Derived from Anthropic Claude Code public examples/guides`
- `Adapted from Anthropic public Claude Code repository`

Typical content in this category includes:

- public command patterns,
- example integration workflows,
- public GitHub automation examples,
- developer-facing CLI or repository usage conventions,
- public documentation-derived structure or interface guidance.

### B. Files derived from shareAI-lab / learn-claude-code
These are files or patterns adapted from the educational harness-engineering repository `learn-claude-code`.

They should be identified as one of the following in file headers, notices, or repository documentation:

- `Source: shareAI-lab learn-claude-code`
- `Derived from shareAI-lab harness engineering sessions`
- `Adapted from shareAI-lab learn-claude-code examples`

Typical content in this category includes:

- progressive session structure for harness building,
- minimal agent loop demonstrations,
- tool-dispatch pedagogy,
- planning and task-system teaching patterns,
- subagent, team, and worktree educational scaffolds,
- mental-model-first explanatory organization.

### C. Files original to VeriLoop
These are the files that define the actual identity of this repository and should be treated as VeriLoop-native original work.

Typical content in this category includes:

- E³-Loop or VeriLoop control logic,
- evidence-gated routing,
- uncertainty and budget handling,
- rollback and correction mechanisms,
- schema contracts and state representations,
- model adapters and product-specific execution policies,
- evaluation logic aligned with VeriLoop objectives,
- any repository design that reinterprets upstream references under a new control architecture.

### Attribution rule
If a file is mixed rather than purely inherited, it should be marked as:

- `Mixed provenance: Anthropic public materials + shareAI-lab pedagogy + VeriLoop original modifications`

This repository should therefore preserve **clear provenance boundaries** rather than pretending that all imported material is homogeneous.

---

## What Was Taken from Each Upstream Line

### From Anthropic public Claude Code materials
This repository draws inspiration from Anthropic's public Claude Code positioning as a terminal-native coding agent that can understand a codebase, execute routine tasks, and interact through natural-language commands. That public framing establishes the practical target: a coding system that is usable by engineers inside real repositories and workflows.

In this project, Anthropic-derived influence is primarily operational rather than doctrinal:

- terminal-first coding interaction,
- coding-agent workflow orientation,
- repository-aware execution expectations,
- public integration patterns,
- real engineering use-case framing.

### From shareAI-lab's learn-claude-code
This repository draws direct methodological value from the harness-engineering decomposition presented by `learn-claude-code`, especially its explicit separation between the **agent loop** and the **surrounding harness mechanisms**.

The most valuable imported ideas from that line are:

- the progression from minimal loop to full harness,
- the notion that the loop stays stable while tools and runtime mechanisms expand,
- structured pedagogy around planning, skills, tasks, background work, teams, and worktree isolation,
- the treatment of harness engineering as a domain-general discipline rather than a coding-only trick.

### What VeriLoop adds beyond both
VeriLoop adds a layer that neither source provides in the same form:

- evidence binding,
- uncertainty-structured routing,
- budget-sensitive control,
- rollback discipline,
- explicit state contracts,
- audit-oriented logs,
- controllable convergence rather than purely free-running orchestration.

That is the point of this repository: not to duplicate either upstream source, but to synthesize them into a stricter and more research-oriented coding-model harness.

---

## Core Repository Layout

This project follows the canonical VeriLoop build order:

```text
01_foundation/          Base abstractions, shared runtime contracts, minimal loop surfaces
02_schema_contracts/    State, claim, evidence, task, and message schemas
03_backbone_adapters/   Connectors for supported model backbones and providers
04_control_plane/       VeriLoop routing, gating, budget, rollback, termination logic
05_memory_evidence/     Evidence stores, retrieval, memory shaping, trace persistence
06_sandbox_runtime/     Tool execution, shell boundaries, worktree or sandbox controls
07_harness_engineering/ Tool registry, task system, planning, teams, agent workflows
08_peft/                Optional adaptation, trajectory shaping, lightweight specialization
09_inference_serving/   Serving paths, session runtime, APIs, deployment interfaces
10_evals/               Harness evaluation, coding benchmarks, audit tests, failure analysis
11_ops/                 CI/CD, observability, governance, maintenance operations
```

This layout reflects the actual logic of the project:

- first define the foundation,
- then define the contracts,
- then bind the model,
- then govern the model,
- then expose tools and memory,
- then engineer the harness,
- then evaluate and operationalize the result.

---

## Core Harness Mechanisms in VeriLoop Coder

The harness mechanisms in this repository are expected to include, at minimum:

1. **Agent loop** — the minimal inference-and-tool cycle.
2. **Tool registry** — a stable dispatch surface for external actions.
3. **Planning surface** — explicit decomposition before execution.
4. **Task graph** — persistent goals, dependencies, and status transitions.
5. **Knowledge loading** — on-demand injection rather than unbounded prompt inflation.
6. **Context control** — compaction, trimming, or structured carry-forward.
7. **Subagent support** — isolated contexts for bounded subtasks.
8. **Team coordination** — asynchronous multi-agent collaboration.
9. **Execution isolation** — worktree, sandbox, or equivalent boundary control.
10. **Permission governance** — controlled action approval and trust boundaries.
11. **Evidence binding** — code claims tied to traces, outputs, or inspections.
12. **Rollback and correction** — ability to revise, reverse, or re-run under control.

These mechanisms define the repository more than any single model call does.

---

## VeriLoop-Specific Coding Logic

VeriLoop Coder is not just a coding assistant shell. It is a coding runtime shaped by the following logic:

### Evidence-governed coding
Code changes, repository claims, and diagnostic conclusions should not be accepted merely because the model states them confidently. The runtime should favor inspection, verification, and trace-linked justification.

### Budget-sensitive iteration
The model should not expand indefinitely. Planning depth, tool invocation count, inspection breadth, and revision loops should be shaped by budget constraints.

### Structured uncertainty
When confidence is low, the system should not simply continue producing text. It should transition into retrieval, inspection, controlled execution, or user escalation.

### Recovery by design
A robust coding system must support correction, not just generation. Rollback and repair are part of the architecture, not afterthoughts.

### Clean separation of intelligence and mechanism
The model decides. The harness exposes the environment. The control plane regulates the conditions of action. This separation is necessary for both engineering clarity and future model substitution.

---

## Intended Outcomes

VeriLoop Coder is designed to support the development of systems that can:

- read and modify large codebases with stronger control,
- operate across multi-step engineering tasks without uncontrolled drift,
- generate auditable coding traces,
- support model replacement without redesigning the runtime,
- serve as a foundation for post-training and evaluation,
- evolve from a coding harness into a broader evidence-governed agent runtime.

---

## Non-Goals

This repository is **not** intended to be:

- a no-code workflow engine,
- a prompt-library product,
- a blind clone of Claude Code,
- a purely educational toy with no path to productization,
- a monolithic claim that orchestration alone creates agency.

Its purpose is narrower and more serious: to provide a rigorous harness foundation for code-capable models under VeriLoop's control logic.

---

## Build Philosophy

The build philosophy of VeriLoop Coder can be summarized in one sentence:

> **Use public coding-agent harness insights where they are useful, adopt progressive harness engineering where it is structurally sound, and re-govern the whole runtime through VeriLoop's evidence-driven control plane.**

That is why this repository matters.

It takes the practical terminal-native coding horizon represented by Anthropic's public Claude Code materials, combines it with the pedagogically clean harness decomposition demonstrated by shareAI-lab, and then restructures both under a stricter control architecture suitable for VeriLoop's long-term model and product ambitions.

---

## License and Upstream Respect

This repository must preserve all upstream license obligations.

- Files derived from Anthropic public materials remain subject to the terms attached to those materials.
- Files derived from shareAI-lab material remain subject to the MIT terms attached to that repository.
- VeriLoop-original files should carry their own project license and attribution policy.

This repository is therefore a **synthesis project**, not an ownership erasure project.

---

## Status

VeriLoop Coder should be understood as a **harness-first coding-model repository**:

- model-aware,
- tool-capable,
- evidence-governed,
- architecture-driven,
- attribution-conscious,
- built for serious coding-agent engineering rather than superficial automation.

---

## Final Statement

The foundation model is the agent.
The harness is the world in which the agent can work.
VeriLoop is the control discipline that keeps that world auditable, evidence-bound, and strategically convergent.

**VeriLoop Coder exists to turn code-capable models into better-governed coding systems.**
