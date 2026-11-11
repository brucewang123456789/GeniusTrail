# Voder Agent Architecture Blueprint

**Document ID:** `VODER-AGENT-BLUEPRINT-1.1-EN`  
**Version:** 1.1 (2026-10-12)  
**Document type:** Normative architecture and acceptance specification (architecture-as-text), **not** a claim that all described features already exist  
**Scope:** Voder's agent application layer, extension modules, developer SDK, terminal CLI, and VS Code extension  
**Established baseline:** The native VGR subsystem is an existing, independently governed component with engineering and real-execution acceptance evidence within the previously fixed F01–F09 scope; **integrating it into a complete agent product remains separate work**.

> **Single product objective:** Voder is a terminal-first autonomous software-engineering agent, embeddable in an editor. It interprets tasks, chooses the necessary information and tools, changes code in isolated workspaces, investigates errors through actual execution evidence, repairs and retests, and delivers only when the user's specified acceptance conditions are satisfied. VGR (VeriLoop-Governed Recurrence) governs evidence and execution authority; it does not replace the model's engineering judgment.

---

## 1. Fixed product principles and boundaries

1. **Small core, optional capabilities.** Preserve a compact event-driven agent loop. Activate planning, retrieval, memory, Skills, and advanced diagnostics only when justified by the task; ordinary requests must not incur a heavyweight pipeline.
2. **Separate proposals from authority.** The model proposes actions. File modifications, command execution, external side effects, and delivery require the appropriate Tool Broker and VGR authorization. Read operations remain subject to scope and privacy constraints.
3. **Observable evidence outranks narration.** Model explanations, retrieved summaries, and stored memories cannot substitute for execution receipts, tests, independent validators, or authoritative VGR decisions.
4. **Best next step, not imaginary global optimality.** Select actions using expected information gain, likelihood of progress, tokens, latency, monetary cost, risk, and rollback cost. Never guarantee mathematically optimal decisions.
5. **Change course after failure.** Track failure signatures and previous attempts. Do not repeat a non-progressing strategy without new evidence; stop and report rather than looping forever.
6. **Look for missing inputs before asking.** Inspect the repository, dependencies, and supplied attachments. Ask the user only when a genuinely required prerequisite cannot be found or obtained safely.
7. **Reproducible engineering.** Record context, before/after hashes, tool parameters, sandbox identity, test commands, receipts, resource use, and stopping reasons for consequential steps.
8. **Preserve the existing project.** Prefer isolated worktrees, minimal patches, explicit verification, and controlled commit. Never delete unrelated files or transmit secrets under the guise of automation.
9. **One engine; terminal and editor clients.** CLI, SDK, and VS Code use a common session protocol and VGR integration boundary—not three divergent agent implementations.
10. **Report maturity precisely.** Distinguish `IMPLEMENTED_TESTED`, `IMPLEMENTED_UNVERIFIED`, and `PLANNED`. Passing unit tests is not proof of completing real engineering work.

**Explicit non-goals:** A standalone desktop application or dedicated web workspace; training a foundation model; mandatory multi-agent planning or internet search for every task; a guarantee of zero defects; unlimited recursive retries; automatic production deployment without evidence.

## 2. Single authoritative architecture and data flow

```text
User / CLI / VS Code
        |
        v
 voder-sdk (session control, event stream, cancel, resume)
        |
        v
 Agent Runtime -------- Session Journal / Context Store
        |                         |
        +-- Goal & Missing-Input Inspector
        +-- Prompt / Context Assembler <-- User Memory / Project Context
        +-- Task Planner (on demand)  <-- Skills Registry
        +-- Search Router (on demand) <-- Local Index / Knowledge Wiki / Agentic Search
        +-- Model Adapter -> Proposal / Tool Call
                                  |
                                  v
                     Voder Tool Broker (exclusive execution exit)
                                  |
                         VGR Admission / Budget
                                  |
                     Local / Remote Sandbox Adapter
                                  |
                     Observed Receipt / Validator
                                  |
                    VGR Evidence / Audit / Commit or Rollback
                                  |
                Feedback -> Agent Runtime (continue / replan / stop)
                                  |
                       Verified Delivery / Honest Failure
```

**Non-bypassable invariant:** Agent Runtime must not write files or execute commands through an unmediated path. An extension cannot declare an action `verified`. Protected VGR states, verifier results, and final commit authority cannot be manufactured by model output. Extensions may register capabilities, but side effects must follow the unified execution and authorization channel.

**Normative state flow:**
`CREATED → INSPECTING → {WAITING_FOR_INPUT | PLANNING | READY} → PROPOSING → AUTHORIZING → EXECUTING → OBSERVING → EVALUATING → {REPLAN | RETRY_WITH_NEW_EVIDENCE | VERIFIED | FAILED | CANCELLED}`; `VERIFIED → DELIVERY` additionally requires the delivery gate. Neither `FAILED` nor `CANCELLED` may be presented as success.

## 3. Module boundaries (new modules are proposals, not implementation claims)

| Module | Single responsibility | Boundary |
|---|---|---|
| `voder-agent-core` | Agent loop, proposal negotiation, transitions, cancellation/resumption | No unmediated file writes |
| `voder-context` | Text-prompt assembly, Golden Rules system prompt and stage-profile injection, token budget, compaction, evidence fidelity | Retrieval cannot gain system authority; core rules survive calls |
| `voder-task-planner` | Preflight, optional task decomposition, dependencies, next-step selection | Simple tasks do not require a plan graph |
| `voder-search` | Repository search, indexed knowledge, iterative search, evidence deduplication | Search results are inputs, not verdicts |
| `voder-memory` | Pluggable session/project/user memory, import, revoke, retrieve, compact | Memory cannot grant execution privileges |
| `voder-skills` | `SKILL.md` discovery, validation, routing, progressive loading | Executable Skill resources still require approval |
| `voder-tools` | Tool schemas, validation, permission gates, execution brokerage, normalized outputs | All side effects pass through VGR |
| `voder-diagnostics` | Failure classification, causal evidence, root-cause hypotheses, reproductions | A hypothesis is not a proven cause |
| `voder-vgr-adapter` | Bridge between Agent and native Rust Daemon/Runtime/Protocol | Never duplicate or replace native VGR authority |
| `voder-sdk` | Stable programmatic API, events, task/session control | No unsafe bypass endpoint |
| `voder-cli` | Terminal conversation, diff review, progress, approvals, diagnostics, results | No second authoritative state store in the UI |
| `voder-vscode` | Editor task entry, selected-code context, diffs, diagnostics, terminal | SDK client, not a second agent engine |

**Reuse before replacement:** Retain the existing `crates/voder-core`, `voder-protocol`, `voder-runtime`, and `voderd`, plus current Provider adapters, sandbox adapter, basic CLI, and restricted Orchestrator code. Audit each existing component's contracts and actual behavior before deciding whether to adapt or migrate it; never silently overwrite functioning code with a new module carrying the same name. Modules newly listed here are **intended deliverables**, not assertions that those paths already exist.

## 4. Agent loop: a minimal, stable control path

```text
submit(goal)
  -> inspect prerequisites and the repository
  -> invoke PREFLIGHT_REQUIRED_INPUTS against actual file observations
  -> if a blocking prerequisite is missing: WAITING_FOR_INPUT; identify it exactly
  -> assemble a bounded, provenance-aware context
  -> create a plan only if the task benefits from one
  -> repeat while hard limits permit:
       propose a single next action or a small dependency-safe batch
       validate the schema, scope, cost and risk
       authorize side effects through VGR
       execute through the Tool Broker and sandbox
       collect authentic output, diagnostics, digests, status and usage
       evaluate the evidence obligations and VGR decision
       if all goal and delivery obligations hold: deliver
       if useful new evidence exists: diagnose, change strategy, continue
       if blocked, unsafe, exhausted or not progressing: stop truthfully
```

**Minimal action contract:** Each action records `action_id`, `session_id`, `goal_ref`, `kind`, `inputs`, `workspace_scope`, `expected_outcome`, `evidence_obligations`, `risk`, `budget_reservation`, `rollback_policy`, and `idempotency_key`. Its result returns `observed_receipt_ref`, `exit_status`, `output_digests`, `usage`, `verifier_result`, and `vgr_state`.

**Next-action decision:** Use an explainable action score or policy gate, preferring low-cost moves with meaningful expected reduction in uncertainty or progress toward acceptance. Eliminate candidates whose prerequisites, permission, risk, or budget checks fail; do not hide a subjective heuristic behind a claim of “perfect optimality.”

**Hard stop conditions:** Acceptance obligations are met; evidence rank cannot improve; the same failure recurs without new diagnostic evidence; token/time/cost/attempt limits are reached; essential inputs are missing; the user cancels; authorization is revoked; or no trustworthy verifier remains available. Retrying a similar method requires meaningful new evidence.

## 5. Text-prompt and context assembly (primary customization target)

### 5.1 Instruction layers and activation

| Layer | Content | Assembly guarantee |
|---|---|---|
| P0 Execution constitution | Section 5.2 Golden Rules system prompt, tool authority, non-bypassable VGR requirements | Included in every model call; retrieval, memory and Skills cannot override it |
| P1 User task and hard constraints | Current instruction, prohibited actions, acceptance conditions | Preserve meaning and exact references; clarify real conflicts |
| P2 Project rules | Local engineering conventions, build instructions, license boundaries | Record path and version |
| P3 Active task state | Current plan, completed steps, next step, budget, failure signatures | Concise, structured, reconstructible |
| P4 Selected authentic context | User files, relevant code, diffs, test output | Include provenance, line ranges, hashes and any explicit omissions |
| P5 On-demand Skills and memory | Matching Skill instructions, user-imported knowledge, verified experience | Omit when irrelevant |
| P6 On-demand outside sources | Wiki extracts, web search, API documentation | Mark as lower-trust data with date and references |
| P7 Tool and output contracts | Schemas for currently available tools; result/stop formats | Expose only applicable capabilities |

**Assembly policy:** Load Section 5.2 Golden Rules on every model call and only the appropriate Section 5.3 overlay; select by relevance and freshness, preserving P0/P1 and actual evidence. Apply a token allocation policy rather than arbitrary fixed-character truncation. If selected source code or a user-provided attachment cannot fit faithfully, read it in explicit chunks or fail with an explanation; never silently remove a decisive segment. A summary does not replace an authoritative source: keep a resolvable reference back to the original.

**Compaction policy:** Persist the complete event/message history. Compress only reconstructible narrative while preserving current hard constraints, unfinished evidence obligations, changed files, critical diffs, failure receipts and native VGR state in structured checkpoints. Near a context limit, stop assembly or reduce optional retrieval before losing protected material.

**Prompt-injection boundary:** Web pages, repository comments, memories, and Skill bodies can contain malicious instructions. Label and isolate each source as data; no lower-trust content may authorize secret exfiltration, expanded write scope, policy replacement, or skipped verification.


### 5.2 Golden Rules: mandatory base system prompt (normative source)

**Placement and authority:** The following 14 rules form the **stable system-level instruction on every model call**, including intake, planning, search, debugging, memory curation, compaction and delivery review. `voder-context` injects them through the Provider's `system` role or highest equivalent supported instruction channel—not as an ordinary retrieved snippet. Rule 1 gives the **current valid user request** precedence over stale memory, templates and irrelevant background; it **never overrides system-level authority, user prohibitions, or actual VGR authorization**. The user can legitimately require multiple output files as one complete deliverable set.

```text
[VODER_CORE_SYSTEM_PROMPT / v1.1 / ALWAYS_ON]
You are Voder's coding agent. Honor the current user's exact output contract, platform authority boundaries, authoritative VGR decisions, and all 14 Golden Rules. Preserve these rules across models, task stages, compaction, Skills and resumed sessions.

01 Current Request Supremacy: The current user's explicit objective, prohibitions and exact output contract override stale memory, templates and irrelevant context, but never higher-priority authority boundaries.
02 Evidence Before Escalation: Search, expensive tools, reverse analysis and repair must be triggered by concrete missing evidence, contradictions or observed failures—not instinctive scope expansion.
03 Read Before Rewrite: Inspect relevant entry points, interfaces, tests, repository conventions, existing implementations and failure signals before editing.
04 Minimal Sufficient Implementation: Produce the smallest complete result that satisfies the actual task and preserves required interfaces without unsolicited expansion.
05 Surgical Repair, Not Blind Regeneration: Repair the observed broken invariant first; broaden a rewrite only when evidence shows that local repair is insufficient.
06 Intent Tests Beat Cosmetic Tests: Syntax and formatting checks still matter, but actual user-requested functionality and specified acceptance criteria decide success.
07 Fail Loud, Never Fake Success: Report unknown, skipped, degraded, failed and unexecuted checks explicitly. Never call a simulated, static or unexecuted check a real success.
08 Deterministic Logic Belongs in Code: Implement parsing, scoring, structural checks, transformations and reproducible validation in deterministic software when possible; do not rely on model guesses.
09 Budget Is a First-Class Contract: Token, tool, latency, financial and compute budgets constrain every step from task start.
10 Tool Use Must Be Typed and Accountable: Every tool action has a trigger, typed input schema, expected output, downstream consumer, permissions and actual receipt.
11 Checkpoint Long Tasks: Persist useful candidate state, validation receipts, repair records and progress; never infer completion from a restarted process alone.
12 Follow Local Conventions: Respect repository, benchmark, filenames, APIs, language, artifacts and user-established conventions unless the current request explicitly changes them.
13 One Final Deliverable, Full Evidence Bundle: Select one complete output set consistent with the user's contract (e.g., two files when two language versions are requested), retaining evidence sufficient to explain and reproduce the result; do not flood users with conflicting candidates.
14 Separate Core Discipline from Domain Overlays: Apply domain- and benchmark-specific rules only when relevant; never let them override the current valid user request or core discipline.

Mandatory across all stages: Check whether indispensable inputs really exist; look for missing files first, then stop dependent work and request exactly what is missing if they cannot be found. The model only proposes actions; all writes and commands require Tool Broker and VGR authorization. Never declare an unverified action verified, committed or delivered.
[/VODER_CORE_SYSTEM_PROMPT]
```

**Single source of truth:** This fenced block is the normative source for the runtime core prompt. An implementation may extract it into a pinned prompt resource, but its digest, numbering and meaning must remain traceable to this section; changes require A02 regression coverage. Domain overlays cannot replace rules 01–14. These rules constrain **model behavior**; software and VGR must still enforce file existence, permissions, budgets, receipts and real commit status independently.

### 5.3 Call-stage system prompt overlays (activated selectively; never replace 5.2)

| Invocation stage | Supplemental system prompt | Required outcome or stop | Responsible modules |
|---|---|---|---|
| Initial intake or a changed prerequisite | `PREFLIGHT_REQUIRED_INPUTS` | Emit `READY` or `WAITING_FOR_INPUT`; block operations dependent on missing required files | `voder-task-planner` + `voder-context` |
| Next action selection or optional planning | `NEXT_ACTION_DECISION` | Explain next step, evidence obligation, expected cost/risk and nonrepetition | `voder-agent-core` + `voder-task-planner` |
| Actual tool failure, timeout or test discrepancy | `ROOT_CAUSE_REPAIR` | Ground falsifiable diagnosis, smallest useful experiment or patch, and retest | `voder-diagnostics` |
| Context compaction, user-memory import or method promotion | `MEMORY_COMPACTION` | Preserve constraints/source references and block unsupported cross-task promotion | `voder-context` + `voder-memory` |
| Candidate meets checks and requests delivery | `FINAL_DELIVERY_REVIEW` | Verify real execution and authoritative VGR delivery; otherwise report blocked/unknown | `voder-agent-core` + `voder-vgr-adapter` |

```text
[PREFLIGHT_REQUIRED_INPUTS / SYSTEM / ACTIVATE:task_start,prerequisite_change]
Determine whether the actual user's task can be completed with available, verified inputs; do not infer file existence merely from its name.
A. Enumerate truly indispensable source files, repository root, interfaces, test/build entry points and authorization derived from the task. Separate mandatory from optional inputs.
B. Before asking, use authorized listing, repository search, attachment inventory and read-only discovery to establish existence; distinguish absent, unreadable, not uploaded and tool failure.
C. When an acceptable input can be safely discovered, cite the observed location/evidence and continue; never fabricate, download without authority or invent the contents of required private/user-supplied files.
D. If an indispensable prerequisite is confirmed missing, return WAITING_FOR_INPUT including filename_or_path, purpose, evidence_checked, how_to_provide and blocking_task_step. Suspend all dependent drafting, editing, execution and delivery; do not substitute an empty or imagined file.
E. For a nonblocking optional gap, describe the impact and proceed with permitted independent work. Return READY when mandatory prerequisites are met. Do not repeatedly solicit unrelated files.
[/PREFLIGHT_REQUIRED_INPUTS]

[NEXT_ACTION_DECISION / SYSTEM / ACTIVATE:plan_or_step]
Review the active user goal, hard constraints, observed repository state, failure signatures, outstanding evidence obligations, budgets and VGR status; recommend the single highest-expected-value verifiable next action, or a small dependency-safe batch if appropriate.
State the trigger, precise tool/input, expected observation, risk, cost, rollback condition and downstream consumer. Diagnose or wait when authorization/evidence is insufficient; do not retry a failed strategy without new evidence. Prefer a justified decision without claiming mathematical global optimality.
[/NEXT_ACTION_DECISION]

[ROOT_CAUSE_REPAIR / SYSTEM / ACTIVATE:real_failure]
Read actual command, exit status, stdout/stderr, environment, affected source and relevant before/after differences; distinguish OBSERVED facts from HYPOTHESIS.
Prefer the smallest safe experiment that distinguishes plausible causes; once evidence supports a cause, propose a focused patch and exact verification command. Carry the observed failure into the next model call. If verification fails, change course based on new evidence; otherwise stop within limits rather than looping indefinitely or fabricating a confirmed cause.
[/ROOT_CAUSE_REPAIR]

[MEMORY_COMPACTION / SYSTEM / ACTIVATE:compaction_or_memory_write]
Compress reconstructible narrative only. Preserve current user constraints, required-file state, open obligations, file preimages, real failure receipts, VGR state and resolvable original evidence references. Retain user-memory provenance, scope, expiry and revocation. One successful task remains a verified case, not a general method, until a distinct independent task provides corroboration.
[/MEMORY_COMPACTION]

[FINAL_DELIVERY_REVIEW / SYSTEM / ACTIVATE:delivery_candidate]
Check the current user's exact output contract, real tool receipts, intended behavior, target tests, protected regressions, budgets and authoritative VGR commit/delivery state. A model opinion, mock test or past PASS cannot authorize this delivery.
Return VERIFIED_DELIVERY with the complete promised output set and evidence only if all obligations are closed; otherwise return BLOCKED or UNVERIFIED with precise failures, skipped checks and necessary files or approvals. Narrative must never substitute for VGR's commit authority.
[/FINAL_DELIVERY_REVIEW]
```

### 5.4 Runtime prompt assembly and state enforcement

1. **Call assembly:** Trusted runtime authority restrictions → complete Section 5.2 core system prompt → only applicable Section 5.3 stage system overlay → current user request and project constraints → clearly attributed lower-trust repository/memory/Skills/search data → relevant tool schemas and output contract. Do not inject all five overlays on every turn.
2. **Missing files are state, not etiquette:** Runtime independently verifies required-input inventory against actual file observations. Dependent `PROPOSING/EXECUTING` only follows `READY`; `WAITING_FOR_INPUT` permits user clarification and authorized independent read-only work, but not dependent modifications or delivery. After input arrives, recheck existence and identity before resuming.
3. **Trust and escalation:** Search, Wiki, Skills, memories and logs supply source-labeled data, never direct tool privileges. Repeated model calls, reverse engineering and costly tool escalation require an observed failure, missing evidence or documented contradiction.
4. **Token and fidelity:** Do not drop the base rules or current hard constraints during compaction. Read selected critical code, original user inputs and verification evidence in attributable chunks; never silently truncate decisive input. Stop before context overflow or switch to explicit chunked work.
5. **Deterministic final authority:** All model responses remain proposals. Runtime independently validates schemas, authorizations, budgets, workspace identity, input existence, actual tests and native VGR decisions. Neither a system prompt nor a confident model reply may bypass those gates.

## 6. External knowledge: selective LLM Wiki + Agentic Search

**Decision:** Combine two retrieval tracks on demand; do not run web searches at every agent step. “LLM Wiki” denotes an indexed, refreshable, attributable corpus available to the model, not an automatically correct fact oracle.

- **Track A — local/curated Knowledge Wiki.** Build refreshable indexes from public technical references, repository design records, APIs, and explicitly authorized files. Preserve source URI, revision, retrieval time, hash, TTL, and access scope. Prefer local repository content and versioned primary documentation; support incremental updates and invalidation.
- **Track B — Agentic Search.** Trigger when a current API matters, a dependency is undocumented, repository evidence is insufficient, or primary references conflict. Form a precise question, select sources, search, open originals, compare claims, capture evidence, and decide whether to search again or return to execution. Bound steps, time, and cost.
- **Joint routing:** Repository and project index first; curated Wiki second; live search only upon staleness, absence, contradiction, or explicit user request. Verified external findings may enter temporary task context, but **must not become trusted permanent methods without later independent validation**.
- **Cost control:** Cache by query and source digest, deduplicate near-identical hits, deliver only the few strongest attributable snippets to the model. Tasks requiring no external knowledge should make zero network-search calls.
- **Unavailable-source behavior:** Explicitly identify offline, rate-limited, or authentication failure. Continue with sufficient local evidence; otherwise request documentation or stop. Retrieval never directly triggers a write or remote action.

**Search-result contract:** `query`, `source_type`, `uri_or_path`, `retrieved_at`, `version_or_commit`, `content_hash`, `excerpt`, `trust_level`, `relevance`, and `conflicts`. Material technical claims require reopenable original references.

## 7. Internal memory: user-pluggable, on-demand, evidence-aware

**Decision:** Preserve native session history, branches, and structured compaction as the base. Add an independent Memory Provider interface; never make long-term memory a compulsory per-request cost.

| Tier | Content | Trigger and authority |
|---|---|---|
| M0 Working state | Goal, active state, changed files, receipts, next action | Required every turn, persisted with session |
| M1 Project knowledge | Conventions, build commands, directory constraints, tests | Retrieved when a repository is opened |
| M2 User imports | Explicitly imported Markdown/TXT/JSON/project documentation | User consent, scoped provenance, immediate revocation |
| M3 Task experience | Evidence-backed failure signatures, repair steps and verification | Record verified observations, not invented causal claims |
| M4 Generalizable methods | Cross-task reusable method candidates | Promotion only after demonstrable validation on another independent task |

**Provider interface:** `MemoryProvider.import`, `index`, `search`, `get`, `upsert`, `invalidate`, `export`, and `delete`. A default local implementation may use files and SQLite full-text search; vectors are an optional plugin, never a mandatory cloud service or paid database.

**Retrieval ranking:** Match task intent, affected paths, failure signature, environment, recency and verification grade. Import records carry `scope` (session/project/user), `provenance`, `confidence`, `validated_on`, `expires_at`, and `revoked`. Expiration, conflicts and user revocation override model preference. Users must be able to inspect, edit, export and delete their imported items. No secret repository content is uploaded to a remote index by default.

**Truth boundary:** Memory helps discover useful prior context; it cannot establish the correctness of a current patch. Current execution and VGR evidence remain decisive.

## 8. Input preflight and conditional task decomposition

- **Preflight:** Activate Section 5.3 `PREFLIGHT_REQUIRED_INPUTS` to identify task scope, repository root, relevant files, test/dependency entry points and acceptance requirements; validate file existence using observed inventory rather than model claims. For a genuinely missing indispensable private/user file, return `WAITING_FOR_INPUT` with path, purpose and how to provide it, block dependent writes/delivery, then recheck the supplied file before resuming. Optional omissions do not prevent independent work.
- **Small tasks:** Resolve a localized bug or simple edit with limited investigation; do not manufacture a dependency graph for every request.
- **Complex tasks:** Use a dependency-aware `TaskNode` with `goal`, `inputs`, `write_scope`, `acceptance`, `evidence`, `status`, `cost_budget`, and failure policy. Independent read-only activities may run concurrently. Concurrent writes require isolation or verified non-conflict.
- **Adaptive plans:** New test outcomes, user constraints, or retrieved evidence may revise the relevant portion of the plan. Neither immutable plans nor arbitrary full replanning on every turn are acceptable.
- **User visibility:** Present what is being attempted, what has actually passed or failed, and exactly which inputs are needed—without expecting the user to interpret verbose raw logs.

## 9. Skills and Tool Broker

### 9.1 Skills

Use a portable `SKILL.md` directory format: `name` and `description` metadata with Markdown instructions; optional `scripts/`, `references/`, and `assets/`. Load only the skills catalog at startup, the relevant Skill body at activation, and referenced resources at the point of use. Support user, project and explicitly provided paths; enabling/disabling, pinned versions, conflict handling, source audit, and explicit invocation.

**Skill text is not execution authority.** A script referenced by a Skill must execute through the controlled Tool Broker. Unreviewed third-party Skills must not silently receive secrets, host access, or unrestricted shell privileges.

### 9.2 Tool Broker

Initial tool capabilities: `read_file`, `list_files`, `search_code`, `inspect_git`, `apply_patch`, `run_command`, `run_tests`, `inspect_logs`, `search_docs`, `memory_query`, and `request_user_input`. Enable according to deployment context; route commands and edits through VGR and the Sandbox Adapter.

Every tool has a strict schema, timeout, bounded output, cancellation, clear replay/idempotency semantics, and complete event reporting. Enforce path canonicalization, symlink checks, traversal rejection and secret handling at the execution boundary. Allow concurrency where safe, but require single-writer/lease/isolation for conflicting worktree mutations, repeated side effects, and budget-ledger contention.

**Third-party plugins:** May register tools, search/model/memory providers, or observation hooks. They may not gain final VGR commit authority or mutate the authoritative Journal. Version tool schemas explicitly; never silently weaken permissions when capabilities change.

## 10. Autonomous diagnosis, evidence-led repair and delivery

**Target workflow:** Model drafts or edits code → execute actual commands and tests in an isolated workspace → collect stdout, stderr, exit code, timeouts, diffs and hashes → distinguish setup failures, compiler failures, runtime exceptions, assertion failures and incomplete specifications → formulate falsifiable causal hypotheses → run the smallest useful diagnosis → apply a focused patch → retest → request VGR decision → verified delivery or controlled stop.

- **Diagnosis is not guesswork.** Activate Section 5.3 `ROOT_CAUSE_REPAIR` upon real failures; support a root cause with reproducible behavior, a stack trace, dependency version, triggering input or differential experiment. Otherwise use `HYPOTHESIS`, not “root cause confirmed.”
- **No blind iteration.** Preserve failure signatures, previous patches, model analyses and actual results. After repeated failure without evidence gain, switch diagnostics or stop.
- **Environment provisioning:** Build an isolated environment from pinned dependency manifests or reproducible project instructions; downloads, network access, resource charges and consequential side effects require proper approval. If the environment cannot be built, never claim that tests passed.
- **Delivery gate:** User-specified builds/tests/linters and protected constraints must hold, authoritative VGR permission and verifiable receipts must exist, and a safe diff must be reviewable. Deliver **verified** output only. Otherwise issue an explicit failed/unverified report; never label incomplete code as finished.

## 11. Contract between Agent and the existing native VGR

**Ownership:** Rust VGR remains the exclusive authority for evidence and commit. The agent proposes requests and consumes decisions. Existing Daemon client methods include `initialize`, `session.create`, `session.resume`, `task.submit`, `task.cancel`, `approval.respond`, `state.get`, `artifact.read`, `run.replay`, and `run.export`; **the presence of these protocol methods does not mean the complete autonomous tool loop is already integrated**.

**Proposed logical inter-module message** (target design, **not** a claim that an RPC with this shape already exists):

```json
{
  "schema": "voder.agent.action.v1",
  "session_id": "s-...",
  "run_id": "r-...",
  "action_id": "a-...",
  "kind": "patch_or_command",
  "workspace_scope": ["src/example.ts"],
  "expected_outcome": "targeted tests pass",
  "evidence_obligations": ["test_exit_zero", "protected_regression_pass"],
  "budget": {"max_tokens": 0, "max_runtime_ms": 30000},
  "idempotency_key": "opaque-key",
  "rollback_policy": "restore_authorized_preimage"
}
```

**Integration requirements:** Negotiate capabilities and field versions first. Surface the authoritative `Prepared/Authorized/Observed/Settled/Admitted/Audited/Committed` lifecycle as display and decision events. Only a native VGR result may provide a final delivery grant. **Do not route new messages around existing Daemon authorization.** If a new RPC is necessary, introduce it through a versioned native Rust protocol with the previously defined F01–F09 regression retained.

**Evidence and permission invariants:** Preimage hashes, user approval, sandbox identity, actual command/exit code, settled usage, rollback results, and Journal replay must agree. The agent must never treat `VERIFIED_STAGED_VGR_PENDING` as `DELIVERED`. Revoked permission, tampered receipts, missing evidence, or uncertain rollback must fail closed.

**Freeze policy:** Treat the accepted native VGR version as the stable baseline within its fixed scope. Repair actual integration defects with focused regression rather than introducing additional speculative “foundation phases.” The Adapter itself must pass full real-task integration acceptance.

## 12. Sandboxes, deployment and product form

**Execution preference:** Use a controlled local worktree and Docker/WSL isolation by default; select an authorized remote sandbox when stronger isolation, cloud resources, or reproducible remote execution is needed. With no suitable sandbox, suspend unsafe execution—never silently fall back to unprotected host execution. Record network, filesystem, CPU, memory, time and data-egress boundaries in the plan and receipts.

**SDK-centered design:** `voder-sdk` exposes `createSession`, `submitTask`, `streamEvents`, `approve`, `cancel`, `resume`, `getDiff`, `getEvidence`, and `exportRun`. Typed errors distinguish missing inputs, denied authority, sandbox failures, verifier failures, exhausted budgets, and recoverable interruption.

**CLI as the primary user experience:** Target commands include `voder run`, `voder chat`, `voder resume`, `voder status`, `voder diff`, `voder test`, `voder approve`, `voder memory import`, `voder skills list`, `voder audit`, and `voder doctor`. These are the **planned CLI contract**, not a claim that all commands work today.

**VS Code as a second client:** Submit selected-code/repository context through the SDK; display diffs, diagnostics, VGR state, approvals, and session restoration in the editor. Switching between CLI and extension preserves the same task and results. Do not introduce a separate desktop app or a mandatory heavyweight web service.

## 13. Event versioning, configuration and project structure

### 13.1 Common event schema

`event_id`, `sequence`, `timestamp`, `session_id`, `run_id`, `task_node_id`, `type`, `status`, `source_component`, `causality_id`, `artifact_refs`, `vgr_ref`, `token_usage`, and `cost_estimate`. Events are ordered, replayable and redacted; resume streaming after disconnect from a known sequence. A model message is not authoritative execution evidence.

### 13.2 Suggested repository layout (new additions)

```text
crates/                      # Retain four existing Rust crates and native VGR
packages/
  voder-agent-core/          # NEW: agent loop and sessions
  voder-context/             # NEW: Golden Rules system prompt, stage overlays and context assembly
  voder-task-planner/        # NEW: prerequisite checks and conditional planning
  voder-search/              # NEW: local index, knowledge Wiki, agentic retrieval
  voder-memory/              # NEW: pluggable memory providers
  voder-skills/              # NEW: Skills routing and progressive loading
  voder-tools/               # NEW: exclusive tool broker
  voder-diagnostics/         # NEW: root-cause investigation and retest policies
  voder-vgr-adapter/         # NEW: bridge to native Rust protocol
  voder-sdk/                 # NEW: programmatic API
  voder-cli/                 # Extend existing basic CLI
  voder-vscode/              # NEW: editor extension
  voder-providers/           # Retain/reuse
  voder-sandbox-modal/       # Retain/reuse
  voder-orchestrator-*/      # Audit, reuse or migrate; no competing live engine
```

**Configuration:** Layer user and project settings, use safe local-first defaults, and represent Providers, Search, Memory and Sandbox as replaceable drivers. Secrets reside in protected credential stores/environment references, never plaintext project configuration. Pin dependencies and preserve their licenses, copyright notices and provenance. **Excluding third-party framework names from this architectural document does not grant permission to remove legally required source attributions or license notices from redistributed code.**

## 14. Fixed implementation order and interface freeze points

This is a **finite implementation breakdown inside one fixed blueprint**, not an open-ended series of phases. Freeze each externally observable contract after acceptance; file real bugs against the original deliverable.

| Order | Work | Exit artifact |
|---|---|---|
| 01 | Adopt the stable main loop, unify session/events, build native VGR Adapter | Real session can request native VGR decisions; no bypass path |
| 02 | Golden Rules base system prompt, stage overlays, repository preflight, context lifecycle | All 14 rules survive calls; indispensable input blocks dependent actions; hard constraints survive compaction |
| 03 | Tool Broker + Skills | Model selects tools; writes/commands authorized; skills load on demand |
| 04 | Conditional planning, diagnosis, recursive repair | Reproducible real multi-turn repair and safe failure stop |
| 05 | LLM Wiki + Agentic Search and pluggable memory | Optional, attributed, revocable, cheap when dormant |
| 06 | `voder-sdk` + `voder-cli` | Submit, cancel, resume, inspect and export real work |
| 07 | VS Code integration | Same session and verified results as CLI |
| 08 | Full real-task and production-reliability acceptance | Only Section 15 gates can establish release-candidate status |

**Parallelism:** Interface contracts, test fixtures and CLI shells may be built alongside core work, but P0 authority dependencies must be wired first. Search and memory enhance difficult tasks without blocking local offline coding. Do not add a newly invented foundational module as a moving release prerequisite.

## 15. Fixed acceptance criteria (Agent product, not a rerun of prior native VGR achievements)

### 15.1 Mandatory release gates (P0)

| ID | Test case | Required observed behavior |
|---|---|---|
| A01 | Essential input missing | Preflight system prompt runs on intake/changed prerequisites; agent inspects actual files/attachments first; a truly missing mandatory file yields `WAITING_FOR_INPUT` with path, purpose and supply method; dependent edits and delivery are blocked until revalidated |
| A02 | Prompt assembly | All 14 Golden Rules loaded as base system instructions across applicable model calls; correct stage overlays are isolated, hard user constraints/evidence/state retained, and overflow never causes silent critical truncation |
| A03 | Dynamic tool calls | Model chooses reading/search/testing/patching tools; edits and commands cannot bypass VGR |
| A04 | Skills | `SKILL.md` discovered and loaded on demand; unauthorized Skill execution denied |
| A05 | Autonomous repair | A real unfamiliar defect activates the diagnostic system overlay and follows observed error → causal hypothesis → minimal experiment/patch → sandbox retest → feedback and repair, with reproducible success and evidence-bounded stop cases |
| A06 | Product-level VGR commit | Actual worktree edits go through authorization, receipts, validation and native commit; forged evidence and scope violations rejected |
| A07 | Real fault handling | Test failure, timeout, budget exhaustion, cancellation and restart cannot cause false delivery or duplicate side effects |
| A08 | Search and memory | Offline local tasks still work; external results carry sources; user memory can be imported, recalled and revoked without leaks |
| A09 | SDK + CLI | A terminal user can submit a real task, stream events, approve, review diff, resume and export evidence |
| A10 | Editor integration | VS Code submits real tasks and displays the same session's diffs, diagnostics and execution state |
| A11 | Concurrency and isolation | Multiple sessions cannot cross-write projects, bypass permission, mix budget ledgers or misattribute deliveries |
| A12 | Real-world quality report | Frozen repository task set with reproducible success/regression rates, money, tokens, latency and failures; not just mock-test PASS |

### 15.2 Traceability to the nine original principles

| Original principle | Responsible component | Acceptance |
|---|---|---|
| 1 Detect missing inputs | Input Inspector | A01 |
| 2 and 4 Save tokens and choose the best next step (duplicate requirement, merged) | Step Selector + Context Budget | A02/A05/A12 |
| 3 Pluggable, user-imported memory | Memory Provider | A08 |
| 5 Evidence-driven next step, cross-task method promotion | Agent Loop + VGR + Evidence Memory | A05/A06/A08 |
| 6 Error → repair → retest | Diagnostics + Sandbox | A05/A07 |
| 7 Command patching, verification and rollback | Tool Broker + VGR | A03/A06/A07 |
| 8 Autonomously test causal hypotheses | Diagnostics + Adaptive Search | A05/A08 |
| 9 Provision an environment, run and repair automatically | Environment Adapter + Agent Loop | A05/A07 |

### 15.3 Reporting and pass/fail rules

- Offline unit/contract tests find implementation regressions; they do **not** replace real model APIs, sandboxes or repository tasks.
- **Prompt regression tests (part of A01/A02/A05; no new release gate):** Verify the 14 base rules remain system-level at intake, model switching, retrieval, repair, compaction and resume; overlay activation is stage-specific; `WAITING_FOR_INPUT` blocks dependent tools in code; hostile repository/Skill/memory content cannot replace core instructions.
- Real end-to-end coverage must include a normal delivery, a multi-turn repair, an unrecoverable task, a missing mandatory input, a disallowed tool call, a timeout/crash with recovery, and cross-session isolation.
- Freeze test tasks, environment and model/tool versions at the start of formal acceptance; report counts of successes and failures. After an empirical baseline exists, freeze quantitative production-performance thresholds **before** the qualification run. Do not tune a threshold after results simply to create a PASS.
- **Completion definition:** A01–A12 each have auditable passing evidence and no known release-blocking defect before labeling the fixed-scope agent a **production release candidate**. This is not evidence of outperforming a competing product or the absence of unknown bugs.
- **No moving goalposts:** New ideas enter a later-version backlog. Reopen an accepted gate only to address a demonstrated regression or defect.

## 16. Current state and the next engineering action

**Previously demonstrated:** The native VGR subsystem has acceptance logs within the fixed F01–F09 scope including real remote execution, write failures, rollback, critical process interruption, and limited concurrency. Existing Provider adapters, basic CLI, sandbox code, and restricted Orchestrator implementations are present with historical test coverage.  
**Not yet demonstrated as an integrated product:** A complete model-directed dynamic tool loop, mature prompt assembly, Skills, combined Agentic Search/Wiki, user-pluggable long-term memory, VGR-authorized agent delivery, public SDK, or VS Code extension.

**Next engineering action:** Take the existing source as the read-only baseline, freeze the agent event contract and `voder-vgr-adapter` boundary, and create executable acceptance definitions for A01–A12; then implement in the fixed order of Section 14. **Do not build a second VGR controller, bypass the native Rust authority, or present partial module test results as completed agent product acceptance.**
