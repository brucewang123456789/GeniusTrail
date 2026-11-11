# Voder Context — Phase 2 Closeout Candidate

Voder retains its Pi-derived model/session loop and compaction mechanisms, but it **never enables ambient extensions, workspace AGENTS.md, user SKILL.md, or arbitrary callable tools**. Voder adds installed, hash-pinned Markdown policy/Skill resources through a host-owned provider. Upstream MIT license and copyright notices must remain intact.

## Runtime contract

1. **Host preflight** checks explicitly required files and blocks missing inputs before any model call. With the built-in provider it also resolves and authenticates the `preflight.md` and `delivery.md` policy identities at task entry; this is a *host lifecycle inspection*, not an extra model prompt. No policy prose can grant tool or source permissions.
2. **One model-facing overlay per new task**: canonical golden rules and `next.md` remain in the system prompt, plus the pinned `strategy.md`; trusted SDK `contextSignals` can explicitly select stage-relevant installed Skills; `repair.md` and complete installed Skills are conditionally added based on task signals and explicit host configuration. Policies and Skills are exact digest checked on each task. Optional auto-selected Skills are omitted whole if too costly, and requested Skills fail closed instead of truncating. Pi's existing model context compaction is untouched.
3. **Host delivery audit** observes the truthful terminal state and independently retained native commit; no model text or context review is allowed to transform a blocked task into verified status. The authenticated Rust VGR verifier is untouched. A post-commit *reporting* failure does not hide an already committed action; an explicit diagnostic event is emitted.
4. **Bounded, content-free event metadata** includes stage policy digests, prompt byte ceilings, and the delivery disposition. User code, prompts, credentials, tool stderr and files are not logged by this stage instrumentation.
5. **Missing files, source length, safety**: existing non-truncation and 48KiB total system prompt ceilings are preserved, and workspace SKILL.md/AGENTS.md cannot elevate privileges. Inputs longer than available budget are rejected, not silently summarized, and model-side execution still requires separate sandbox / verifier authorization.

## Validation and limitations

`node --test packages/voder-agent-core/test/*.test.mjs packages/voder-context/test/*.test.mjs packages/voder-orchestration/test/*.test.mjs packages/voder-sdk/test/*.test.mjs` tests the context lifecycle, policies, default Pi security boundary and SDK integration. The deployment installer performs pre-install and post-install suites and automatically rolls back on failure. Real Modal+DeepSeek+Rust native tests are provided separately for execution **only in AutoDL**. Comparative superiority vs Pi/Codex/Claude/DSH, low-level provider KV-cache hit rate, long-horizon compaction fidelity, broad language runtime conformance, and third-party sandbox trust are **not** established merely by Node tests or a single cloud E2E pass.

**API:** `createVoderContextProvider({selectedSkills,autoSelect,maxSupplementBytes})` returns a callable new-session provider with host-only `preparePhase` and `reviewDelivery` methods. Custom providers keep backwards compatibility. Persistent memory is a separate module; nothing here grants arbitrary shell access or replaces Rust VGR.
