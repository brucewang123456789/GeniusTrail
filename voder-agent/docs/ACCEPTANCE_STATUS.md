# Evidence-based development handoff

**Snapshot:** 2026-10-12; **status:** DEVELOPMENT / NOT PRODUCTION. This handoff makes the files available for continued cloud integration; it is not a success claim.

## Present and executable in this package

- Original user-provided VGR source baseline preserved unchanged under `source/crates/` and the existing `source/packages/`.
- New `source/packages/voder-agent-core`: bounded preflight checks; stable action schema; fail-closed synthetic native transport **interface**; idempotency checks in one process; always-on English golden-rule prompt; Pi 1.1.0-targeted host wrapper.
- New `source/packages/voder-sdk`: basic programmatic session ID, task submission, event query, cancel, dispose.
- Two full bilingual blueprint Markdown files.
- Offline Node test suite (10 assertions/tests at the snapshot).
- Non-overwriting installation script and read-only upstream SDK-export probe.

## Critical incomplete boundaries

1. `native.executeAuthorized(...)` is a **required interface stub**. It is not wired to the uploaded Rust `voderd` authenticated native VGR authorization/commit API. A handcrafted object with `authority: VODER_NATIVE_VGR` is **not cryptographic authentication**. Never treat mocked receipts as genuine native VGR proof.
2. Official package `@earendil-works/pi-coding-agent@1.1.0` is not bundled and was not executed against an actual model API. Export checks and developer fixtures are insufficient evidence of compatibility.
3. Required-file lists are currently supplied by the caller, not autonomously discovered by an LLM. Some stage-specific prompts, context compaction, Skills, memory, search, deep root-cause repair and full task planner have not been implemented.
4. Model host's high-level action/tool parameter typing and current upstream behavior require real TypeScript compilation + actual SDK execution. Time/token limits, approval escalation, native commit binding to an entire task and persistent restart safety are not product-qualified.
5. The SDK has queryable in-memory events, not yet a full persistent streaming/replay SDK or CLI integration. This is NOT a finished production agent.
6. Real external model call, Modal sandbox write/rollback/commit, Rust cargo regressions, fault injection, and real coding benchmark tasks have not been run in this sandbox for this new integration.

## Handoff and acceptance rules

- Run `scripts/test_offline.sh` on extracted archive first. Passing this alone proves **only** local, isolated tests.
- Run `scripts/install_to_autodl.sh /root/autodl-tmp` to add exactly two new packages, refusing to overwrite existing packages. The provided VGR baseline is for reference and will NOT be copied over the target's VGR core.
- Test upstream API separately. Do NOT attach real write/exec privileges to model-selected tools until a trusted native VGR bridge and real failure/recovery acceptance exist.
- Product gate remains **FAIL/UNVERIFIED** until native Rust authority integration and real end-to-end tests succeed.
