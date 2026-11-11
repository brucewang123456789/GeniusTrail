---
name: sandbox-verification
description: Design reproducible build-run-test checks under an authorized sandbox boundary.
triggers: sandbox,test,pytest,compile,build,code,script,沙盒,测试,编译,执行,运行,代码
---
# Sandboxed execution and verification
1. Inspect repository manifests, lockfiles and toolchain constraints to choose real commands for the language at hand (e.g. Python, JS/TS, Rust, Go, Java, C/C++, shell), not invented generic commands.
2. Before execution, bind: inputs and environment, dependency/setup command if necessary, expected behavior, timeout/tool budget, sandbox authority, failure signals and rollback condition. Treat untrusted code as untrusted.
3. Execute only through an authorized sandbox tool; capture exact exit status, stderr, failing tests, produced files and environment fingerprint. A proposed command is not an executed check.
4. On failure, return the real concise diagnostic to the model; reopen the failure hypothesis, make an evidence-supported minimal change, then re-execute targeted tests followed by relevant regressions. Never imply that a prompt alone implements the repair loop.
5. Distinguish compile/syntax/smoke from behavioral, integration and user acceptance. For nondeterministic behavior, use seeds and bounded repeatability checks when meaningful.
6. Stop on verified obligations or explicit budget/permission/time boundaries; do not claim infinite retries or full correctness. Trusted VGR commitment remains a separate host gate.
