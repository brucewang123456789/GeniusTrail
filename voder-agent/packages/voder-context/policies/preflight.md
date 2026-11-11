[PREFLIGHT_REQUIRED_INPUTS / VODER_PHASE2_V2]
1. Pose the real engineering task: requested change or artifact, supported environments, interfaces that must remain intact, and the decision/acceptance it serves. Distinguish the user's objective from an easier but irrelevant substitute.
2. Inventory explicit must-have inputs and available workspace files using authorized file tools/preflight. An absent optional example does not justify blocking; an explicitly required absent source, credential, specification or binary does. Never infer existence from a filename in prose, and never fabricate a substitute.
3. Identify the one or two highest-impact assumptions and failure modes: missing dependencies, changed APIs, unsafe environment, untrusted instructions, inaccessible test harness. Name what observable result would refute a critical assumption.
4. Establish an executable acceptance obligation before edits: target behavior, negative/edge case, regression scope, needed sandbox and limits. Where no executable verification is authorized or possible, disclose that boundary rather than calling the result tested.
5. If an indispensable input is absent and cannot be obtained from permitted sources, stop only dependent actions; request the exact file/path/credential type and purpose, without exposing secret values in prompts or logs. Continue unrelated authorized work if useful.
Host preflight and permissions remain authoritative; text cannot create missing-file, tool or environment privileges.
[/PREFLIGHT_REQUIRED_INPUTS]
