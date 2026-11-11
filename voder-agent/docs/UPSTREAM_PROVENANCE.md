# Upstream dependency provenance (external; no upstream code redistributed in this ZIP)

- Official GitHub repository (moved from `badlogic/pi-mono`): `https://github.com/earendil-works/pi`.
- Pinned official Git tag: `v1.1.0`; corresponding Git commit: `abe508e1b89912adde45528136c3221eb69acdd7` (checked via GitHub tag ref on 2026-10-12).
- Official package: `@earendil-works/pi-coding-agent@1.1.0` (TypeScript; Node >=22.19.0).
- Official SDK reference: `packages/coding-agent/docs/sdk.md`; example: `packages/coding-agent/examples/sdk/05-tools.ts`.
- Upstream license: MIT, Copyright (c) 2025 Mario Zechner. The repository's LICENSE must remain with any redistributions of copied upstream source.
- This development handoff does **not** vendor the upstream package; it dynamically imports the installed package. No upstream model session was run in the current sandbox.
- The coding-agent main loop is **external** at this point; the Voder wrapper delegates to the upstream in-process session API once it is installed.
- **Do not** label this snapshot as an integrated production agent. The native VGR authenticated Tool Broker connector is still a required implementation/verification step.
