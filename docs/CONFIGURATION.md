# Voder user configuration (five categories)

1. **Primary model:** choose a provider, model ID, endpoint (when needed), and a private API key using `Voder: Configure Models, Modal and Search`.
2. **Secondary/fallback model:** configure an optional model alias and provider in the same wizard; dotted aliases such as `glm-5.3-flash` are supported by the current catalog.
3. **Modal sandbox:** provide your own Modal account/token through the wizard or user-private credential store, never through checked-in source.
4. **Web search:** select a provider and enter the API key in the user-private search configuration (`~/.config/voder/search-config.json`), not `voder-agent/packages/voder-search/search-config.json`.
5. **Native execution / workspace:** open a trusted local project, choose the native VGR gateway configuration or an explicitly unverified draft workflow, and confirm the workspace path. The gateway and Rust build are local prerequisites for verified delivery.

Extension releases may expose the above as separate screens. The names in this guide are conceptual categories, not guarantees of exactly five UI fields. Never publish user-specific configuration or keys.
