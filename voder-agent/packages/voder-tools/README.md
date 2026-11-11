# Voder Tool Broker — third module

A **host-owned, opt-in** adapter registry for bounded read-only tool capabilities, running alongside the existing VGR-authorized patch and exact candidate verification tools.

- **Core default tools remain unchanged**: `voder_authorized_action`, `voder_read_file`, `voder_list_files`, `voder_search_code`, and, when configured, `voder_verify_candidate`.
- Registration requires an explicit `HostToolRegistry` constructed by the integrating administrator; there is **no ambient Pi extension, MCP, Shell, or workspace tool auto-discovery**.
- Only `voder_ext_*` names with bounded, closed scalar JSON schema, `read/search/verify` capability and host function are accepted. Execution is limited by per-task budgets, output size, cancellation/timeout, duplicate-failure/duplicate-success suppression and a circuit breaker.
- `rankToolCapabilities` produces a transparent predicted-value ranking considering expected evidence, token estimate, latency and risk; **this is a heuristic, not a proof of global optimality**. A host-installed tool is executable JavaScript and is not intrinsically sandboxed by its `read_only` declaration: only administrators should register trusted code or a properly isolated remote adapter. Future tool providers must receive external isolation certification before allowing untrusted plugin binaries.
- Every plugin result is marked `UNTRUSTED_PLUGIN_DATA`; it cannot grant model permission or VGR commit authority. Model decisions and observed tool events remain separate from actual native VGR receipts.
- All modifications to files and the system remain confined to the existing Rust-authorized VGR route; no arbitrary `run_command` or `apply_patch` tool is exposed by this package.

```js
import {HostToolRegistry} from '@voder/tools';
const registry=new HostToolRegistry().register({
  name:'voder_ext_docs', description:'Read public pinned documentation',
  capability:'search', risk:0, estimatedTokens:200, estimatedMs:100,
  parameters:{type:'object',properties:{query:{type:'string',maxLength:120}},required:['query'],additionalProperties:false},
  execute:async ({query},signal)=>({query, matches: await trustedReadOnlyDocIndex(query,signal)})
});
// Pass `toolRegistry: registry` from your trusted SDK host. No model can register it.
```
