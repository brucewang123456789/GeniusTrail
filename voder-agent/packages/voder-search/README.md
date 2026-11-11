# Voder Search — Evidence-Gated Retrieval v2

The fifth-module search layer is **read-only and opt-in**: Pi's native repository search remains the default, followed by a narrowly bounded in-memory documentation index, and then Claude-style `Search` / focused `Fetch` **only when a real evidence gap or an explicit online user request warrants it**. All source content is untrusted; the native VGR commit authority is unchanged.

## What's new in v2 (in addition to the verified v1 interface)

- Trusted **host-side allowed/blocked domain policy** applies to both web result admission and subsequent source extraction. Domain options never enter the model's writable tool schema. HTTPS only; credential-bearing links, local/internal hosts and literal IP addresses are rejected; extracted pages must have been discovered by the same task. A hostname policy is *not* a substitute for network egress firewall rules when adding custom providers.
- Claude-like split of lightweight search snippets and focused full-page extraction remains **one stable `voder_ext_search_evidence` tool**, so Pi's native model loop and prompt/tool budget stay compact.
- Codex-inspired **bounded result cache**: positive TTL default 15m, negative TTL default 1m, at most 32 cached queries, LRU eviction; no cache after provider errors. Per-task web call limits still apply even to cached reads, while host provider requests are skipped on a hit.
- Fault isolation: sanitize injected provider exception text, bound remote page before hashing/excerpting, reject suspicious credential query parameters, and trip a fail-closed provider circuit after repeated failures. Local/offline repository tasks remain available if the provider is offline.
- Stable host-injected provider interface (`search(query,{signal})`, optional `extract(url,{focus,signal})`). The built-in Tavily adapter remains optional, not enabled by default. A different vetted, version-pinned adapter can be supplied **without** changing the Pi engine; no third-party adapter is automatically installed.

## Host setup

```js
import {VoderSDK} from './packages/voder-sdk/src/index.mjs';

// Local-only: no search API charges.
const sdkOffline = new VoderSDK({cwd:'/workspace', native:trustedNative,
  searchOptions:{allowWeb:false}});

// Optional, tightly scoped external source discovery; requires an explicit host API key.
const sdkDocs = new VoderSDK({cwd:'/workspace', native:trustedNative,
  searchOptions:{allowWeb:true, webMaxQueries:3,
    allowedDomains:['docs.python.org','nodejs.org'],
    webCacheTtlMs:900000, negativeCacheTtlMs:60000,
    providerFailureLimit:2}});
// export TAVILY_API_KEY=... in protected host environment; never paste into model prompt.
```

`allowedDomains` and `blockedDomains` cannot both be active. Allowed and blocked names are ASCII hostnames, not paths, URL schemes or wildcard patterns; an entry covers exact host and its subdomains. Source pages are not fetched directly by Voder; the pinned default provider uses static HTTPS Tavily search/extract endpoints with redirect refusal. Wiki performs lazy discovery of README and docs and stays capped at 96 documents / 1 MiB.

## Engineering and validation boundary

The source update modifies **only `voder-search` source, tests, README and one Pi host integration test**, not the Pi engine, VGR, Modal, Context, Tasks, Skills or SDK logic. User prohibitions override search; model invocation cannot grant network access; search results have `CONTEXT_ONLY` status, digest and timestamp and cannot authorize writes.

Offline tests cover permitted and denied domains, spoofed domains, source provenance, cache TTL/eviction, input/response limits, provider circuit, error redaction, source cancellation/timeout and Pi/VGR integration. **Not established without separate AutoDL tests**: real provider credentials/network, DeepSeek's autonomous query strategy, task reliability at production scale, benchmarking against stock Pi or Codex/Claude, or certified production superiority. Search-only improvements do not imply that the entire Voder product has completed A01–A12 acceptance.

## v3 keyless public Web Search / focused Fetch (no paid search API)

Default policy stays **offline**. When the trusted host explicitly sets `searchOptions.allowWeb=true`, and does **not** supply a paid provider or `apiKey`, Voder now uses the optional built-in `KeylessWebProvider` (`bing-rss`), without registering, entering a Tavily/Exa key, or changing Pi's local search loop.

```js
// Trusted host options, NOT a model-controlled argument:
searchOptions: {
  allowWeb: true,
  webMaxQueries: 3,
  keylessEngine: 'bing-rss',
  allowedDomains: ['docs.python.org', 'nodejs.org', 'github.com']
}
```

The **only** built-in zero-key discovery endpoint in this mode is `https://www.bing.com/search?q=...&format=rss`. Results are discovered first; the agent must request `mode=fetch` for a URL discovered in this task. The fetch adapter downloads a **single** bounded HTTPS document, rejects redirects, unsafe destination hosts, private/reserved DNS addresses (resolved and pinned at the connection), unsupported types and large content, and extracts only text. Source snippets and documents are always `UNTRUSTED_WEB_DATA`, with digest and timestamp, never authorization or verification. Abort, request budget, failure circuit, host domain policy and cache rules from v2 remain in force. No browser scripting, cookies, authentication, or automatic download/execution.

**Optional self-host search:** a trusted operator can run [SearXNG](https://docs.searxng.org/) separately with JSON responses enabled and choose `keylessEngine:'searxng'`, `searxngUrl:'http://127.0.0.1:8888'` (loopback only for plain HTTP), or a vetted HTTPS SearXNG endpoint. SearXNG is a third-party service and must be maintained and patched by its operator; JSON may be disabled by public instances. Voder does not install or automatically activate it. Existing Tavily provider is retained for backward compatibility but is **not required** for keyless mode.

**Production limitations:** Bing's RSS feed is a public, unversioned service with no guaranteed availability, rate limit, stability, ranking, licensing or SLA. Its owner may block automation or change its format. This is a best-effort, no-fee backend, not evidence of production-grade external search infrastructure. For managed production reliability, use a dedicated SearXNG instance/approved provider plus deployment egress policy, operational monitoring, and real-world canary tasks. Restrict allowedDomains where possible; application-level DNS/URL filters complement, but do not replace a host/network egress firewall.

**Validation:** the package provides offline unit/regression and safe failure-injection tests, plus a separate real AutoDL Search+Fetch command. A PASS of offline tests never substitutes for real network E2E, and real Search+Fetch alone does not establish LLM autonomous tool choice or comparable reliability versus other agents.


## Revision 15: user-owned cumulative request budgets

The installer preserves providers, keys and DNS settings and adds `budget` only
when absent. An empty budget object defaults every limit to null (unlimited).
This configured module policy replaces the legacy `webMaxQueries: 3` passed by
existing CLI/SDK/editor callers. No caller source needs modification. Without a
configured budget, legacy host limits retain their prior behavior. An explicit
host `budgetPolicy` argument takes priority over the configuration.

`search-config.json` accepts:

```json
"budget": {
  "max_requests": null,
  "max_search_requests": null,
  "max_fetch_requests": null,
  "task_timeout_ms": null
}
```

All fields are optional; null means no cumulative cap. Counts accept nonnegative
safe integers (0 disables that class); task_timeout_ms accepts positive integers
up to 2147483647. Example: total 100, search 20, fetch 80, time 900000. Counts
are per task and include metered supplier fallbacks; cache hits consume no new
request. A transport's redirect/DNS attempts remain governed by its existing
per-operation bounds, not counted as independent logical fetch requests.
When total is finite, one slot remains reserved for the first fetch unless
fetch is disabled. Budget checks and increments are synchronous before dispatch,
so concurrent calls cannot overdraw a configured limit. Task time starts at
newTask and propagates cancellation to external operations; it includes elapsed
model time. Policies are snapshotted when the service is constructed; reload
its host after changing settings. Neither null nor a larger cap grants web
permission, changes SSRF checks, bypasses tool-loop budgets, or disables timeout,
response size limits, cache bounds, provider cooldowns or user cancellation.
The separate agent orchestration limits are deliberately unchanged.

No supplier fee is guessed or automatically inferred. A finite request limit
can bound request counts; this revision does not enforce a monetary budget.
Do not share search-config.json or PRIVATE_PENDING_CONFIG.json: they may hold keys.
