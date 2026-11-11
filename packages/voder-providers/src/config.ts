import {ProviderError} from "./errors.js";
export type AdapterKind = "openai" | "anthropic";
export type ProviderConfiguration = Readonly<{
  id: string;
  kind: AdapterKind;
  endpoint: string;
  apiKey?: string;
  /** Loopback HTTP requires explicit opt-in; never permitted for non-loopback hosts. */
  allowLoopbackHttp?: boolean;
  timeoutMs?: number;
  maxResponseBytes?: number;
  /** Optional transport injection for mock servers and user-selected networking. */
  fetchImpl?: typeof fetch;
}>;
export type ResolvedConfiguration = Readonly<{
  id: string; kind: AdapterKind; endpoint: URL; apiKey?: string;
  timeoutMs: number; maxResponseBytes: number; fetchImpl: typeof fetch;
}>;
export function resolveConfiguration(cfg: ProviderConfiguration): ResolvedConfiguration {
  if (!/^[a-z][a-z0-9_-]{0,63}$/u.test(cfg.id)) throw new ProviderError("INVALID_CONFIGURATION", "invalid provider id");
  if (cfg.kind !== "openai" && cfg.kind !== "anthropic") throw new ProviderError("INVALID_CONFIGURATION", "invalid adapter");
  let url: URL;
  try { url = new URL(cfg.endpoint); } catch { throw new ProviderError("INVALID_CONFIGURATION", "invalid endpoint"); }
  const loopback = ["localhost", "127.0.0.1", "[::1]"].includes(url.hostname);
  if (!(url.protocol === "https:" || (url.protocol === "http:" && loopback && cfg.allowLoopbackHttp === true))) throw new ProviderError("INVALID_CONFIGURATION", "HTTPS required; HTTP allowed only for explicitly enabled loopback");
  if (url.username || url.password || url.search || url.hash) throw new ProviderError("INVALID_CONFIGURATION", "endpoint credentials/query/fragment forbidden");
  if (url.pathname.includes("..")) throw new ProviderError("INVALID_CONFIGURATION", "invalid endpoint path");
  if (cfg.apiKey !== undefined && (cfg.apiKey.length === 0 || /[\r\n]/u.test(cfg.apiKey))) throw new ProviderError("INVALID_CONFIGURATION", "invalid API key");
  const timeoutMs = cfg.timeoutMs ?? 60_000;
  const maxResponseBytes = cfg.maxResponseBytes ?? (8 * 1024 * 1024);
  if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 10 || timeoutMs > 300_000) throw new ProviderError("INVALID_CONFIGURATION", "timeout out of range");
  if (!Number.isSafeInteger(maxResponseBytes) || maxResponseBytes < 1024 || maxResponseBytes > 64 * 1024 * 1024) throw new ProviderError("INVALID_CONFIGURATION", "response limit out of range");
  if (cfg.fetchImpl !== undefined && typeof cfg.fetchImpl !== "function") throw new ProviderError("INVALID_CONFIGURATION", "invalid fetch implementation");
  return {id: cfg.id, kind: cfg.kind, endpoint: url, ...(cfg.apiKey ? {apiKey: cfg.apiKey} : {}), timeoutMs, maxResponseBytes, fetchImpl: cfg.fetchImpl ?? fetch};
}
export function endpointPath(cfg: ResolvedConfiguration, path: string): string {
  const prefix = cfg.endpoint.pathname.replace(/\/+$/u, "");
  // Caller config points to API root, not arbitrary caller-provided request URL.
  return new URL(`${prefix}/${path.replace(/^\/+/, "")}`, cfg.endpoint.origin).href;
}
