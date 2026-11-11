import type {ResolvedConfiguration} from "./config.js";
import {ProviderError, fromHttpStatus} from "./errors.js";
export type ResponseLease = Readonly<{response: Response; release(): void; classify(error: unknown): ProviderError}>;
/** Each request has a dedicated deadline and cancellation scope; redirects are always denied. */
export async function beginRequest(cfg: ResolvedConfiguration, url: string, body: unknown, headers: Record<string, string>, signal?: AbortSignal): Promise<ResponseLease> {
  const timer = new AbortController();
  const timeout = setTimeout(() => timer.abort(new ProviderError("TIMEOUT", "provider deadline exceeded", true)), cfg.timeoutMs);
  const merged = signal ? AbortSignal.any([timer.signal, signal]) : timer.signal;
  let released = false;
  const release = () => {if (released) return; released = true; clearTimeout(timeout);};
  const classify = (error: unknown): ProviderError => {
    if (signal?.aborted) return new ProviderError("CANCELLED", "provider call cancelled");
    if (timer.signal.aborted) return new ProviderError("TIMEOUT", "provider deadline exceeded", true);
    if (error instanceof ProviderError) return error;
    return new ProviderError("NETWORK", "provider transport failure", true);
  };
  try {
    if (merged.aborted) throw merged.reason;
    const response = await cfg.fetchImpl(url, {
      method: "POST", headers: {"content-type": "application/json", ...headers}, body: JSON.stringify(body),
      signal: merged, redirect: "manual", cache: "no-store",
    });
    if (!response.ok) {
      release();
      await response.body?.cancel().catch(() => {});
      throw fromHttpStatus(response.status, response.headers.get("retry-after"));
    }
    return {response, release, classify};
  } catch (error) {
    release();
    // Upstream exception strings can contain credentials/host internals; do not include them.
    throw classify(error);
  }
}
export async function readBounded(response: Response, maxBytes: number): Promise<string> {
  if (!response.body) throw new ProviderError("MALFORMED_RESPONSE", "provider response has no body");
  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8", {fatal: true});
  let size = 0;
  const fragments: string[] = [];
  try {
    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      size += value.length;
      if (size > maxBytes) throw new ProviderError("RESPONSE_TOO_LARGE", "provider response exceeds limit");
      fragments.push(decoder.decode(value, {stream: true}));
    }
    fragments.push(decoder.decode());
    return fragments.join("");
  } catch (error) {
    if (error instanceof ProviderError) throw error;
    throw new ProviderError("MALFORMED_RESPONSE", "invalid provider response encoding or body");
  } finally {await reader.cancel().catch(() => {}); reader.releaseLock();}
}
export async function requestJson(cfg: ResolvedConfiguration, url: string, body: unknown, headers: Record<string,string>, signal?: AbortSignal): Promise<unknown> {
  const lease = await beginRequest(cfg, url, body, headers, signal);
  try {
    const data = await readBounded(lease.response, cfg.maxResponseBytes);
    try {return JSON.parse(data) as unknown;} catch {throw new ProviderError("MALFORMED_RESPONSE", "invalid JSON response");}
  } catch (error) {throw lease.classify(error);} finally {lease.release();}
}
