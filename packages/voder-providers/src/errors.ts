export type ProviderErrorCode =
  | "INVALID_CONFIGURATION" | "BAD_REQUEST" | "AUTHENTICATION" | "RATE_LIMITED"
  | "UPSTREAM_UNAVAILABLE" | "TIMEOUT" | "CANCELLED" | "MALFORMED_RESPONSE"
  | "RESPONSE_TOO_LARGE" | "UNSAFE_REDIRECT" | "NETWORK" | "UNSUPPORTED";
export class ProviderError extends Error {
  constructor(
    public readonly code: ProviderErrorCode,
    message: string,
    public readonly retryable = false,
    public readonly status?: number,
    public readonly retryAfterMs?: number,
  ) {
    super(message);
    this.name = "ProviderError";
  }
}
export function fromHttpStatus(status: number, retryAfter: string | null): ProviderError {
  const delay = parseRetryAfter(retryAfter);
  if (status === 401 || status === 403) return new ProviderError("AUTHENTICATION", `provider HTTP ${status}`, false, status);
  if (status === 429) return new ProviderError("RATE_LIMITED", "provider rate limited", true, status, delay);
  if (status === 408 || status >= 500) return new ProviderError("UPSTREAM_UNAVAILABLE", `provider HTTP ${status}`, true, status, delay);
  if (status >= 300 && status < 400) return new ProviderError("UNSAFE_REDIRECT", "provider redirect refused", false, status);
  return new ProviderError("BAD_REQUEST", `provider HTTP ${status}`, false, status);
}
function parseRetryAfter(header: string | null): number | undefined {
  if (!header) return undefined;
  if (/^\d{1,8}$/u.test(header)) return Math.min(Number(header) * 1000, 86_400_000);
  const value = Date.parse(header);
  if (!Number.isFinite(value)) return undefined;
  return Math.max(0, Math.min(value - Date.now(), 86_400_000));
}
