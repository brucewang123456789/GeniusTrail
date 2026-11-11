import {ProviderError} from "./errors.js";
export type SseFrame = Readonly<{event: string; data: string; id?: string}>;
/** Streaming UTF-8 decoder with hard limits on bytes, logical lines and frame lengths. */
export async function* parseSse(response: Response, maxBytes: number): AsyncGenerator<SseFrame> {
  if (!response.body) throw new ProviderError("MALFORMED_RESPONSE", "SSE body missing");
  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8", {fatal: true});
  let pending = "";
  let data: string[] = [];
  let dataLength = 0;
  let event = "message";
  let id: string | undefined;
  let size = 0;
  const MAX_LINE = 512 * 1024;
  const MAX_EVENT = 1024 * 1024;
  function line(raw: string): SseFrame | undefined {
    const text = raw.endsWith("\r") ? raw.slice(0, -1) : raw;
    if (text === "") {
      if (!data.length) {event = "message"; id = undefined; return;}
      const emitted = {event, data: data.join("\n"), ...(id === undefined ? {} : {id})};
      data = []; dataLength = 0; event = "message"; id = undefined;
      return emitted;
    }
    if (text.startsWith(":")) return;
    const colon = text.indexOf(":");
    const field = colon < 0 ? text : text.slice(0, colon);
    const value = colon < 0 ? "" : text.slice(colon + 1).replace(/^ /u, "");
    if (field === "data") {
      if (dataLength + value.length > MAX_EVENT) throw new ProviderError("RESPONSE_TOO_LARGE", "SSE event exceeds limit");
      data.push(value); dataLength += value.length;
    } else if (field === "event") event = value;
    else if (field === "id" && !value.includes("\0")) id = value;
  }
  try {
    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      size += value.length;
      if (size > maxBytes) throw new ProviderError("RESPONSE_TOO_LARGE", "SSE stream exceeds limit");
      pending += decoder.decode(value, {stream: true});
      let cut: number;
      while ((cut = pending.indexOf("\n")) >= 0) {
        const raw = pending.slice(0, cut);
        pending = pending.slice(cut + 1);
        if (raw.length > MAX_LINE) throw new ProviderError("RESPONSE_TOO_LARGE", "SSE line exceeds limit");
        const frame = line(raw);
        if (frame) yield frame;
      }
      if (pending.length > MAX_LINE) throw new ProviderError("RESPONSE_TOO_LARGE", "SSE line exceeds limit");
    }
    pending += decoder.decode();
    if (pending.length > 0) {
      if (pending.length > MAX_LINE) throw new ProviderError("RESPONSE_TOO_LARGE", "SSE line exceeds limit");
      const frame = line(pending);
      if (frame) yield frame;
    }
    const frame = line("");
    if (frame) yield frame;
  } catch (error) {
    if (error instanceof ProviderError) throw error;
    throw new ProviderError("MALFORMED_RESPONSE", "SSE transport or UTF-8 failure");
  } finally {await reader.cancel().catch(() => {}); reader.releaseLock();}
}
export function jsonFrame(data: string): Record<string, unknown> {
  try {
    const parsed: unknown = JSON.parse(data);
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) throw new Error();
    return parsed as Record<string, unknown>;
  } catch {throw new ProviderError("MALFORMED_RESPONSE", "invalid SSE JSON frame");}
}
