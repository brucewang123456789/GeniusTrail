import {endpointPath, resolveConfiguration, type ProviderConfiguration, type ResolvedConfiguration} from "./config.js";
import {normalizeUsage, validateRequest, type FinishReason, type GenerationRequest, type GenerationResult, type ModelProvider, type ProposedToolCall, type ProviderCapabilities, type StreamEvent, type TokenUsage} from "./contracts.js";
import {ProviderError} from "./errors.js";
import {jsonFrame, parseSse} from "./sse.js";
import {beginRequest, requestJson} from "./transport.js";

type Obj = Record<string, unknown>;
function obj(value: unknown): Obj {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new ProviderError("MALFORMED_RESPONSE", "expected JSON object");
  return value as Obj;
}
function str(value: unknown): string {return typeof value === "string" ? value : "";}
function reason(value: unknown): FinishReason {
  if (value === "stop") return "stop";
  if (value === "length") return "length";
  if (value === "tool_calls" || value === "function_call") return "tool_calls";
  return "other";
}
function toolList(value: unknown): ProposedToolCall[] {
  if (value === undefined) return [];
  if (!Array.isArray(value) || value.length > 32) throw new ProviderError("MALFORMED_RESPONSE", "invalid tool call list");
  return value.map((item: unknown) => {
    const call = obj(item); const func = obj(call["function"]);
    const id = str(call["id"]); const name = str(func["name"]); const args = str(func["arguments"]);
    if (!id || !name || args.length > 1024 * 1024) throw new ProviderError("MALFORMED_RESPONSE", "incomplete tool call");
    ensureObjectJson(args);
    return {id, name, argumentsJson: args};
  });
}
function ensureObjectJson(data: string): void {
  try {obj(JSON.parse(data) as unknown);} catch {throw new ProviderError("MALFORMED_RESPONSE", "invalid tool arguments JSON");}
}
function buildRequest(req: GenerationRequest, stream: boolean): Obj {
  const tools = req.tools?.map(t => ({type: "function", function: {name: t.name, description: t.description, parameters: t.parameters}}));
  return {
    model: req.model,
    messages: req.messages.map(m => ({role: m.role, content: m.content, ...(m.name ? {name: m.name} : {}), ...(m.toolCallId ? {tool_call_id: m.toolCallId} : {})})),
    max_tokens: req.maxOutputTokens, ...(req.temperature !== undefined ? {temperature: req.temperature} : {}),
    ...(tools?.length ? {tools} : {}), stream,
    ...(stream ? {stream_options: {include_usage: true}} : {}),
  };
}
export class OpenAICompatibleProvider implements ModelProvider {
  readonly id: string;
  readonly #cfg: ResolvedConfiguration;
  constructor(configuration: ProviderConfiguration) {
    if (configuration.kind !== "openai") throw new TypeError("adapter kind must be openai");
    this.#cfg = resolveConfiguration(configuration); this.id = this.#cfg.id;
  }
  capabilities(): ProviderCapabilities {return {streaming: true, toolCalls: true, usageReporting: true};}
  cancel(controller: AbortController): void {controller.abort();}
  #headers(): Record<string, string> {return this.#cfg.apiKey ? {authorization: `Bearer ${this.#cfg.apiKey}`} : {};}
  #url(): string {return endpointPath(this.#cfg, "chat/completions");}
  async generate(req: GenerationRequest): Promise<GenerationResult> {
    validateRequest(req);
    const raw = obj(await requestJson(this.#cfg, this.#url(), buildRequest(req, false), this.#headers(), req.signal));
    const choices = raw["choices"];
    if (!Array.isArray(choices) || choices.length !== 1) throw new ProviderError("MALFORMED_RESPONSE", "expected exactly one choice");
    const choice = obj(choices[0]); const message = obj(choice["message"]);
    const content = message["content"];
    if (content !== null && typeof content !== "string") throw new ProviderError("MALFORMED_RESPONSE", "invalid assistant content");
    const model = str(raw["model"]);
    if (!model) throw new ProviderError("MALFORMED_RESPONSE", "missing actual model");
    const tools = toolList(message["tool_calls"]);
    const usage = normalizeUsage(raw["usage"]);
    return {providerId: this.id, model, text: content ?? "", toolCalls: tools, finishReason: reason(choice["finish_reason"]), ...(usage ? {usage} : {})};
  }
  async *stream(req: GenerationRequest): AsyncIterable<StreamEvent> {
    validateRequest(req);
    const lease = await beginRequest(this.#cfg, this.#url(), buildRequest(req, true), this.#headers(), req.signal);
    let ended = false; let completed = false; let text = ""; let finish: FinishReason = "other";
    let model = ""; let usage: TokenUsage | undefined;
    const calls = new Map<number, {id: string; name: string; args: string}>();
    try {
      for await (const frame of parseSse(lease.response, this.#cfg.maxResponseBytes)) {
        if (frame.data === "[DONE]") {ended = true; break;}
        const event = jsonFrame(frame.data);
        if (event["error"] !== undefined) throw new ProviderError("UPSTREAM_UNAVAILABLE", "provider streamed error", true);
        if (typeof event["model"] === "string") model = event["model"];
        const nextUsage = normalizeUsage(event["usage"]);
        if (nextUsage) {usage = nextUsage; yield {kind: "usage", usage};}
        const choices = event["choices"];
        if (!Array.isArray(choices) || choices.length > 1) throw new ProviderError("MALFORMED_RESPONSE", "invalid streaming choices");
        if (choices.length === 0) continue; // usage-only frame
        const choice = obj(choices[0]);
        if (choice["finish_reason"] !== null && choice["finish_reason"] !== undefined) {
          finish = reason(choice["finish_reason"]); completed = true;
        }
        if (choice["delta"] === undefined) continue;
        const delta = obj(choice["delta"]);
        if (typeof delta["content"] === "string" && delta["content"]) {
          text += delta["content"];
          if (text.length > this.#cfg.maxResponseBytes) throw new ProviderError("RESPONSE_TOO_LARGE", "generated content too large");
          yield {kind: "text_delta", text: delta["content"]};
        }
        if (delta["tool_calls"] !== undefined) {
          if (!Array.isArray(delta["tool_calls"])) throw new ProviderError("MALFORMED_RESPONSE", "invalid tool call delta");
          for (const part of delta["tool_calls"] as unknown[]) {
            const call = obj(part);
            const index = call["index"];
            if (!Number.isSafeInteger(index) || (index as number) < 0 || (index as number) > 31) throw new ProviderError("MALFORMED_RESPONSE", "invalid tool call index");
            const n = index as number;
            const earlier = calls.get(n) ?? {id: "", name: "", args: ""};
            if (call["id"] !== undefined) {
              if (typeof call["id"] !== "string" || (earlier.id && earlier.id !== call["id"])) throw new ProviderError("MALFORMED_RESPONSE", "unstable tool call id");
              earlier.id = call["id"];
            }
            let name: string | undefined; let argumentsDelta: string | undefined;
            if (call["function"] !== undefined) {
              const fn = obj(call["function"]);
              if (fn["name"] !== undefined) {
                if (typeof fn["name"] !== "string" || (earlier.name && earlier.name !== fn["name"])) throw new ProviderError("MALFORMED_RESPONSE", "unstable function name");
                earlier.name = fn["name"]; name = earlier.name;
              }
              if (fn["arguments"] !== undefined) {
                if (typeof fn["arguments"] !== "string") throw new ProviderError("MALFORMED_RESPONSE", "invalid argument delta");
                earlier.args += fn["arguments"]; argumentsDelta = fn["arguments"];
              }
            }
            if (earlier.args.length > 1024 * 1024) throw new ProviderError("RESPONSE_TOO_LARGE", "tool arguments exceed limit");
            calls.set(n, earlier);
            yield {kind: "tool_delta", index: n, ...(earlier.id ? {id: earlier.id} : {}), ...(name ? {name} : {}), ...(argumentsDelta !== undefined ? {argumentsDelta} : {})};
          }
        }
      }
      if (!ended || !completed || !model) throw new ProviderError("MALFORMED_RESPONSE", "incomplete OpenAI stream");
      const sorted: ProposedToolCall[] = [];
      for (const [, call] of [...calls.entries()].sort((a,b) => a[0] - b[0])) {
        if (!call.id || !call.name) throw new ProviderError("MALFORMED_RESPONSE", "incomplete tool call");
        ensureObjectJson(call.args);
        sorted.push({id: call.id, name: call.name, argumentsJson: call.args});
      }
      yield {kind: "done", result: {providerId: this.id, model, text, toolCalls: sorted, finishReason: finish, ...(usage ? {usage} : {})}};
    } catch (error) {throw lease.classify(error);} finally {await lease.response.body?.cancel().catch(() => {}); lease.release();}
  }
}
