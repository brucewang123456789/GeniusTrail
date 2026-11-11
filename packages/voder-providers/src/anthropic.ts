import {endpointPath, resolveConfiguration, type ProviderConfiguration, type ResolvedConfiguration} from "./config.js";
import {normalizeUsage, validateRequest, type FinishReason, type GenerationRequest, type GenerationResult, type ModelProvider, type ProposedToolCall, type ProviderCapabilities, type StreamEvent, type TokenUsage} from "./contracts.js";
import {ProviderError} from "./errors.js";
import {jsonFrame, parseSse} from "./sse.js";
import {beginRequest, requestJson} from "./transport.js";
type Obj = Record<string, unknown>;
function obj(value: unknown): Obj {if (!value || typeof value !== "object" || Array.isArray(value)) throw new ProviderError("MALFORMED_RESPONSE", "expected object"); return value as Obj;}
function str(value: unknown): string {return typeof value === "string" ? value : "";}
function finish(value: unknown): FinishReason {return value === "end_turn" ? "stop" : value === "max_tokens" ? "length" : value === "tool_use" ? "tool_calls" : "other";}
function jsonArgs(value: unknown): string {const item = obj(value); return JSON.stringify(item);}
function reqBody(req: GenerationRequest, stream: boolean): Obj {
  if (req.messages.some(m => m.role === "tool")) throw new ProviderError("UNSUPPORTED", "Anthropic tool results require explicit tool_result mapping; never silently coerce");
  const system = req.messages.filter(m => m.role === "system").map(m => m.content).join("\n\n");
  const messages = req.messages.filter(m => m.role !== "system").map(m => ({role: m.role, content: m.content}));
  if (!messages.length) throw new ProviderError("BAD_REQUEST", "Anthropic requires at least one conversation message");
  return {model: req.model, max_tokens: req.maxOutputTokens, messages, ...(system ? {system} : {}),
    ...(req.temperature !== undefined ? {temperature: req.temperature} : {}),
    ...(req.tools?.length ? {tools: req.tools.map(t => ({name: t.name, description: t.description, input_schema: t.parameters}))} : {}), stream};
}
export class AnthropicProvider implements ModelProvider {
  readonly id: string;
  readonly #cfg: ResolvedConfiguration;
  constructor(config: ProviderConfiguration) {
    if (config.kind !== "anthropic") throw new TypeError("adapter kind must be anthropic");
    this.#cfg = resolveConfiguration(config); this.id = config.id;
    if (!config.apiKey) throw new ProviderError("INVALID_CONFIGURATION", "Anthropic API key is required");
  }
  capabilities(): ProviderCapabilities {return {streaming: true, toolCalls: true, usageReporting: true};}
  cancel(controller: AbortController): void {controller.abort();}
  #headers(): Record<string,string> {return {"x-api-key": this.#cfg.apiKey!, "anthropic-version": "2023-06-01"};}
  #url(): string {return endpointPath(this.#cfg, "v1/messages");}
  async generate(req: GenerationRequest): Promise<GenerationResult> {
    validateRequest(req);
    const raw = obj(await requestJson(this.#cfg, this.#url(), reqBody(req, false), this.#headers(), req.signal));
    const blocks = raw["content"];
    if (!Array.isArray(blocks) || blocks.length > 64) throw new ProviderError("MALFORMED_RESPONSE", "invalid Anthropic content");
    let text = ""; const toolCalls: ProposedToolCall[] = [];
    for (const item of blocks) {
      const block = obj(item);
      if (block["type"] === "text") text += str(block["text"]);
      else if (block["type"] === "tool_use") {
        if (!str(block["id"]) || !str(block["name"])) throw new ProviderError("MALFORMED_RESPONSE", "invalid tool use");
        toolCalls.push({id: str(block["id"]), name: str(block["name"]), argumentsJson: jsonArgs(block["input"])});
      } else throw new ProviderError("UNSUPPORTED", "unsupported Anthropic content block type");
    }
    const model = str(raw["model"]);
    if (!model) throw new ProviderError("MALFORMED_RESPONSE", "missing model");
    const usage = normalizeUsage(raw["usage"]);
    return {providerId: this.id, model, text, toolCalls, finishReason: finish(raw["stop_reason"]), ...(usage ? {usage} : {})};
  }
  async *stream(req: GenerationRequest): AsyncIterable<StreamEvent> {
    validateRequest(req);
    const lease = await beginRequest(this.#cfg, this.#url(), reqBody(req, true), this.#headers(), req.signal);
    let model = ""; let text = ""; let stop: FinishReason = "other";
    let messageStopped = false; let usage: TokenUsage | undefined;
    let input = 0; let output = 0;
    const blocks = new Map<number,{kind: "text" | "tool"; id: string; name: string; args: string}>();
    try {
      for await (const frame of parseSse(lease.response, this.#cfg.maxResponseBytes)) {
        const event = jsonFrame(frame.data);
        const type = str(event["type"]);
        if (type === "error") throw new ProviderError("UPSTREAM_UNAVAILABLE", "Anthropic stream error", true);
        if (type === "message_start") {
          const msg = obj(event["message"]); model = str(msg["model"]);
          const u = normalizeUsage(msg["usage"]);
          if (u) {input = u.inputTokens; output = u.outputTokens;}
        } else if (type === "content_block_start") {
          const index = event["index"];
          if (!Number.isSafeInteger(index) || (index as number) < 0 || (index as number) > 63 || blocks.has(index as number)) throw new ProviderError("MALFORMED_RESPONSE", "invalid block index");
          const block = obj(event["content_block"]);
          if (block["type"] === "tool_use") {
            const id = str(block["id"]); const name = str(block["name"]);
            if (!id || !name) throw new ProviderError("MALFORMED_RESPONSE", "invalid tool use");
            blocks.set(index as number, {kind:"tool", id, name, args:""});
            yield {kind:"tool_delta", index: index as number, id, name};
          } else if (block["type"] === "text") blocks.set(index as number, {kind:"text", id:"",name:"",args:""});
          else throw new ProviderError("UNSUPPORTED", "unsupported Anthropic content block type");
        } else if (type === "content_block_delta") {
          const index = event["index"];
          const block = typeof index === "number" ? blocks.get(index) : undefined;
          if (!block) throw new ProviderError("MALFORMED_RESPONSE", "unknown stream block");
          const delta = obj(event["delta"]);
          if (block.kind === "text" && delta["type"] === "text_delta") {
            if (typeof delta["text"] !== "string") throw new ProviderError("MALFORMED_RESPONSE", "invalid text delta");
            text += delta["text"];
            if (text.length > this.#cfg.maxResponseBytes) throw new ProviderError("RESPONSE_TOO_LARGE", "text exceeds limit");
            yield {kind:"text_delta", text:delta["text"]};
          } else if (block.kind === "tool" && delta["type"] === "input_json_delta") {
            if (typeof delta["partial_json"] !== "string") throw new ProviderError("MALFORMED_RESPONSE", "invalid tool delta");
            block.args += delta["partial_json"];
            if (block.args.length > 1024 * 1024) throw new ProviderError("RESPONSE_TOO_LARGE", "tool input exceeds limit");
            yield {kind:"tool_delta", index: index as number, argumentsDelta: delta["partial_json"]};
          } else throw new ProviderError("UNSUPPORTED", "unexpected Anthropic delta type");
        } else if (type === "message_delta") {
          const delta = obj(event["delta"]); stop = finish(delta["stop_reason"]);
          const partialUsage = event["usage"];
          if (partialUsage !== undefined) {
            const u = obj(partialUsage);
            if (Number.isSafeInteger(u["output_tokens"]) && (u["output_tokens"] as number) >= 0) output = u["output_tokens"] as number;
          }
        } else if (type === "message_stop") {messageStopped = true; break;}
        else if (type === "ping" || type === "content_block_stop") { /* no-op */ }
        else throw new ProviderError("UNSUPPORTED", "unknown Anthropic event");
      }
      if (!messageStopped || !model) throw new ProviderError("MALFORMED_RESPONSE", "incomplete Anthropic stream");
      const calls: ProposedToolCall[] = [];
      for (const [, block] of [...blocks.entries()].sort((a,b) => a[0]-b[0])) {
        if (block.kind === "tool") {
          let args: unknown;
          try {args = JSON.parse(block.args || "{}");} catch {throw new ProviderError("MALFORMED_RESPONSE", "invalid tool JSON");}
          calls.push({id:block.id,name:block.name,argumentsJson:jsonArgs(args)});
        }
      }
      if (input || output) {usage = {inputTokens:input, outputTokens:output, totalTokens:input+output, source:"provider"}; yield {kind:"usage",usage};}
      yield {kind:"done",result:{providerId:this.id,model,text,toolCalls:calls,finishReason:stop,...(usage ? {usage} : {})}};
    } catch (error) {throw lease.classify(error);} finally {await lease.response.body?.cancel().catch(() => {}); lease.release();}
  }
}
