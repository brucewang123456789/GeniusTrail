/** Model providers propose candidate text/tool intents. They never grant tool authority or task success. */
export type Role = "system" | "user" | "assistant" | "tool";
export type ModelMessage = Readonly<{
  role: Role;
  content: string;
  name?: string;
  toolCallId?: string;
}>;
export type ToolDefinition = Readonly<{
  name: string;
  description: string;
  parameters: Readonly<Record<string, unknown>>;
}>;
export type GenerationRequest = Readonly<{
  model: string;
  messages: readonly ModelMessage[];
  maxOutputTokens: number;
  temperature?: number;
  tools?: readonly ToolDefinition[];
  signal?: AbortSignal;
}>;
export type TokenUsage = Readonly<{
  inputTokens: number;
  outputTokens: number;
  totalTokens: number;
  source: "provider";
}>;
export type ProposedToolCall = Readonly<{
  id: string;
  name: string;
  argumentsJson: string;
}>;
export type FinishReason = "stop" | "length" | "tool_calls" | "other";
export type GenerationResult = Readonly<{
  providerId: string;
  model: string;
  text: string;
  toolCalls: readonly ProposedToolCall[];
  finishReason: FinishReason;
  usage?: TokenUsage;
}>;
export type StreamEvent =
  | Readonly<{kind: "text_delta"; text: string}>
  | Readonly<{kind: "tool_delta"; index: number; id?: string; name?: string; argumentsDelta?: string}>
  | Readonly<{kind: "usage"; usage: TokenUsage}>
  | Readonly<{kind: "done"; result: GenerationResult}>;
export type ProviderCapabilities = Readonly<{
  streaming: boolean;
  toolCalls: boolean;
  usageReporting: boolean;
}>;
export interface ModelProvider {
  readonly id: string;
  capabilities(): ProviderCapabilities;
  generate(request: GenerationRequest): Promise<GenerationResult>;
  stream(request: GenerationRequest): AsyncIterable<StreamEvent>;
  /** Cancellation is owned by caller's AbortSignal; no global cancellation of other requests. */
  cancel(controller: AbortController): void;
}
export function validateRequest(request: GenerationRequest): void {
  if (!request.model || request.model.length > 256 || /[\u0000-\u001f]/u.test(request.model)) throw new TypeError("invalid model id");
  if (!Array.isArray(request.messages) || request.messages.length === 0 || request.messages.length > 256) throw new TypeError("invalid messages");
  if (!Number.isSafeInteger(request.maxOutputTokens) || request.maxOutputTokens < 1 || request.maxOutputTokens > 131072) throw new TypeError("invalid output budget");
  if (request.temperature !== undefined && (!Number.isFinite(request.temperature) || request.temperature < 0 || request.temperature > 2)) throw new TypeError("invalid temperature");
  let size = 0;
  for (const message of request.messages) {
    if (!message || !["system", "user", "assistant", "tool"].includes(message.role) || typeof message.content !== "string") throw new TypeError("invalid message");
    if (message.role === "tool" && !message.toolCallId) throw new TypeError("tool result must reference call id");
    if (message.toolCallId !== undefined && (message.toolCallId.length > 256 || !message.toolCallId)) throw new TypeError("invalid call id");
    size += new TextEncoder().encode(message.content).length;
  }
  if (size > 2 * 1024 * 1024) throw new TypeError("context exceeds 2 MiB");
  if ((request.tools?.length ?? 0) > 32) throw new TypeError("too many tools");
  for (const tool of request.tools ?? []) {
    if (!/^[A-Za-z0-9_-]{1,64}$/u.test(tool.name) || tool.description.length > 4096 || !tool.parameters || typeof tool.parameters !== "object" || Array.isArray(tool.parameters)) throw new TypeError("invalid tool schema");
    if (new TextEncoder().encode(JSON.stringify(tool)).length > 65536) throw new TypeError("tool schema too large");
  }
}
export function normalizeUsage(value: unknown): TokenUsage | undefined {
  if (!value || typeof value !== "object") return undefined;
  const o = value as Record<string, unknown>;
  const input = o["prompt_tokens"] ?? o["input_tokens"];
  const output = o["completion_tokens"] ?? o["output_tokens"];
  if (!Number.isSafeInteger(input) || !Number.isSafeInteger(output) || (input as number) < 0 || (output as number) < 0) return undefined;
  const total = (input as number) + (output as number);
  if (!Number.isSafeInteger(total)) return undefined;
  return {inputTokens: input as number, outputTokens: output as number, totalTokens: total, source: "provider"};
}
