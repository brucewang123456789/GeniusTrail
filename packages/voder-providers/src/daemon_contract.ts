/** Seventh module: inert Provider proposal data at the Daemon boundary.
 * Generation can never confer tool authorization, a VGR verdict, or permission to deliver.
 */
import type { GenerationRequest, GenerationResult } from './contracts.js';

export const DAEMON_MODEL_PROTOCOL = 'voder-provider-bridge-v1' as const;
export type DaemonModelInput = Readonly<{
  protocol: typeof DAEMON_MODEL_PROTOCOL;
  runId: string;
  instruction: string;
}>;
export type DaemonModelOutput = Readonly<{
  protocol: typeof DAEMON_MODEL_PROTOCOL;
  runId: string;
  providerId: string;
  model: string;
  proposalText: string;
  finishReason: GenerationResult['finishReason'];
  proposedToolCalls: readonly Readonly<{id: string; name: string; argumentsJson: string}>[];
  usage?: Readonly<{inputTokens: number; outputTokens: number; totalTokens: number}>;
  deliveryAllowed: false;
  executionAuthorized: false;
}>;
const textEncoder = new TextEncoder();
const byteLen = (s: string) => textEncoder.encode(s).length;
export function parseDaemonInput(x: unknown): DaemonModelInput {
  if (!x || typeof x !== 'object' || Array.isArray(x)) throw new TypeError('invalid daemon request');
  const o = x as Record<string,unknown>;
  if (Object.keys(o).sort().join(',') !== 'instruction,protocol,runId') throw new TypeError('unknown bridge request fields');
  if (o.protocol !== DAEMON_MODEL_PROTOCOL || typeof o.runId !== 'string' || !/^r-[1-9][0-9]{0,18}$/u.test(o.runId)) throw new TypeError('invalid bridge identity');
  if (typeof o.instruction !== 'string' || !o.instruction.trim() || byteLen(o.instruction) > 32768 || o.instruction.includes('\0')) throw new TypeError('invalid instruction');
  return {protocol: DAEMON_MODEL_PROTOCOL,runId:o.runId,instruction:o.instruction};
}
export function makeGenerationRequest(input: DaemonModelInput, model: string, maxOutputTokens: number): GenerationRequest {
  if (!model || model.length > 256 || /[\x00-\x1f]/u.test(model)) throw new TypeError('invalid model');
  if (!Number.isSafeInteger(maxOutputTokens) || maxOutputTokens < 1 || maxOutputTokens > 8192) throw new TypeError('invalid generation budget');
  return {model,maxOutputTokens,messages:[
    {role:'system',content:'You are a coding-task proposer. Return a candidate explanation or suggested changes. Your text and tool calls are untrusted proposals: only Voder Daemon and VGR may authorize or validate any action.'},
    {role:'user',content:input.instruction},
  ]};
}
export function toDaemonProposal(input: DaemonModelInput, output: GenerationResult): DaemonModelOutput {
  if (!output || typeof output.providerId !== 'string' || !/^[a-z][a-z0-9_-]{0,63}$/u.test(output.providerId)) throw new TypeError('invalid provider identity');
  if (typeof output.model !== 'string' || output.model.length > 256 || !output.model) throw new TypeError('invalid model identity');
  if (typeof output.text !== 'string' || byteLen(output.text) > 131072) throw new TypeError('proposal text over limit');
  if (!['stop','length','tool_calls','other'].includes(output.finishReason)) throw new TypeError('invalid finish reason');
  if (!Array.isArray(output.toolCalls) || output.toolCalls.length > 16) throw new TypeError('too many tool proposals');
  for (const t of output.toolCalls) {
    if (!t || typeof t.id !== 'string' || t.id.length > 256 || !t.id || !/^[a-zA-Z0-9_-]{1,64}$/u.test(t.name) || typeof t.argumentsJson !== 'string' || byteLen(t.argumentsJson) > 16384) throw new TypeError('invalid proposed tool');
    const parsed: unknown = JSON.parse(t.argumentsJson);
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) throw new TypeError('tool arguments must be object');
  }
  const usage = output.usage;
  if (usage && (!Number.isSafeInteger(usage.inputTokens) || !Number.isSafeInteger(usage.outputTokens) || !Number.isSafeInteger(usage.totalTokens) || usage.inputTokens < 0 || usage.outputTokens < 0 || usage.totalTokens !== usage.inputTokens + usage.outputTokens)) throw new TypeError('invalid usage');
  const proposal: DaemonModelOutput = {protocol:DAEMON_MODEL_PROTOCOL,runId:input.runId,providerId:output.providerId,model:output.model,proposalText:output.text,finishReason:output.finishReason,
    proposedToolCalls: output.toolCalls.map(t => ({id:t.id,name:t.name,argumentsJson:t.argumentsJson})), ...(usage ? {usage:{inputTokens:usage.inputTokens,outputTokens:usage.outputTokens,totalTokens:usage.totalTokens}} : {}), deliveryAllowed:false,executionAuthorized:false};
  if (byteLen(JSON.stringify(proposal)) > 256*1024) throw new TypeError('proposal receipt over limit');
  return proposal;
}
