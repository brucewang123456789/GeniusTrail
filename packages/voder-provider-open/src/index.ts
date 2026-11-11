/** Self-hosted inference plugin. Runtime selection never executes model weights or shell commands. */
import {OpenAICompatibleProvider, type ModelProvider} from '../../voder-providers/dist/index.js';
export const OPEN_PROTOCOL = 'voder-model-plugin-v1' as const;
export const OPEN_RUNTIMES = Object.freeze(['vllm','sglang','ollama','llamacpp','compatible'] as const);
type Env = Readonly<Record<string, string | undefined>>;
export type PluginResolution = Readonly<{pluginId: 'open'; model: string; provider: ModelProvider; maxOutputTokens: number}>;
function positiveInt(raw: string | undefined, fallback: number, low: number, high: number): number {
  const value = raw === undefined ? fallback : Number(raw);
  if (!Number.isSafeInteger(value) || value < low || value > high) throw new TypeError('invalid model limit');
  return value;
}
export function createOpenProvider(env: Env): PluginResolution {
  const runtime = env.VODER_OPEN_RUNTIME;
  if (!runtime || !OPEN_RUNTIMES.some(v=>v===runtime)) throw new TypeError('unsupported inference runtime');
  const endpoint = env.VODER_OPEN_ENDPOINT;
  const model = env.VODER_OPEN_MODEL;
  if (!endpoint || !model) throw new TypeError('explicit inference endpoint and model are required');
  if (/\s|[\u0000-\u001f]/u.test(model) || model.length > 256) throw new TypeError('invalid model name');
  const apiKey = env.VODER_OPEN_API_KEY;
  const timeoutMs = positiveInt(env.VODER_OPEN_TIMEOUT_MS,60000,1000,180000);
  const maxOutputTokens = positiveInt(env.VODER_OPEN_MAX_OUTPUT_TOKENS,1024,1,131072);
  const provider = new OpenAICompatibleProvider({
    id:`open_${runtime}`,kind:'openai',endpoint,
    ...(apiKey?{apiKey}:{}),timeoutMs,maxResponseBytes:512*1024,
    allowLoopbackHttp:env.VODER_OPEN_ALLOW_LOOPBACK_HTTP==='1',
  });
  return {pluginId:'open',provider,model,maxOutputTokens};
}
