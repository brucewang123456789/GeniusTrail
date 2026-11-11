/** Managed API plugin. Provider results are inert proposals; Daemon alone governs authorization. */
import {OpenAIResponsesProvider} from './responses.js';
import {
  OpenAICompatibleProvider, AnthropicProvider,
  type ModelProvider,
} from '../../voder-providers/dist/index.js';

export const CLOUD_PROTOCOL = 'voder-model-plugin-v1' as const;
export const CLOUD_PROVIDERS = Object.freeze({
  openai:    {kind: 'openai', endpoint: 'https://api.openai.com/v1', credential: 'OPENAI_API_KEY'},
  anthropic: {kind: 'anthropic', endpoint: 'https://api.anthropic.com', credential: 'ANTHROPIC_API_KEY'},
  gemini:    {kind: 'openai', endpoint: 'https://generativelanguage.googleapis.com/v1beta/openai', credential: 'GEMINI_API_KEY'},
  kimi:      {kind: 'openai', endpoint: 'https://api.moonshot.ai/v1', credential: 'KIMI_API_KEY'},
  glm:       {kind: 'openai', endpoint: 'https://open.bigmodel.cn/api/paas/v4', credential: 'GLM_API_KEY'},
  deepseek:  {kind: 'openai', endpoint: 'https://api.deepseek.com/v1', credential: 'DEEPSEEK_API_KEY'},
  grok:      {kind: 'openai', endpoint: 'https://api.x.ai/v1', credential: 'XAI_API_KEY'},
  qwen:      {kind: 'openai', endpoint: 'https://dashscope.aliyuncs.com/compatible-mode/v1', credential: 'DASHSCOPE_API_KEY'},
  openrouter:{kind: 'openai', endpoint: 'https://openrouter.ai/api/v1', credential: 'OPENROUTER_API_KEY'},
} as const);

type Env = Readonly<Record<string, string | undefined>>;
export type PluginResolution = Readonly<{pluginId: 'cloud'; model: string; provider: ModelProvider; maxOutputTokens: number}>;
function positiveInt(raw: string | undefined, fallback: number, low: number, high: number): number {
  const value = raw === undefined ? fallback : Number(raw);
  if (!Number.isSafeInteger(value) || value < low || value > high) throw new TypeError('invalid model output budget');
  return value;
}
export function createCloudProvider(env: Env): PluginResolution {
  const vendor = env.VODER_CLOUD_VENDOR;
  if (!vendor) throw new TypeError('VODER_CLOUD_VENDOR is required');
  if (vendor !== 'custom' && !(vendor in CLOUD_PROVIDERS)) throw new TypeError('unknown cloud vendor');
  const preset = vendor === 'custom' ? undefined : CLOUD_PROVIDERS[vendor as keyof typeof CLOUD_PROVIDERS];
  const kind = preset?.kind ?? 'openai';
  const endpoint = env.VODER_CLOUD_ENDPOINT ?? preset?.endpoint;
  const apiKey = env.VODER_CLOUD_API_KEY ?? (preset ? env[preset.credential] : undefined);
  const model = env.VODER_CLOUD_MODEL;
  if (!endpoint || !apiKey || !model) throw new TypeError('missing cloud endpoint, credential or model');
  if (/\s|[\u0000-\u001f]/u.test(model) || model.length > 256) throw new TypeError('invalid model name');
  const maxOutputTokens = positiveInt(env.VODER_CLOUD_MAX_OUTPUT_TOKENS, 1024, 1, 131072);
  const timeoutMs = positiveInt(env.VODER_CLOUD_TIMEOUT_MS, 60000, 1000, 180000);
  const cfg = {id: `cloud_${vendor}`, kind, endpoint, apiKey, timeoutMs, maxResponseBytes: 512*1024};
  const protocol = env.VODER_CLOUD_PROTOCOL ?? 'chat';
  if (!['chat','responses'].includes(protocol)) throw new TypeError('invalid cloud protocol');
  if (protocol === 'responses' && vendor !== 'openai' && vendor !== 'custom') throw new TypeError('Responses mode is only for OpenAI or explicitly configured custom endpoint');
  const provider = protocol === 'responses' ? new OpenAIResponsesProvider(cfg.id,endpoint,apiKey,timeoutMs) : kind === 'anthropic' ? new AnthropicProvider({...cfg,kind:'anthropic'}) :
    new OpenAICompatibleProvider({...cfg,kind:'openai'});
  return {pluginId:'cloud',provider,model,maxOutputTokens};
}
