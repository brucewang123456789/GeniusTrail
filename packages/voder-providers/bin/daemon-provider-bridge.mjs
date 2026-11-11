#!/usr/bin/env node
/** Internal one-shot Provider adapter; stdout is NEVER the protocol channel.
 * Args: private_request.json private_result.json; subprocess exit code is advisory.
 */
import {readFile, open, link, unlink} from 'node:fs/promises';
import {dirname, join, isAbsolute} from 'node:path';
import {randomBytes} from 'node:crypto';
import {OpenAICompatibleProvider, AnthropicProvider, ProviderRegistry} from '../dist/index.js';
import {parseDaemonInput, makeGenerationRequest, toDaemonProposal} from '../dist/daemon_contract.js';

export async function runBridge(requestPath, resultPath, env=process.env) {
  if (!isAbsolute(requestPath) || !isAbsolute(resultPath) || requestPath === resultPath) throw new Error('invalid bridge paths');
  const b = await readFile(requestPath);
  if (b.length > 64*1024) throw new Error('oversized bridge input');
  const input = parseDaemonInput(JSON.parse(b.toString('utf8')));
  // VODER_PROVIDER_BRIDGE_PLUGIN_V1: only fixed, installed plugin paths are executable.
  // Legacy mode remains for backwards compatibility with the already verified module seven.
  const plugin = env.VODER_MODEL_PLUGIN;
  let provider, model, maxOutputTokens;
  if (plugin === 'cloud') {
    const {createCloudProvider} = await import('../../voder-provider-cloud/dist/index.js');
    ({provider, model, maxOutputTokens} = createCloudProvider(env));
  } else if (plugin === 'open') {
    const {createOpenProvider} = await import('../../voder-provider-open/dist/index.js');
    ({provider, model, maxOutputTokens} = createOpenProvider(env));
  } else if (plugin === undefined || plugin === '') {
    const kind = env.VODER_MODEL_PROVIDER_KIND;
    const endpoint = env.VODER_MODEL_API_ENDPOINT;
    model = env.VODER_MODEL_NAME;
    const apiKey = env.VODER_MODEL_API_KEY;
    maxOutputTokens = Number(env.VODER_MODEL_MAX_OUTPUT_TOKENS ?? '1024');
    const timeoutMs = Number(env.VODER_MODEL_TIMEOUT_MS ?? '60000');
    if (!(kind === 'openai' || kind === 'anthropic') || !endpoint || !model || !apiKey) throw new Error('model provider is not configured');
    if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 1000 || timeoutMs > 180000) throw new Error('unsafe model timeout');
    const registry = new ProviderRegistry();
    const cfg = {id:'voder_default',kind,endpoint,apiKey,timeoutMs,maxResponseBytes:512*1024,allowLoopbackHttp:env.VODER_MODEL_ALLOW_LOOPBACK_HTTP === '1'};
    if (kind === 'openai') registry.register(new OpenAICompatibleProvider(cfg));
    else registry.register(new AnthropicProvider(cfg));
    provider = registry.get('voder_default');
  } else throw new Error('unsupported provider plugin');
  const request = makeGenerationRequest(input, model, maxOutputTokens);
  const output = toDaemonProposal(input,await provider.generate(request));
  const path = join(dirname(resultPath),`.voder-provider-${randomBytes(12).toString('hex')}.tmp`);
  let fd;
  try {
    fd = await open(path,'wx',0o600);
    await fd.writeFile(`${JSON.stringify(output)}\n`,{encoding:'utf8'});
    await fd.sync();
    await fd.close();fd=undefined;
    // Hard-link publish is atomic and exclusive: never overwrite an existing receipt.
    await link(path,resultPath);
    await unlink(path);
  } finally {
    if (fd) await fd.close().catch(()=>{});
    await unlink(path).catch(()=>{});
  }
}
if (process.argv[1] && import.meta.url === new URL(`file://${process.argv[1]}`).href) {
  if (process.argv.length !== 4) process.exitCode=2;
  else {
    runBridge(process.argv[2],process.argv[3]).catch(()=>{console.error('VODER_PROVIDER_BRIDGE_FAILED');process.exitCode=1;});
  }
}
