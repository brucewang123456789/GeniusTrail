/** Voder model API plugin. Adapts pre-existing voder-provider-cloud to Pi ModelRuntime. */
const API_PRESETS=Object.freeze({
  deepseek:{api:'openai-completions',endpoint:'https://api.deepseek.com/v1'},
  openai:{api:'openai-responses',endpoint:'https://api.openai.com/v1'},
  anthropic:{api:'anthropic-messages',endpoint:'https://api.anthropic.com'},
  gemini:{api:'openai-completions',endpoint:'https://generativelanguage.googleapis.com/v1beta/openai'},
  kimi:{api:'openai-completions',endpoint:'https://api.moonshot.ai/v1'},
  glm:{api:'openai-completions',endpoint:'https://open.bigmodel.cn/api/paas/v4'},
  grok:{api:'openai-completions',endpoint:'https://api.x.ai/v1'},
  qwen:{api:'openai-completions',endpoint:'https://dashscope.aliyuncs.com/compatible-mode/v1'},
  openrouter:{api:'openai-completions',endpoint:'https://openrouter.ai/api/v1'},
});
export const pluginId='voder-provider-cloud';
export const mode='api';
export function inferVendor(model){
 const id=String(model).toLowerCase();
 if(/^deepseek([-.\/]|$)/.test(id))return 'deepseek';
 if(/^(gpt[-\d]|o[134][-\d]|chatgpt)/.test(id))return 'openai';
 if(/^claude[-\/]/.test(id))return 'anthropic';
 if(/^gemini[-\/]/.test(id))return 'gemini';
 if(/^(kimi|moonshot)[-\/]/.test(id))return 'kimi';
 if(/^(glm|chatglm)[-\/]/.test(id))return 'glm';
 if(/^grok[-\/]/.test(id))return 'grok';
 if(/^qwen[-\/]/.test(id))return 'qwen';
 return null;
}
export function resolveConfiguration(env){
 const model=env.VODER_MODEL_ID;
 const key=env.VODER_MODEL_API_KEY;
 if(!model||!key)throw Error('MODEL_ID_AND_API_KEY_REQUIRED');
 if(!/^[\w./:+-]{1,256}$/.test(model))throw Error('INVALID_MODEL_ID');
 if(/\r|\n/.test(key)||key.length>4096)throw Error('INVALID_API_KEY');
 const vendor=env.VODER_MODEL_VENDOR||inferVendor(model);
 if(!vendor)throw Error('MODEL_VENDOR_AMBIGUOUS_SET_VODER_MODEL_VENDOR');
 const preset=API_PRESETS[vendor];
 if(!preset && vendor!=='custom')throw Error('UNSUPPORTED_VENDOR');
 const api=env.VODER_MODEL_PROTOCOL ||preset?.api||'openai-completions';
 if(!['openai-completions','openai-responses','anthropic-messages'].includes(api))throw Error('UNSUPPORTED_PROTOCOL');
 if(vendor==='custom' && !env.VODER_MODEL_ENDPOINT)throw Error('CUSTOM_ENDPOINT_REQUIRED');
 const endpoint=env.VODER_MODEL_ENDPOINT||preset?.endpoint;
 const url=new URL(endpoint);
 if(url.protocol!=='https:'||url.username||url.password||url.search||url.hash)throw Error('INVALID_API_ENDPOINT');
 if(url.pathname.includes('..'))throw Error('INVALID_API_ENDPOINT_PATH');
 return Object.freeze({provider:'voder-model-api', model, api,baseUrl:url.href.replace(/\/$/,''),apiKey:key});
}
