/** Model limits are provider capabilities, never VGR authority. No generation calls. */
const integer=v=>Number.isSafeInteger(v)&&v>=64;
const pair=m=>{const contextWindow=m?.context_window??m?.contextWindow;const maxTokens=m?.max_output_tokens??m?.maxTokens??m?.top_provider?.max_completion_tokens;
 return integer(contextWindow)&&integer(maxTokens)&&maxTokens<=contextWindow?{contextWindow,maxTokens}:null;};
function override(env,key){if(env[key]===undefined)return null;const n=Number(env[key]);if(!integer(n))throw Error('INVALID_MODEL_LIMIT_'+key);return n;}
export async function resolveModelLimits({configuration,env=process.env,runtime,fetcher=globalThis.fetch}){
 const p=configuration,contextOverride=override(env,'VODER_MODEL_CONTEXT_WINDOW'),outputOverride=override(env,'VODER_MODEL_MAX_TOKENS');
 let found=null,source='PI_COMPATIBILITY_DEFAULT_UNVERIFIED',discovery='NOT_ATTEMPTED';
 // Read only the operator-configured endpoint, never links returned by a server. No redirect with credentials.
 if((contextOverride===null||outputOverride===null)&&typeof fetcher==='function'){
  try{
   const endpoint=new URL(p.baseUrl.replace(/\/$/,'')+'/models');
   if(!(endpoint.protocol==='https:'||endpoint.protocol==='http:'&&['127.0.0.1','[::1]','localhost'].includes(endpoint.hostname))||endpoint.username||endpoint.password||endpoint.search||endpoint.hash)throw Error('UNSAFE_ENDPOINT');
   const headers=p.api==='anthropic-messages'?{'x-api-key':p.apiKey,'anthropic-version':'2023-06-01'}:{authorization:'Bearer '+p.apiKey};
   const response=await fetcher(endpoint.href,{headers,redirect:'error',signal:AbortSignal.timeout(8000)});
   if(!response.ok)throw Error('HTTP_STATUS');
   const reader=response.body.getReader();let size=0;const chunks=[];
   try{while(true){const item=await reader.read();if(item.done)break;size+=item.value.length;if(size>2*1024*1024)throw Error('METADATA_TOO_LARGE');chunks.push(Buffer.from(item.value));}}
   finally{await reader.cancel().catch(()=>{});}
   const data=JSON.parse(Buffer.concat(chunks).toString('utf8'));
   const rows=Array.isArray(data.data)?data.data:Array.isArray(data.models)?data.models:[];
   const matches=rows.filter(x=>x?.id===p.model);if(matches.length===1)found=pair(matches[0]);
   discovery=found?'MATCHED_MODEL_LIMITS':'LIMITS_NOT_ADVERTISED';if(found)source='CONFIGURED_ENDPOINT_MODELS';
  }catch{discovery='UNAVAILABLE_NO_GENERATION_RETRY';}
 }
 if(!found){
  // Do not transfer official-model claims to an unrelated proxy with the same model ID.
  const endpointIdentity=value=>{const u=new URL(value);if(u.username||u.password||u.search||u.hash)throw Error('UNSAFE_ENDPOINT');return u.origin+u.pathname.replace(/\/+$/,'');};
  const endpoint=endpointIdentity(p.baseUrl);
  const rows=typeof runtime?.getModels==='function'?runtime.getModels():[];
  const matches=rows.filter(m=>{try{return m.id===p.model&&m.api===p.api&&endpointIdentity(m.baseUrl)===endpoint&&pair(m);}catch{return false;}});
  if(matches.length){const candidates=matches.map(pair);if(candidates.every(x=>x.contextWindow===candidates[0].contextWindow&&x.maxTokens===candidates[0].maxTokens)){found=candidates[0];source='INSTALLED_MODEL_CATALOG';}}
 }
 // This is Pi's compatibility default, explicitly not a claim about an unknown provider's true limits.
 const defaults=found??{contextWindow:128000,maxTokens:16384};
 const contextWindow=contextOverride??defaults.contextWindow,maxTokens=outputOverride??Math.min(defaults.maxTokens,contextWindow);
 if(contextWindow<1024||maxTokens>contextWindow)throw Error('MAX_TOKENS_EXCEED_CONTEXT');
 if(found&&((contextOverride!==null&&contextOverride>found.contextWindow)||(outputOverride!==null&&outputOverride>found.maxTokens)))throw Error('EXPLICIT_MODEL_LIMIT_EXCEEDS_ADVERTISED_CAPABILITY');
 return Object.freeze({contextWindow,maxTokens,source:(contextOverride!==null||outputOverride!==null)?'EXPLICIT_OPERATOR_OVERRIDE_WITH_'+source:source,discovery,capability_verified:found!==null,explicit_output_limit:outputOverride!==null});
}
