/** Tokens and transport bytes are different units. No tokenizer result is invented. */
const valid=n=>Number.isSafeInteger(n)&&n>=0;
export function estimateTextTokens(text){
 // Unknown-tokenizer fallback only: ASCII code and non-ASCII use separate estimates.
 // Real provider usage supersedes this estimate for every already-observed prefix.
 let ascii=0,otherBytes=0;for(const c of text){if(c.codePointAt(0)<128)ascii++;else otherBytes+=Buffer.byteLength(c);}
 return Math.ceil(ascii/3.5+otherBytes/2);
}
function usageTokens(usage){
 if(!usage||typeof usage!=='object')return 0;
 const fields=['input','output','cacheRead','cacheWrite'];
 if(fields.some(k=>usage[k]!==undefined&&!valid(usage[k])))return 0;
 const sum=fields.reduce((n,k)=>n+(usage[k]??0),0);
 // reasoning is part of output; cached input is already included in totalTokens.
 return valid(usage.totalTokens)&&usage.totalTokens>0?usage.totalTokens:valid(sum)?sum:0;
}
function messageTokens(m,imageAllowance){
 let tokens=8;
 if(typeof m?.content==='string')tokens+=estimateTextTokens(m.content);
 for(const b of Array.isArray(m?.content)?m.content:[]){
  if(b?.type==='image')tokens+=imageAllowance;
  else if(b?.type==='text')tokens+=estimateTextTokens(b.text??'');
  else if(b?.type==='thinking')tokens+=estimateTextTokens(b.thinking??'');
  else if(b?.type==='toolCall')tokens+=estimateTextTokens(JSON.stringify([b.name,b.arguments]));
 }
 if(m?.role==='system'){
  for(const v of Object.values(m.sections??{}))if(typeof v==='string')tokens+=estimateTextTokens(v);
  for(const key of ['toolsAdded','toolsRemoved'])if(m[key]?.length)tokens+=estimateTextTokens(JSON.stringify(m[key]));
 }
 return tokens;
}
export function contextTokens(model,context,{imageAllowance=16384}={}){
 const messages=context.messages??[];let anchor=-1,used=0,newest=-Infinity;
 for(let i=0;i<messages.length;i++){
  const m=messages[i],stamp=Number.isFinite(m?.timestamp)?m.timestamp:null;
  if(m?.role==='assistant'&&m.provider===model.provider&&m.model===model.id&&stamp!==null&&stamp>=newest&&!['error','aborted'].includes(m.stopReason)){
   const n=usageTokens(m.usage);if(n){anchor=i;used=n;}
  }
  if(stamp!==null)newest=Math.max(newest,stamp);
 }
 const tools=estimateTextTokens(JSON.stringify(context.tools??[]));let tail=tools;
 for(let i=anchor+1;i<messages.length;i++)tail+=messageTokens(messages[i],imageAllowance);
 return {tokens:used+tail,usage_tokens:used,trailing_estimate_tokens:tail,source:anchor>=0?'PROVIDER_USAGE_PLUS_TAIL_ESTIMATE':'UNICODE_HEURISTIC_NO_PROVIDER_USAGE',usage_message_index:anchor};
}
export function capacityReserves(model){
 const window=model.contextWindow;
 if(!valid(window)||window<1)return {safety:1024,finalization:0,output:0,compaction:1024};
 const safety=Math.max(64,Math.min(8192,Math.ceil(window/100)));
 const finalization=Math.min(32768,Math.floor(window/8));
 const output=Math.min(valid(model.maxTokens)?model.maxTokens:0,Math.floor(window/2));
 return {safety,finalization,output,compaction:Math.min(window-64,safety+finalization+output)};
}
export function compactionSettings(inherited,models){
 const current=inherited.compaction??{},overrides={...current.modelOverrides};
 for(const model of models){const key=model.provider+'/'+model.id,own=overrides[key]??{};
  overrides[key]={...own,reserveTokens:Math.max(own.reserveTokens??current.reserveTokens??16384,capacityReserves(model).compaction)};
 }
 return {...inherited,compaction:{...current,modelOverrides:overrides},images:{...inherited.images,autoResize:false}};
}
