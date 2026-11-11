/** Per-request host invariant, including Pi's independent compaction summarizer calls. */
import {contextTokens,capacityReserves} from './context-capacity.mjs';
import {createHash} from 'node:crypto';import {VoderError} from './contracts.mjs';
const hash=x=>createHash('sha256').update(x).digest('hex');
const systemText=messages=>messages.filter(m=>m.role==='system').map(m=>[typeof m.content==='string'?m.content:Array.isArray(m.content)?m.content.filter(x=>x.type==='text').map(x=>x.text).join('\n'):'',...Object.values(m.sections??{}).filter(x=>typeof x==='string')].join('\n')).join('\n');
export function guardedStream({raw,normalizeContext,systemPrompt,inspect,allowedModels,currentModel,thinkingFor,onAudit,onFailure,onPressure,maxRequests=64}){
 let count=0;
 const guard=(model,context,options)=>{
  let guard_branch='REQUEST_POLICY',capacity;
  try{
  inspect();if(!allowedModels.some(m=>m.provider===model?.provider&&m.id===model?.id))throw new VoderError('MODEL_NOT_ALLOWED','Request model differs from host registry');
  const selected=currentModel?.();if(selected&&(selected.provider!==model.provider||selected.id!==model.id))throw new VoderError('MODEL_SELECTION_DRIFT','Request differs from the operator-selected model');
  if(++count>maxRequests)throw new VoderError('MODEL_REQUEST_BUDGET','Host request budget exhausted');
  if(!Array.isArray(context?.messages))throw new VoderError('INVALID_MODEL_CONTEXT','Transcript context required');
  let next=context,text=systemText(context.messages);
  if(!text.includes(systemPrompt)){next=normalizeContext({...context,messages:[{role:'system',content:systemPrompt},...context.messages]});text=systemText(next.messages);}
  guard_branch='SYSTEM_RULES_OR_64K';
  if(!text.includes(systemPrompt)||Buffer.byteLength(text)>65536)throw new VoderError('CONTEXT_OVERFLOW','Request rules/context cannot be retained within host bounds; no truncation');
  const targetMax=Number.isSafeInteger(model.maxTokens)?model.maxTokens:undefined;const preferred=thinkingFor?.(model);
  // Base64 is transport bytes, not text tokens. Keep every image in the actual request;
  // use a conservative image allowance in the host estimate instead of counting its encoding as prose.
  let imageCount=0;
  const budgetMessages=next.messages.map(m=>({...m,content:Array.isArray(m.content)?m.content.map(b=>{if(b?.type==='image'){imageCount++;return {type:'image',mimeType:b.mimeType};}return b;}):m.content}));
  const textBudgetBytes=Buffer.byteLength(JSON.stringify({messages:budgetMessages,tools:next.tools??[]}));
  const officialDeepSeek=model.id==='deepseek-flash'&&/^https:\/\/api\.deepseek\.com(?:\/|$)/.test(model.baseUrl||'');
  const imageTokenAllowance=imageCount*(officialDeepSeek?1024:16384);
  if(officialDeepSeek&&imageCount>600)throw new VoderError('MODEL_IMAGE_CAPACITY','DeepSeek permits at most 600 images per request; no images were omitted');
  if(officialDeepSeek&&Buffer.byteLength(JSON.stringify({messages:next.messages,tools:next.tools??[]}))>48*1024*1024)throw new VoderError('MODEL_INPUT_CAPACITY','Complete payload exceeds the documented DeepSeek inline request capacity; extract/select a smaller section; no content was omitted');
  const estimate=contextTokens(model,next,{imageAllowance:imageCount?(imageTokenAllowance/imageCount):0});
  const reserves=capacityReserves(model);
  const payloadTokens=estimate.tokens;
  const contextMax=Number.isSafeInteger(model.contextWindow)?model.contextWindow:undefined;
  const remaining=contextMax?contextMax-payloadTokens-reserves.safety-reserves.finalization:undefined;
  guard_branch='TOKEN_HEADROOM';capacity={context_estimate_tokens:payloadTokens,context_source:estimate.source,context_bytes_upper_bound:textBudgetBytes,usage_tokens:estimate.usage_tokens,context_remaining_tokens:remaining??null};
  if(contextMax&&payloadTokens+reserves.compaction>=contextMax)onPressure?.({code:'CONTEXT_FINALIZATION_RESERVE',context_estimate_tokens:payloadTokens,context_source:estimate.source,context_remaining_tokens:remaining});
  if(remaining!==undefined&&remaining<64)throw new VoderError('CONTEXT_OVERFLOW','Token estimate leaves insufficient reserved response capacity; use controlled compaction; no context was dropped');
  const effectiveMax=targetMax?Math.min(options?.maxTokens??targetMax,targetMax,remaining??Infinity):undefined;
  if(effectiveMax!==undefined&&(!Number.isSafeInteger(effectiveMax)||effectiveMax<1))throw new VoderError('INVALID_MODEL_LIMIT','Invalid output token budget');
  const guardedOptions={...options,...(effectiveMax?{maxTokens:effectiveMax}:{}),...(preferred&&preferred!=='off'?{reasoning:preferred}:{})};
  onAudit?.({sequence:count,provider:model.provider,model:model.id,system_rules_sha256:hash(systemPrompt),system_bytes:Buffer.byteLength(text),rules_exact:true,registered_max_tokens:targetMax??null,effective_max_tokens:guardedOptions.maxTokens??null,context_window:contextMax??null,context_bytes_upper_bound:textBudgetBytes,image_count:imageCount,image_token_allowance:imageTokenAllowance,context_estimate_tokens:payloadTokens,context_source:estimate.source,usage_tokens:estimate.usage_tokens,trailing_estimate_tokens:estimate.trailing_estimate_tokens,context_remaining_tokens:remaining??null,context_reserve_tokens:reserves.safety,finalization_reserve_tokens:reserves.finalization});
  return raw(model,next,guardedOptions);
  }catch(error){if(error instanceof VoderError)onFailure?.(error.code,{guard_branch,...capacity});throw error;}
 };
 guard.beginModelStage=()=>{count=0;};
 return guard;
}
