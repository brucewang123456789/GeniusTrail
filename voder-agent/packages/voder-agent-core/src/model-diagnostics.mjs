/** Bounded metadata only. No raw prompts, file content, provider secrets or tool arguments. */
import {createHash} from 'node:crypto';
const sha=s=>createHash('sha256').update(s).digest('hex');
export function modelEventMetadata(event){
 if(!event||typeof event!=='object')return null;
 if(['image_request','vision_selection'].includes(event.type)){const detail={};for(const key of ['provider','model','status','images_sha256'])if(typeof event[key]==='string'&&/^[A-Za-z0-9_./:+-]{1,128}$/.test(event[key]))detail[key]=event[key];if(Number.isSafeInteger(event.image_count)&&event.image_count>=0&&event.image_count<=512)detail.image_count=event.image_count;return detail;}
 if(['request_start','model_policy','output_recovery','model_failure','context_pressure','compaction_start','compaction_end'].includes(event.type)){const detail={};for(const key of ['code','request_sequence','recovery_count','effective_max_tokens','registered_max_tokens','context_window','limit_source','capability_verified','tool_call_count','context_bytes_upper_bound','context_estimate_tokens','context_source','usage_tokens','trailing_estimate_tokens','context_remaining_tokens','context_reserve_tokens','finalization_reserve_tokens','guard_branch','ledger_unchanged','status','reason'])if(['string','number','boolean'].includes(typeof event[key]))detail[key]=event[key];return detail;}
 if(event.type==='message_end'&&event.message?.role==='assistant'){
  const m=event.message;const calls=Array.isArray(m.content)?m.content.filter(b=>b?.type==='toolCall'):[];
  return {tool_call_count:calls.length,tool_calls_digest:sha(JSON.stringify(calls.map(b=>({name:b.name,arguments:b.arguments??null})))),stop_reason:typeof m.stopReason==='string'?m.stopReason.slice(0,64):'UNKNOWN',
   ...(typeof m.errorMessage==='string'?{upstream_error_sha256:sha(m.errorMessage)}:{}),
   output_tokens:Number.isFinite(m.usage?.output)?m.usage.output:undefined,
   ...Object.fromEntries(['input','cacheRead','cacheWrite','totalTokens'].filter(k=>Number.isSafeInteger(m.usage?.[k])&&m.usage[k]>=0).map(k=>['usage_'+k,m.usage[k]])),
   tool_calls:Array.isArray(m.content)?m.content.filter(b=>b?.type==='toolCall').map(b=>({name:String(b.name??'').slice(0,96),arguments_sha256:sha(JSON.stringify(b.arguments??null)),arguments_bytes:Buffer.byteLength(JSON.stringify(b.arguments??null))})).slice(0,32):[]};
 }
 if(!['tool_execution_start','tool_execution_end','agent_start','agent_end','agent_settled'].includes(event.type))return null;
 const texts=(event.result?.content??[]).filter(x=>x?.type==='text'&&typeof x.text==='string').map(x=>x.text);
 const truncated=event.isError===true&&texts.some(s=>s.includes('was not executed: the response hit the output token limit'));
 return {tool:typeof event.toolName==='string'?event.toolName:undefined,
  code:typeof event.result?.details?.error_code==='string'?event.result.details.error_code:truncated?'MODEL_OUTPUT_TRUNCATED':event.isError?'UPSTREAM_TOOL_ERROR':undefined,
  status:typeof event.result?.details?.status==='string'?event.result.details.status:undefined,
  ...(event.isError?{upstream_error_sha256:sha(texts.join('\n'))}:{})};
}

/** Only the dedicated public status envelope is eligible for the UI event channel. */
export function extractPublicProgress(event){
 if(event?.type!=='message_end'||event.message?.role!=='assistant'||!Array.isArray(event.message.content))return null;
 const texts=event.message.content.filter(b=>b?.type==='text'&&typeof b.text==='string').map(b=>b.text);
 for(const text of texts){
  // Exact first-line framing; no searching arbitrary model output or reasoning blocks.
  const match=/^[ \t]*\[VODER_PROGRESS\]([^\r\n]{1,160})\[\/VODER_PROGRESS\](?=\s|$)/u.exec(text);
  if(!match)continue;
  const value=match[1].trim();
  if(value.length<12||value.length>160||Buffer.byteLength(value,'utf8')>480)return null;
  // Guard against accidental disclosure of paths, credentials, identifiers and markup.
  if(/[\u0000-\u001f\u007f<>`\\]/u.test(value)||/(?:^|\s)\/|:\/\/|@|\b(?:key|token|secret|password|credential|authorization|bearer|sk-[A-Za-z0-9]|api[_-]?key)\b/iu.test(value))return null;
  if(/[0-9a-f]{24,}/iu.test(value))return null;
  return value;
 }
 return null;
}
