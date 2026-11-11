/** Private diagnostic observations only. Never grant authority or alter control flow. */
import * as fs from 'node:fs';
import path from 'node:path';
import {randomUUID,createHash} from 'node:crypto';
import {performance} from 'node:perf_hooks';
import {AsyncLocalStorage} from 'node:async_hooks';

const cache=new Map(),sha=s=>createHash('sha256').update(s).digest('hex');
const transportContext=new AsyncLocalStorage();let fetchInstalled=false;
function installFetchObservation(){
 if(fetchInstalled||typeof globalThis.fetch!=='function')return;
 const original=globalThis.fetch;
 globalThis.fetch=function(...args){
  const context=transportContext.getStore();
  if(!context)return Reflect.apply(original,this,args);
  const started=performance.now();context.write('fetch_start',{request_local_id:context.id});
  try{
   const promise=Reflect.apply(original,this,args);
   // Do not consume/clone response bodies or replace the original response/promise.
   promise.then(response=>context.write('fetch_headers',{request_local_id:context.id,http_status:response.status,duration_ms:performance.now()-started}),
    error=>context.error('fetch_error',error,{request_local_id:context.id,duration_ms:performance.now()-started}));
   return promise;
  }catch(error){context.error('fetch_error',error,{request_local_id:context.id,duration_ms:performance.now()-started});throw error;}
 };
 fetchInstalled=true;
}
const LIMIT=8*1024*1024;
const FIELDS=new Set(['sequence','session_id','type','state','status','code','tool','toolName',
 'provider','model','model_id','reason','request_sequence','recovery_count','effective_max_tokens',
 'registered_max_tokens','context_window','output_tokens','usage_input','usage_totalTokens',
 'stop_reason','tool_call_count','evidence_epoch','attempt','maxAttempts','delayMs','success',
 'file_count','approved','error_sha256','error_summary','error_category','http_status',
 'duration_ms','since_request_ms','first_event_ms','first_delta_ms','thinking_chars','text_chars',
 'tool_args_chars','delta_count','message_chars','tool_arguments_bytes','thinking_level',
 'trace_version','monotonic_ms','event_type','request_local_id','limit_source',
 'config_retry_enabled','config_retry_count','config_retry_delay_ms','tool_error_code']);
export function safeError(value){
 let raw=typeof value==='string'?value:String(value?.message??value??'Unknown error');
 let summary=raw;
 // Drop echoed payloads; retain the preceding error explanation only.
 summary=summary.split(/(?:request\s+(?:body|payload)|["']?(?:messages|system_prompt|prompt)["']?\s*[:=])/i)[0];
 for(const [key,val] of Object.entries(process.env))if(/key|token|secret|password|authorization/i.test(key)&&val&&val.length>=6)summary=summary.split(val).join('[REDACTED]');
 summary=summary.replace(/\bBearer\s+[^\s"',;}]+/gi,'Bearer [REDACTED]')
  .replace(/(["']?(?:api[_-]?key|token|secret|password|authorization)["']?\s*[:=]\s*)(?:"[^"\n]*"|'[^'\n]*'|[^\s,;}]+)/gi,'$1[REDACTED]')
  .replace(/\b(?:sk|pk|rk)-[A-Za-z0-9_-]{8,}/g,'[REDACTED]')
  .replace(/https?:\/\/[^\s"'<>]+/gi,'[URL_REDACTED]')
  .replace(/[A-Za-z0-9_+/=-]{100,}/g,'[LONG_VALUE_REDACTED]')
  .replace(/[\r\n\x00-\x1f]+/g,' ').slice(0,1000);
 const match=raw.match(/(?:http(?:\s+status)?|status(?:code)?)[\s:"'=]*(\d{3})\b/i)||raw.match(/^\s*(\d{3})\b/);
 const category=/rate.?limit|too many requests|\b429\b/i.test(raw)?'RATE_LIMIT':
 /\b(?:401|403)\b|unauthori[sz]ed|authentication/i.test(raw)?'AUTH':
 /timeout|timed out|ETIMEDOUT/i.test(raw)?'TIMEOUT':
 /ECONNRESET|ECONNREFUSED|fetch failed|socket|network/i.test(raw)?'NETWORK':
 /context|maximum.*tokens|token.*limit|too long/i.test(raw)?'CAPACITY':
 /invalid|bad request|\b400\b|tool.*call/i.test(raw)?'REQUEST_OR_TOOL_FORMAT':'OTHER';
 return {error_sha256:sha(raw),error_summary:summary,error_category:category,
  ...(match?{http_status:Number(match[1])}:{})};
}
function checkPath(p){
 let cursor=path.parse(p).root;
 for(const part of path.relative(cursor,p).split(path.sep).filter(Boolean)){
  cursor=path.join(cursor,part);const s=fs.lstatSync(cursor);
  if(s.isSymbolicLink()||!s.isDirectory()||(s.mode&0o022))throw Error('TRACE_UNSAFE_PATH');
 }
}
export function createTrace(cwd){
 const root=path.resolve(cwd);if(cache.has(root))return cache.get(root);
 let file=null,disabled=false,request=null,total=0,number=0;
 try{
  // Only Voder's private working-copy location is eligible. No project writes.
  const parent=path.dirname(root);
  if(path.basename(parent)!=='working-copies'||!/^[a-f0-9-]{36}$/i.test(path.basename(root)))throw Error('TRACE_NOT_PRIVATE_COPY');
  checkPath(root);
  const dir=path.join(path.dirname(parent),'private-request-traces');
  try{fs.mkdirSync(dir,{mode:0o700});}catch(e){if(e.code!=='EEXIST')throw e;}
  checkPath(dir);const st=fs.statSync(dir);
  if((st.mode&0o077)||(process.getuid&&st.uid!==process.getuid()))throw Error('TRACE_OWNER_OR_MODE');
  file=path.join(dir,path.basename(root)+'-'+process.pid+'-'+randomUUID()+'.jsonl');
  const fd=fs.openSync(file,fs.constants.O_CREAT|fs.constants.O_EXCL|fs.constants.O_WRONLY|fs.constants.O_NOFOLLOW,0o600);fs.closeSync(fd);
 }catch{disabled=true;}
 const write=(event,fields={})=>{
  if(disabled||!file)return;
  try{
   const row={schema:'voder.private-request-trace.v1',timestamp:new Date().toISOString(),
    monotonic_ms:Math.round(performance.now()*1000)/1000,event,
    ...(request?{request_local_id:request.id}:{})};
   for(const [k,v] of Object.entries(fields))if(FIELDS.has(k)&&['string','number','boolean'].includes(typeof v))row[k]=typeof v==='string'?v.slice(0,k==='error_summary'?1000:160):v;
   const line=JSON.stringify(row)+'\n';total+=Buffer.byteLength(line);
   if(total>LIMIT){disabled=true;return;}
   const fd=fs.openSync(file,fs.constants.O_WRONLY|fs.constants.O_APPEND|fs.constants.O_NOFOLLOW);
   try{const st=fs.fstatSync(fd);if(!st.isFile()||st.nlink!==1||(st.mode&0o077))throw Error('TRACE_FILE_CHANGED');fs.writeSync(fd,line);fs.fdatasyncSync(fd);}finally{fs.closeSync(fd);}
  }catch{disabled=true;}
 };
 const host=e=>{
  try{
   if(e.type==='request_start'){
    request={id:++number,start:performance.now(),first:false,delta:false,thinking:0,text:0,args:0,count:0,last:performance.now()};
    write('request_start',e);
   }else write('host.'+e.type,e);
  }catch{}
 };
 const observe=e=>{
  try{
   const now=performance.now();
   if(e.type==='message_update'&&request){
    const v=e.assistantMessageEvent??{},kind=v.type??'unknown';
    if(!request.first){request.first=true;write('first_model_event',{first_event_ms:now-request.start,event_type:kind});}
    if(/_delta$/.test(kind)){
     if(!request.delta){request.delta=true;write('first_content_delta',{first_delta_ms:now-request.start,event_type:kind});}
     const size=typeof v.delta==='string'?v.delta.length:0;
     if(kind==='thinking_delta')request.thinking+=size;
     if(kind==='text_delta')request.text+=size;
     if(kind==='toolcall_delta')request.args+=size;
     request.count++;
     if(now-request.last>=5000){request.last=now;write('stream_progress',{since_request_ms:now-request.start,thinking_chars:request.thinking,text_chars:request.text,tool_args_chars:request.args,delta_count:request.count});}
    }
    return;
   }
   if(e.type==='message_end'&&e.message?.role==='assistant'){
    const m=e.message,blocks=Array.isArray(m.content)?m.content:[];
    const size=type=>blocks.filter(b=>b?.type===type).reduce((n,b)=>n+String(type==='thinking'?b.thinking??'':b.text??'').length,0);
    const calls=blocks.filter(b=>b?.type==='toolCall');
    write('response_end',{...(request?{duration_ms:now-request.start}:{}),stop_reason:m.stopReason??'unknown',
     output_tokens:m.usage?.output??0,usage_input:m.usage?.input??0,usage_totalTokens:m.usage?.totalTokens??0,
     thinking_chars:size('thinking'),text_chars:size('text'),tool_call_count:calls.length,
     tool_arguments_bytes:calls.reduce((n,b)=>n+Buffer.byteLength(JSON.stringify(b.arguments??null)),0),
     ...(typeof m.errorMessage==='string'?safeError(m.errorMessage):{})});return;
   }
   if(e.type==='tool_execution_end'&&e.isError){
    const text=(e.result?.content??[]).filter(b=>b?.type==='text').map(b=>b.text??'').join('\n');
    write('engine.tool_error',{tool:e.toolName??'unknown',tool_error_code:e.result?.details?.error_code??'UNSPECIFIED',...safeError(text)});return;
   }
   if(['auto_retry_start','auto_retry_end','compaction_start','compaction_end','agent_start','agent_end','agent_settled'].includes(e.type))
    write('engine.'+e.type,{...e,...(e.errorMessage||e.finalError? safeError(e.errorMessage??e.finalError):{})});
  }catch{}
 };
 const error=(event,e,fields={})=>{try{write(event,{...fields,...safeError(e)});}catch{}};
 if(!disabled)try{installFetchObservation();}catch{}
 const result=Object.freeze({write,host,observe,error,
  transport:fn=>transportContext.run({write,error,id:request?.id??0},fn),get file(){return file;}});
 cache.set(root,result);write('trace_open',{trace_version:1});return result;
}
