/** Voder Tool Broker extension seam. Native VGR remains the only write authority.
 * Adapter implementations are trusted host-installed executable code; installing one
 * grants it process permissions, so approval and SHA pinning must happen upstream.
 */
import {createHash} from 'node:crypto';
import {VoderError} from '../../voder-agent-core/src/contracts.mjs';

const APPROVED=new WeakSet();
const NAME=/^voder_ext_[a-z][a-z0-9]*(?:_[a-z0-9]+)*$/;
const digest=x=>createHash('sha256').update(x).digest('hex');
const fail=(code,msg)=>{throw new VoderError(code,msg);};
function deepFreeze(value){if(value&&typeof value==='object'){for(const v of Object.values(value))deepFreeze(v);Object.freeze(value);}return value;}
function shape(schema){
 if(!schema||schema.type!=='object'||!schema.properties||typeof schema.properties!=='object'||Array.isArray(schema.properties)||
    schema.additionalProperties!==false||!Array.isArray(schema.required)||schema.required.some(k=>!Object.hasOwn(schema.properties,k)))
    fail('INVALID_TOOL_SCHEMA','Tool arguments must have an explicit closed object schema');
 const allowed=new Set(['string','integer','number','boolean']);
 if(Object.keys(schema.properties).length>16||Object.values(schema.properties).some(p=>!p||!allowed.has(p.type)||Object.keys(p).some(k=>!['type','minimum','maximum','maxLength','description','enum'].includes(k))))
   fail('INVALID_TOOL_SCHEMA','Only bounded scalar argument fields are supported');
 return deepFreeze(structuredClone(schema));
}
function validateArgs(schema,args){
 if(!args||typeof args!=='object'||Array.isArray(args)||Object.keys(args).some(k=>!Object.hasOwn(schema.properties,k))||
    schema.required.some(k=>!Object.hasOwn(args,k)))fail('INVALID_TOOL_ARGS','Unknown or missing tool arguments');
 for(const [k,v] of Object.entries(args)){
  const p=schema.properties[k];
  if((p.type==='string'&&(typeof v!=='string'||Buffer.byteLength(v,'utf8')>(p.maxLength??(k==='content'?Infinity:1024))))||
     (p.type==='integer'&&(!Number.isSafeInteger(v)||v<(p.minimum??-Infinity)||v>(p.maximum??Infinity)))||
     (p.type==='number'&&(typeof v!=='number'||!Number.isFinite(v)))||
     (p.type==='boolean'&&typeof v!=='boolean')||(p.enum&&!p.enum.includes(v)))fail('INVALID_TOOL_ARGS',`Invalid field: ${k}`);
 }
}
/** Testable capability ranking: descriptive planning hint, not an oracle of global optimality. */
export function rankToolCapabilities(entries,{purpose='inspect',maxRisk=2}={}){
 const needs={inspect:'read',locate:'search',validate:'verify',explain:'read'};
 return entries.filter(x=>x.risk<=maxRisk).map(x=>({name:x.name,
   score:(x.capability===needs[purpose]?40:0)+(x.evidenceValue??0)*10-(x.estimatedTokens??0)/100-(x.estimatedMs??0)/200-x.risk*15,
   estimatedTokens:x.estimatedTokens,estimatedMs:x.estimatedMs,risk:x.risk}))
   .sort((a,b)=>b.score-a.score||a.name.localeCompare(b.name));
}
/** Per-task adaptive read-tool governor: bounded, observable, no duplicate costly errors. */
export class ToolEconomy {
 constructor({maxCalls=24,maxReturnBytes=256*1024,maxFailuresPerTool=2,maxSingleReturnBytes=32768}={}){
  for(const [k,v,lo,hi] of [['maxCalls',maxCalls,1,128],['maxReturnBytes',maxReturnBytes,4096,2**22],
    ['maxFailuresPerTool',maxFailuresPerTool,1,8],['maxSingleReturnBytes',maxSingleReturnBytes,1024,65536]])
   if(!Number.isSafeInteger(v)||v<lo||v>hi)fail('INVALID_TOOL_POLICY',`${k} outside policy`);
  Object.assign(this,{maxCalls,maxReturnBytes,maxFailuresPerTool,maxSingleReturnBytes});
  this.calls=0;this.sharedCalls=0;this.retrievalCalls=0;this.sharedReturnBytes=0;this.returnBytes=0;this.failures=new Map();this.failedInputs=new Set();this.completedInputs=new Set();this.inflight=new Set();this.receipts=[];
 }
 async run(tool,args,execute,signal,{timeoutMs=30000,budgetClass='shared'}={}){
  const retrieval=budgetClass==='retrieval'&&tool==='voder_ext_search_evidence'&&['web','research','fetch','read'].includes(args?.mode);
  if(signal?.aborted)fail('CANCELLED','Task cancelled');
  const failureCount=this.failures.get(tool)??0;
  if(failureCount>=this.maxFailuresPerTool)fail('TOOL_CIRCUIT_OPEN','Read adapter repeatedly failed; select another path');
  if(!retrieval&&this.sharedCalls>=this.maxCalls)fail('TOOL_CALL_BUDGET','Extension tool budget exhausted');
  const signature=digest(JSON.stringify([tool,args]));
  if(this.failedInputs.has(signature)||(this.completedInputs.has(signature)&&!(retrieval&&args.mode==='fetch'))||this.inflight.has(signature))
    fail('DUPLICATE_NO_PROGRESS','Repeating an identical extension tool call provides no new evidence');
  this.calls++;if(retrieval)this.retrievalCalls++;else this.sharedCalls++;this.inflight.add(signature);
  const started=process.hrtime.bigint();
  try{
   // A timeout aborts ONLY read-only adapters; native side effects have separate
   // unknown-outcome semantics and are never automatically replayed.
   const controller=new AbortController();
   const relay=()=>controller.abort();
   signal?.addEventListener('abort',relay,{once:true});
   const timer=setTimeout(()=>controller.abort(),timeoutMs)
   let aborter;
   const aborted=new Promise((_,reject)=>{aborter=()=>reject(new VoderError(signal?.aborted?'CANCELLED':'TOOL_TIMEOUT','Read-only tool timed out or was cancelled'));
      controller.signal.addEventListener('abort',aborter,{once:true});});
   let result;
   try{result=await Promise.race([Promise.resolve().then(()=>execute(controller.signal)),aborted]);}
   finally{clearTimeout(timer);signal?.removeEventListener('abort',relay);controller.signal.removeEventListener('abort',aborter);}
   if(signal?.aborted)fail('CANCELLED','Task cancelled');
   const raw=JSON.stringify(result);
   if(typeof raw!=='string')fail('INVALID_TOOL_RECEIPT','Undefined or unserializable tool receipt');
   const bytes=Buffer.byteLength(raw,'utf8');
   if(bytes>this.maxSingleReturnBytes||(!retrieval&&this.sharedReturnBytes+bytes>this.maxReturnBytes))
     fail('TOOL_RESULT_OVER_BUDGET','Tool output exceeds byte budget; narrow query and retry');
   this.returnBytes+=bytes;if(!retrieval)this.sharedReturnBytes+=bytes;this.failures.set(tool,0);this.completedInputs.add(signature);
   const receipt=Object.freeze({tool,input_sha256:signature,output_sha256:digest(raw),bytes,
     elapsed_ms:Number((process.hrtime.bigint()-started)/1000000n),status:'OBSERVED'});
   this.receipts.push(receipt);
   return {trust:'UNTRUSTED_PLUGIN_DATA',receipt_sha256:receipt.output_sha256,bytes,data:result};
  }catch(e){
   // Policy/precondition errors are actionable refusals, not adapter outages.
   const policyCodes=new Set(['SOURCE_NOT_DISCOVERED','WEB_EVIDENCE_GATE','SEARCH_LOCAL_FIRST',
    'WEB_FORBIDDEN_BY_TASK','WEB_DISABLED','WEB_DOMAIN_DENIED','WEB_QUERY_BUDGET',
    'INVALID_SEARCH_QUERY','INVALID_SEARCH_MODE','UNSAFE_SOURCE_URL','TOOL_RESULT_OVER_BUDGET','CANCELLED']);
   const searchManaged=tool==='voder_ext_search_evidence'&&(/^(?:WEB_|DNS_PUBLIC_ADDRESS_REQUIRED|SEARCH_RESPONSE_TOO_LARGE)/.test(e?.code??''));
   if(searchManaged&&!policyCodes.has(e?.code))this.failedInputs.add(signature);
   if(!policyCodes.has(e?.code)&&!searchManaged){
    this.failedInputs.add(signature);
    // Read current state on settlement: concurrent failures must not overwrite one another.
    this.failures.set(tool,(this.failures.get(tool)??0)+1);
   }
   this.receipts.push(Object.freeze({tool,input_sha256:signature,status:'FAILED',code:String(e?.code??'PLUGIN_FAILURE').slice(0,60),
      elapsed_ms:Number((process.hrtime.bigint()-started)/1000000n)}));
   throw e;
  }finally{this.inflight.delete(signature);
   // Bound telemetry and recent duplicate detection for long retrieval sessions.
   while(this.receipts.length>1024)this.receipts.shift();
   for(const set of [this.completedInputs,this.failedInputs])while(set.size>2048)set.delete(set.values().next().value);
  }
 }
 snapshot(){return Object.freeze({calls:this.calls,shared_calls:this.sharedCalls,retrieval_calls:this.retrievalCalls,return_bytes:this.returnBytes,
   failed_calls:this.receipts.filter(x=>x.status==='FAILED').length,receipts:Object.freeze([...this.receipts])});}
}
/** Read-only extension registrations must be built by the host, never generated from model text. */
export class HostToolRegistry {
 constructor(){this.records=new Map();}
 register({name,description,parameters,execute,capability='read',risk=0,estimatedTokens=200,estimatedMs=100,budgetClass='shared'}={}){
  if(typeof name!=='string'||!NAME.test(name)||name.length>64||this.records.has(name)||this.records.size>=12)
   fail('INVALID_TOOL_REGISTRATION','Invalid, duplicate or excessive extension tool name');
  if(typeof description!=='string'||!description.trim()||description.length>450||typeof execute!=='function'||
    !['read','search','verify'].includes(capability)||!Number.isInteger(risk)||risk<0||risk>2||
    !Number.isSafeInteger(estimatedTokens)||estimatedTokens<0||estimatedTokens>50000||
    !Number.isSafeInteger(estimatedMs)||estimatedMs<0||estimatedMs>120000)
    fail('INVALID_TOOL_REGISTRATION','Read-only host registration contract violated');
  if(!['shared','retrieval'].includes(budgetClass)||(budgetClass==='retrieval'&&(name!=='voder_ext_search_evidence'||capability!=='search')))fail('INVALID_TOOL_REGISTRATION','Retrieval lane is reserved for the host evidence tool');
  const entry=Object.freeze({name,description,parameters:shape(parameters),execute,budgetClass,
     capability,risk,estimatedTokens,estimatedMs});
  this.records.set(name,entry);return this;
 }
 list(){return [...this.records.values()].map(({execute,...entry})=>Object.freeze(entry));}
 makeTools({economy=new ToolEconomy(),stillCurrent=()=>{},event=()=>{},inspectRead}={}){
  return [...this.records.values()].map(entry=>{
   const t={name:entry.name,description:`Read-only host-installed adapter. Estimated ${entry.estimatedTokens} tokens / ${entry.estimatedMs} ms. ${entry.description}`,
     parameters:entry.parameters,execute:async(args,signal)=>{
      stillCurrent();validateArgs(entry.parameters,args);
      const run=()=>economy.run(entry.name,args,combinedSignal=>entry.execute(structuredClone(args),combinedSignal),signal,
         {timeoutMs:Math.min(60000,Math.max(1000,entry.estimatedMs*4)),budgetClass:entry.budgetClass});
      const value=await (inspectRead?inspectRead(entry.name,args,run):run());
      stillCurrent();event({tool:entry.name,output_sha256:value.receipt_sha256,bytes:value.bytes});return value;
     }};
   APPROVED.add(t);return Object.freeze(t);
  });
 }
}
export function isHostApprovedReadTool(tool){return !!tool&&APPROVED.has(tool)&&NAME.test(tool.name);}
