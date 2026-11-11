/** Module-local orchestration; public SDK and task budgets remain unchanged.
 * All default general-web search and page IO uses the Rust binary. No Fetch API.
 */
import {spawn} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {readSearchConfig,normalizeSearchConfig,buildSearchJob,parseSearchResponse,configIdentity} from './search-config.mjs';
import {checkedPublicUrl} from './keyless-web.mjs';
import {siteScope,matchesSiteScope,rankResearchCandidates} from './research.mjs';
const BINARY=fileURLToPath(new URL('../native/bin/voder-retrieval',import.meta.url));
const CODES=new Set(['CANCELLED','INVALID_SEARCH_QUERY','INVALID_SEARCH_PROVIDER','INVALID_SEARCH_POLICY','UNSAFE_SOURCE_URL','WEB_CREDENTIAL_MISSING','WEB_CONFIG_INVALID','WEB_SEARCH_NOT_CONFIGURED','WEB_NATIVE_UNAVAILABLE','WEB_NATIVE_BUSY','WEB_PROVIDER_CIRCUIT_OPEN','WEB_TLS_FAILED','WEB_PROXY_UNSUPPORTED','WEB_AUTH_FAILED','WEB_CONNECT_FAILED','WEB_DNS_FAILED','WEB_SECURE_DNS_UNAVAILABLE','WEB_DNS_INVALID_RESPONSE','WEB_DNSSEC_REQUIRED','DNS_PUBLIC_ADDRESS_REQUIRED','WEB_TIMEOUT','WEB_PROVIDER_FAILED','WEB_PROVIDER_INVALID','WEB_RESPONSE_INCOMPLETE','WEB_SOURCE_NOT_FOUND','WEB_ACCESS_DENIED','WEB_HTTP_ERROR','WEB_RATE_LIMITED','WEB_CONTENT_TYPE_DENIED','WEB_CONTENT_ENCODING_DENIED','WEB_REDIRECT_DENIED','WEB_REDIRECT_LOOP','WEB_REDIRECT_LIMIT','WEB_DOMAIN_DENIED','SEARCH_RESPONSE_TOO_LARGE','WEB_EXTRACTION_FAILED']);
const failure=(code,extra={})=>Object.assign(new Error(CODES.has(code)?code:'WEB_PROVIDER_FAILED'),{code:CODES.has(code)?code:'WEB_PROVIDER_FAILED',...extra});
let active=0;const MAX_ACTIVE=4;
function invokeNative(job,{signal}={}){
 if(signal?.aborted)return Promise.reject(failure('CANCELLED'));
 // Backpressure is explicit; no unbounded children or waiting queue.
 if(active>=MAX_ACTIVE)return Promise.reject(failure('WEB_NATIVE_BUSY'));
 active++;
 return new Promise((resolve,reject)=>{
  let child,done=false,size=0;const chunks=[];
  const finish=(err,value)=>{if(done)return;done=true;active--;clearTimeout(timer);signal?.removeEventListener('abort',abort);child?.kill('SIGKILL');err?reject(err):resolve(value);};
  const abort=()=>finish(failure('CANCELLED'));
  const timer=setTimeout(()=>finish(failure('WEB_TIMEOUT')),job.timeout_ms+100);
  signal?.addEventListener('abort',abort,{once:true});
  try{
   child=spawn(BINARY,[],{stdio:['pipe','pipe','ignore'],windowsHide:true});
   child.once('error',()=>finish(failure('WEB_NATIVE_UNAVAILABLE')));
   child.stdin.on('error',()=>finish(failure('WEB_NATIVE_UNAVAILABLE')));
   child.stdout.on('data',chunk=>{size+=chunk.length;if(size>5*1024*1024)return finish(failure('SEARCH_RESPONSE_TOO_LARGE'));chunks.push(chunk);});
   child.once('close',code=>{
    if(done)return;if(code!==0)return finish(failure('WEB_PROVIDER_FAILED'));
    let response;try{response=JSON.parse(Buffer.concat(chunks).toString('utf8'));}catch{return finish(failure('WEB_PROVIDER_INVALID'));}
    if(response?.ok===true&&response.data&&typeof response.data==='object')return finish(null,response.data);
    const e=response?.error;
    // Native error messages/bodies are never passed through. Trace contains host/IP/status only.
    finish(failure(e?.code,{...(Number.isInteger(e?.http_status)?{httpStatus:e.http_status}:{}),...(e?.network_receipt?{network_receipt:e.network_receipt}:{})}));
   });
   child.stdin.end(JSON.stringify(job));
   if(signal?.aborted)abort();
  }catch{finish(failure('WEB_NATIVE_UNAVAILABLE'));}
 });
}
function dnsKey(job){
 try {const u=new URL(job.url??(job.provider==='bigmodel'?'https://open.bigmodel.cn/':job.provider==='deepseek'?'https://api.deepseek.com/':''));return u.hostname+'|'+u.port+'|'+configIdentity(job.network??{});}catch{return null;}
}
function receiptCache(receipt){const cached=receipt?.dns_cache;if(receipt)delete receipt.dns_cache;return cached;}
function combineReceipt(first,next){if(!first)return next;return {...next,previous_cached_attempts:first,cache_refresh:true};}
export function createNativeRunner(invoke=invokeNative,{now=Date.now}={}){
 const dnsCache=new Map();
 return async function runNative(job,{signal}={}){
 const started=now(),key=dnsKey(job);let cached=key?dnsCache.get(key):undefined;
 if(cached?.expires_ms<=now()){dnsCache.delete(key);cached=undefined;}
 let previous;
 for(let pass=0;pass<2;pass++){
  const remaining=job.timeout_ms-(now()-started);if(remaining<500)throw failure('WEB_TIMEOUT',{network_receipt:previous});
  try{
   const out=await invoke({...job,timeout_ms:remaining,...(cached?{cached_dns:cached}:{})},{signal});
   const candidate=receiptCache(out.network_receipt);
   const verified=out.network_receipt?.attempts?.some(a=>a.result==='HTTP_RESPONSE'&&a.tls_validated===true);
   if(key&&candidate&&verified&&candidate.expires_ms>now()){dnsCache.set(key,candidate);if(dnsCache.size>256)dnsCache.delete(dnsCache.keys().next().value);}
   out.network_receipt=combineReceipt(previous,out.network_receipt);return out;
  }catch(e){
   const candidate=receiptCache(e.network_receipt);
   const verified=e.network_receipt?.attempts?.some(a=>a.result==='HTTP_RESPONSE'&&a.tls_validated===true);
   if(key&&candidate&&verified&&candidate.expires_ms>now()){dnsCache.set(key,candidate);if(dnsCache.size>256)dnsCache.delete(dnsCache.keys().next().value);}
   const safeReconnect=e.network_receipt?.attempts?.length>0&&e.network_receipt.attempts.every(a=>a.retry_before_request===true);
   const retry=pass===0&&cached&&safeReconnect&&['WEB_CONNECT_FAILED','WEB_TLS_FAILED','WEB_TIMEOUT'].includes(e?.code)&&now()-started<job.timeout_ms-500&&!signal?.aborted;
   if(cached&&['WEB_CONNECT_FAILED','WEB_TLS_FAILED','WEB_TIMEOUT'].includes(e?.code)&&!verified&&key)dnsCache.delete(key);
   if(!retry){e.network_receipt=combineReceipt(previous,e.network_receipt);throw e;}
   if(key)dnsCache.delete(key);cached=undefined;previous=e.network_receipt;
  }
 }
}
}
export const runNative=createNativeRunner();
function queryCheck(q){
 if(typeof q!=='string'||!q.trim()||q.length>180||/[\x00-\x1f\x7f]/.test(q)||/(?:api[_-]?key|password|secret)\s*[:=]\s*\S{8,}|\bsk-[A-Za-z0-9_-]{16,}/i.test(q))throw failure('INVALID_SEARCH_QUERY');
 return q.trim();
}
function bounded(action,ms,signal){
 if(signal?.aborted)return Promise.reject(failure('CANCELLED'));
 return new Promise((resolve,reject)=>{
  let done=false;
  const finish=(err,value)=>{if(done)return;done=true;clearTimeout(timer);signal?.removeEventListener('abort',abort);err?reject(err):resolve(value);};
  const abort=()=>finish(failure('CANCELLED'));
  const timer=setTimeout(()=>finish(failure('WEB_TIMEOUT')),Math.max(1,ms));
  signal?.addEventListener('abort',abort,{once:true});
  Promise.resolve().then(action).then(v=>finish(null,v),e=>finish(e));
 });
}
export function networkSummary(r){
 if(!r||typeof r!=='object')return r;
 return {host:r.host,proxy:r.proxy,tls_validation:r.tls_validation,automatic_redirects:r.automatic_redirects,
  dns_addresses:r.dns_addresses?.slice(0,6),attempts:r.attempts?.slice(0,6),
  ...(r.dns?{dns:{mode:r.dns.mode,cache_hit:r.dns.cache_hit,ttl:r.dns.ttl,queries_scheduled:r.dns.queries_scheduled,
   resolvers:r.dns.resolvers?.slice(0,4).map(x=>({resolver:x.resolver,status:x.status,code:x.code}))}}:{}),
  ...(r.cache_refresh?{cache_refresh:true}:{}),details:'BOUNDED_SUMMARY_FULL_DETAILS_IN_PROVIDER_DIAGNOSTICS'};
}
export class RustWebProvider {
 constructor({runImpl=runNative,configImpl=readSearchConfig,timeoutMs=12000,now=Date.now}={}){
  if(typeof runImpl!=='function'||typeof configImpl!=='function'||!Number.isSafeInteger(timeoutMs)||timeoutMs<1000||timeoutMs>30000)throw failure('INVALID_SEARCH_POLICY');
  this.run=runImpl;this.readConfig=configImpl;this.timeoutMs=timeoutMs;this.now=now;this.circuits=new Map();this.lastAttempts=[];
 }
 cacheIdentity(){try{return configIdentity(normalizeSearchConfig(this.readConfig()));}catch{return 'invalid-search-config';}}
 async search(query,{signal,acquireRequest,sourceAllowed=()=>true}={}){
  query=queryCheck(query);const scopes=siteScope(query),cfg=normalizeSearchConfig(this.readConfig()),deadline=this.now()+this.timeoutMs;
  const attempts=[];this.lastAttempts=attempts;let attempted=false,last,leads=[];
  const finish=rows=>Object.defineProperty(rows.slice(0,4),'discovery_attempts',{value:attempts.map(a=>({...a,...(a.network_receipt?{network_receipt:networkSummary(a.network_receipt)}:{})})),enumerable:false});
  const available=cfg.providers.filter(p=>p.ready);
  if(!cfg.providers.length)throw failure('WEB_SEARCH_NOT_CONFIGURED');
  if(!available.length)throw failure('WEB_CREDENTIAL_MISSING');
  for(const entry of available){
   const provider=entry.id;
   if(signal?.aborted)throw failure('CANCELLED');
   if([...query].length>entry.max_query_chars){attempts.push({provider,status:'SKIPPED_QUERY_LIMIT'});last=failure('INVALID_SEARCH_QUERY');continue;}
   // Credential fingerprint scopes cooldowns; rotating keys immediately permits recovery.
   const id=provider+':'+configIdentity(entry);
   const circuit=this.circuits.get(id);
   if(circuit&&(circuit.probing||circuit.until>this.now())){attempts.push({provider,status:'SKIPPED_CIRCUIT_OPEN'});continue;}
   const remaining=deadline-this.now();if(remaining<500){last=failure('WEB_TIMEOUT');break;}
   if(attempted){
    if(typeof acquireRequest!=='function'){attempts.push({provider,status:'SKIPPED_UNMETERED_FALLBACK'});break;}
    try{if(acquireRequest()===false){attempts.push({provider,status:'SKIPPED_BUDGET'});break;}}catch(e){if(e?.code!=='WEB_QUERY_BUDGET')throw e;attempts.push({provider,status:'SKIPPED_BUDGET'});break;}
   }
   attempted=true;
   if(circuit)circuit.probing=true;
   try{
    const hasBackup=available.slice(available.indexOf(entry)+1).some(p=>[...query].length<=p.max_query_chars);
    const attemptTimeout=hasBackup?Math.max(500,Math.floor(remaining/2)):remaining;
    const buildStarted=this.now();
    const job=await bounded(()=>buildSearchJob(entry,query,attemptTimeout),attemptTimeout,signal);
    job.timeout_ms=attemptTimeout-(this.now()-buildStarted);if(job.timeout_ms<500)throw failure('WEB_TIMEOUT');
    if(signal?.aborted)throw failure('CANCELLED');
    const out=await this.run(job,{signal});
    const rows=await bounded(()=>parseSearchResponse(entry,out),deadline-this.now(),signal);
    this.circuits.delete(id);
    const eligible=rows.slice(0,16).filter(r=>{try{return checkedPublicUrl(r.url)&&sourceAllowed(r.url)&&matchesSiteScope(r.url,scopes)&&typeof r.title==='string'&&typeof r.excerpt==='string';}catch{return false;}});
    const ranked=rankResearchCandidates(query,eligible,{maxResults:8});
    attempts.push({provider,status:ranked.length?'CANDIDATES_FOUND':eligible.length?'WEAK_RESULTS':'NO_SCOPED_RESULTS',returned:rows.length,eligible:eligible.length,...(out.network_receipt?{network_receipt:out.network_receipt}:{})});
    leads.push(...eligible);if(ranked.length)return finish(ranked);
   }catch(e){
    if(e?.code==='CANCELLED'||signal?.aborted){if(circuit)circuit.probing=false;throw failure('CANCELLED');}
    const code=CODES.has(e?.code)?e.code:'WEB_PROVIDER_FAILED';last=failure(code,{httpStatus:e.httpStatus});
    attempts.push({provider,status:'FAILED',code,...(e.network_receipt?{network_receipt:e.network_receipt}:{})});
    if(['WEB_TIMEOUT','WEB_TLS_FAILED','WEB_CONNECT_FAILED','WEB_DNS_FAILED','WEB_SECURE_DNS_UNAVAILABLE','WEB_HTTP_ERROR','WEB_RATE_LIMITED','WEB_AUTH_FAILED','WEB_ACCESS_DENIED','WEB_PROVIDER_INVALID'].includes(code)){
     this.circuits.set(id,{until:this.now()+(['WEB_AUTH_FAILED','WEB_ACCESS_DENIED'].includes(code)?60000:30000),probing:false});
     if(this.circuits.size>8)this.circuits.delete(this.circuits.keys().next().value);
    }else if(circuit)circuit.probing=false;
    if(['WEB_NATIVE_UNAVAILABLE','WEB_CONFIG_INVALID','WEB_PROXY_UNSUPPORTED','INVALID_SEARCH_POLICY'].includes(code))break;
   }
  }
  // A valid empty result is distinct from every backend failing. Failed backups cannot erase valid leads.
  if(attempts.some(a=>['NO_SCOPED_RESULTS','WEAK_RESULTS'].includes(a.status))){const seen=new Set();return finish(leads.filter(r=>{if(seen.has(r.url))return false;seen.add(r.url);return true;}));}
  throw Object.assign(last??failure('WEB_PROVIDER_CIRCUIT_OPEN'),{discovery_attempts:attempts});
 }
 async extract(url,{focus='',signal,urlAllowed=()=>true}={}){
  const original=checkedPublicUrl(url);if(typeof focus!=='string'||focus.length>140||/[\x00-\x1f\x7f]/.test(focus))throw failure('INVALID_SEARCH_QUERY');
  const deadline=this.now()+this.timeoutMs,visited=new Set(),redirects=[],receipts=[];this.lastFetchReceipts=receipts;let current=original;
  while(true){
   if(signal?.aborted)throw failure('CANCELLED');
   current=checkedPublicUrl(current);if(!urlAllowed(current))throw failure('WEB_DOMAIN_DENIED');
   if(visited.has(current))throw failure('WEB_REDIRECT_LOOP');visited.add(current);
   const remaining=deadline-this.now();if(remaining<500)throw failure('WEB_TIMEOUT');
   const out=await this.run({op:'fetch',url:current,network:normalizeSearchConfig(this.readConfig()).network,timeout_ms:remaining},{signal});
   if(out.network_receipt)receipts.push(out.network_receipt);
   if(out.redirect!==undefined){
    if(redirects.length>=4)throw failure('WEB_REDIRECT_LIMIT');
    const next=checkedPublicUrl(out.redirect);if(!urlAllowed(next))throw failure('WEB_DOMAIN_DENIED');
    redirects.push({from:current,to:next,status:out.status});current=next;continue;
   }
   if(typeof out.text!=='string'||!out.text.trim()||out.url!==current||!/^[a-f0-9]{64}$/.test(out.raw_sha256))throw failure('WEB_EXTRACTION_FAILED');
   return {url:original,final_url:current,redirects,raw_sha256:out.raw_sha256,text:out.text,network_receipts:receipts.map(networkSummary),evidence_kind:'FETCHED_PAGE'};
  }
 }
}
