/** Host-owned declarative search configuration. Never evaluated as code. */
import {normalizeBudget} from './request-budget.mjs';
import {normalizeNetwork} from './network-config.mjs';
import {closeSync,constants,fstatSync,openSync,readFileSync,realpathSync} from 'node:fs';
import {homedir} from 'node:os';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {resolve,sep} from 'node:path';
import {createHash} from 'node:crypto';
import {checkedPublicUrl} from './keyless-web.mjs';
const ROOT=fileURLToPath(new URL('../',import.meta.url));
const CONFIG=new URL('../search-config.json',import.meta.url);
function readSearchBytes(){
 const profile=resolve(homedir(),'.config','voder','search-config.json');
 let fd;
 try{fd=openSync(profile,constants.O_RDONLY|constants.O_NOFOLLOW|constants.O_NONBLOCK);}
 catch(e){if(e.code==='ENOENT')return readFileSync(CONFIG);throw e;}
 try{
  const st=fstatSync(fd);
  if(!st.isFile()||(st.mode&0o077)!==0||(process.getuid&&st.uid!==process.getuid())||st.size>128*1024)fail('WEB_CONFIG_INVALID');
  return readFileSync(fd);
 }finally{closeSync(fd);}
}
const fail=code=>{throw Object.assign(new Error(code),{code})};
const plain=x=>!!x&&typeof x==='object'&&!Array.isArray(x)&&[Object.prototype,null].includes(Object.getPrototypeOf(x));
const clone=x=>JSON.parse(JSON.stringify(x));
const forbidden=new Set(['__proto__','constructor','prototype']);
const PRESETS={
 brave:{endpoint:'https://api.search.brave.com/res/v1/web/search',method:'GET',headers:{'X-Subscription-Token':'{{secret:api_key}}'},params:{q:'{{query}}',count:8},response:{items:'/web/results',url:'/url',title:'/title',excerpt:'/description'}},
 tavily:{endpoint:'https://api.tavily.com/search',method:'POST',headers:{Authorization:'Bearer {{secret:api_key}}'},body:{query:'{{query}}',max_results:8,search_depth:'basic',include_answer:false,include_raw_content:false},response:{items:'/results',url:'/url',title:'/title',excerpt:'/content'}},
 serper:{endpoint:'https://google.serper.dev/search',method:'POST',headers:{'X-API-KEY':'{{secret:api_key}}'},body:{q:'{{query}}',num:8},response:{items:'/organic',url:'/link',title:'/title',excerpt:'/snippet'}},
 serpapi:{endpoint:'https://serpapi.com/search.json',method:'GET',params:{q:'{{query}}',engine:'google',num:8,api_key:'{{secret:api_key}}'},response:{items:'/organic_results',url:'/link',title:'/title',excerpt:'/snippet'}},
 bigmodel:{endpoint:'https://open.bigmodel.cn/api/paas/v4/web_search',method:'POST',headers:{Authorization:'Bearer {{secret:api_key}}'},body:{search_query:'{{query}}',search_engine:'search_pro',count:8,search_intent:false,search_recency_filter:'noLimit'},response:{items:'/search_result',url:'/link',title:'/title',excerpt:'/content'}},
 deepseek:{endpoint:'https://api.deepseek.com/v1/tools/search',method:'POST',headers:{Authorization:'Bearer {{secret:api_key}}'},body:{query:'{{query}}',count:8,need_content:false},max_query_chars:128,response:{items:'/results',url:'/url',title:'/title',excerpt:['/summary','/snippet']}},
 searxng:{method:'GET',params:{q:'{{query}}',format:'json'},response:{items:'/results',url:'/url',title:'/title',excerpt:'/content'}}
};
export const SEARCH_PRESETS=Object.freeze(Object.keys(PRESETS));
const ALLOWED=new Set(['id','preset','endpoint','method','headers','params','body','body_format','response_format','allow_loopback','response','credentials','max_query_chars','adapter','network']);
function pointer(path){
 if(typeof path!=='string'||path.length>512||!['', '/'].some(p=>p===''?path==='':path.startsWith(p)))fail('WEB_CONFIG_INVALID');
 const parts=path===''?[]:path.slice(1).split('/').map(s=>s.replace(/~1/g,'/').replace(/~0/g,'~'));
 if(parts.length>16||parts.some(s=>forbidden.has(s))||/~(?![01])/u.test(path))fail('WEB_CONFIG_INVALID');
 return parts;
}
function templates(x,depth=0){
 if(depth>12)fail('WEB_CONFIG_INVALID');
 if(typeof x==='string'){
  if(x.length>8192)fail('WEB_CONFIG_INVALID');
  const refs=[...x.matchAll(/\{\{([^{}]*)\}\}/g)].map(m=>m[1]);
  if(refs.some(v=>v!=='query'&&!/^secret:[a-z][a-z0-9_]{0,47}$/.test(v)))fail('WEB_CONFIG_INVALID');
  if(x.replace(/\{\{[^{}]*\}\}/g,'').includes('{{'))fail('WEB_CONFIG_INVALID');
  return refs;
 }
 if(Array.isArray(x))return x.flatMap(v=>templates(v,depth+1));
 if(plain(x)){if(Object.keys(x).length>64||Object.keys(x).some(k=>forbidden.has(k)||k.includes('{{')))fail('WEB_CONFIG_INVALID');return Object.values(x).flatMap(v=>templates(v,depth+1));}
 if(x===null||typeof x==='boolean'||(typeof x==='number'&&Number.isFinite(x)))return [];
 fail('WEB_CONFIG_INVALID');
}
function fields(obj,allowed){if(!plain(obj)||Object.keys(obj).some(k=>!allowed.includes(k)))fail('WEB_CONFIG_INVALID');}
export function readSearchConfig(){
 try{const b=readSearchBytes();if(b.length>128*1024)fail('WEB_CONFIG_INVALID');return JSON.parse(b.toString('utf8'));}
 catch(e){if(e?.code==='ENOENT')return {version:1,providers:[]};if(e?.code==='WEB_CONFIG_INVALID')throw e;fail('WEB_CONFIG_INVALID');}
}
export function normalizeSearchConfig(raw,env=process.env){
 if(!plain(raw)||JSON.stringify(raw).length>128*1024)fail('WEB_CONFIG_INVALID');
 // Existing revision-12 files and explicit host test/provider injection remain compatible.
 if(raw.version===undefined&&Array.isArray(raw.providers)&&raw.providers.every(x=>typeof x==='string')){
  fields(raw,['providers','keys','network','budget']);if(raw.budget!==undefined)normalizeBudget(raw.budget);const network=normalizeNetwork(raw.network);const keys=raw.keys??{};
  if(!plain(keys)||Object.keys(keys).some(k=>!['bigmodel','deepseek'].includes(k))||raw.providers.length>2||new Set(raw.providers).size!==raw.providers.length)fail('WEB_CONFIG_INVALID');
  return {network,providers:raw.providers.map(id=>{
   if(!['bigmodel','deepseek'].includes(id))fail('WEB_CONFIG_INVALID');
   const key=env[`VODER_SEARCH_${id.toUpperCase()}_API_KEY`]??keys[id]??'';
   if(typeof key!=='string'||key.length>4096||/[\x00-\x1f\x7f]/.test(key))fail('WEB_CONFIG_INVALID');
   return {id,network,legacy:true,key,ready:!!key.trim(),max_query_chars:id==='deepseek'?128:180};
  })};
 }
 fields(raw,['version','providers','network','budget']);if(raw.budget!==undefined)normalizeBudget(raw.budget);const network=normalizeNetwork(raw.network);if(raw.version!==1||!Array.isArray(raw.providers)||raw.providers.length>8)fail('WEB_CONFIG_INVALID');
 const seen=new Set();
 return {network,providers:raw.providers.map(input=>{
  if(!plain(input)||Object.keys(input).some(k=>!ALLOWED.has(k)))fail('WEB_CONFIG_INVALID');
  const preset=input.preset;
  if(preset!==undefined&&!Object.hasOwn(PRESETS,preset))fail('WEB_CONFIG_INVALID');
  const p={...(preset?clone(PRESETS[preset]):{}),...clone(input)};p.network=normalizeNetwork(p.network,network);
  if(typeof p.id!=='string'||!/^[-a-zA-Z0-9_]{1,64}$/.test(p.id)||seen.has(p.id))fail('WEB_CONFIG_INVALID');seen.add(p.id);
  if(typeof p.endpoint!=='string'||p.endpoint.includes('{{'))fail('WEB_CONFIG_INVALID');
  if(p.allow_loopback!==undefined&&typeof p.allow_loopback!=='boolean')fail('WEB_CONFIG_INVALID');
  let loopback=false;try{const u=new URL(p.endpoint);loopback=p.allow_loopback===true&&u.protocol==='http:'&&['127.0.0.1','[::1]'].includes(u.hostname)&&!u.username&&!u.password&&!u.hash&&p.endpoint.length<=1300&&!/[\x00-\x20\x7f]/.test(p.endpoint);if(loopback)p.endpoint=u.href;else {if(p.endpoint.length>1300||/[\x00-\x20\x7f]/.test(p.endpoint))fail('WEB_CONFIG_INVALID');const port=u.port;u.port='';p.endpoint=checkedPublicUrl(u.href);const approved=new URL(p.endpoint);approved.port=port;p.endpoint=approved.href;}}catch{fail('WEB_CONFIG_INVALID');}
  p.allow_loopback=loopback;
  p.method=p.method??'GET';p.body_format=p.body_format??'json';p.response_format=p.response_format??'json';
  if(!['GET','POST'].includes(p.method)||!['json','form','text'].includes(p.body_format)||!['json','text'].includes(p.response_format)||(p.response_format==='text'&&!p.adapter)||(p.body_format==='text'&&typeof p.body!=='string')||(p.method==='GET'&&p.body!==undefined))fail('WEB_CONFIG_INVALID');
  p.headers=p.headers??{};p.params=p.params??{};
  if(!plain(p.headers)||Object.keys(p.headers).length>24||Object.entries(p.headers).some(([k,v])=>!/^[-!#$%&'*+.^_`|~a-zA-Z0-9]{1,80}$/.test(k)||['host','content-length','connection','transfer-encoding','proxy-authorization','proxy-connection','accept-encoding','user-agent'].includes(k.toLowerCase())||typeof v!=='string'||/[\r\n\x00]/.test(v)||v.includes('{{query}}')))fail('WEB_CONFIG_INVALID');
  if(!plain(p.params)||Object.keys(p.params).length>64||Object.values(p.params).some(v=>!['string','number','boolean'].includes(typeof v)))fail('WEB_CONFIG_INVALID');
  const refs=templates({headers:p.headers,params:p.params,body:p.body??null});
  if(!refs.includes('query')&&!p.adapter)fail('WEB_CONFIG_INVALID');
  p.max_query_chars=p.max_query_chars??180;if(!Number.isSafeInteger(p.max_query_chars)||p.max_query_chars<1||p.max_query_chars>180)fail('WEB_CONFIG_INVALID');
  p.credentials=p.credentials??(preset&&preset!=='searxng'?{api_key:{env:`VODER_SEARCH_${preset.toUpperCase()}_API_KEY`}}:{});
  if(!plain(p.credentials)||Object.keys(p.credentials).length>12)fail('WEB_CONFIG_INVALID');
  p.secrets={};p.ready=true;
  for(const [name,source] of Object.entries(p.credentials)){
   if(!/^[a-z][a-z0-9_]{0,47}$/.test(name))fail('WEB_CONFIG_INVALID');
   fields(source,['env','value']);
   if((source.env===undefined)===(source.value===undefined))fail('WEB_CONFIG_INVALID');
   if(source.env!==undefined&&(typeof source.env!=='string'||!/^[_a-zA-Z][_a-zA-Z0-9]{0,127}$/.test(source.env)))fail('WEB_CONFIG_INVALID');
   const value=source.env!==undefined?env[source.env]:source.value;
   if(value!==undefined&&(typeof value!=='string'||value.length>4096||/[\x00-\x1f\x7f]/.test(value)))fail('WEB_CONFIG_INVALID');
   p.secrets[name]=value??'';
  }
  for(const r of refs.filter(x=>x.startsWith('secret:'))){const name=r.slice(7);if(!Object.hasOwn(p.secrets,name))fail('WEB_CONFIG_INVALID');if(!p.secrets[name].trim())p.ready=false;}
  if(p.adapter&&Object.values(p.secrets).some(v=>!v.trim()))p.ready=false;
  if(p.adapter!==undefined&&(!/^adapters\/[-a-zA-Z0-9_]+\.mjs$/.test(p.adapter)))fail('WEB_CONFIG_INVALID');
  if(p.response!==undefined){
   fields(p.response,['items','url','title','excerpt']);pointer(p.response.items);pointer(p.response.url);
   for(const key of ['title','excerpt'])if(p.response[key]!==undefined){const paths=Array.isArray(p.response[key])?p.response[key]:[p.response[key]];if(!paths.length||paths.length>4)fail('WEB_CONFIG_INVALID');paths.forEach(pointer);}
  }else if(!p.adapter)fail('WEB_CONFIG_INVALID');
  return p;
 })};
}
function render(value,query,secrets){
 if(typeof value==='string')return value.replace(/\{\{([^{}]*)\}\}/g,(_,v)=>v==='query'?query:secrets[v.slice(7)]);
 if(Array.isArray(value))return value.map(v=>render(v,query,secrets));
 if(plain(value))return Object.fromEntries(Object.entries(value).map(([k,v])=>[k,render(v,query,secrets)]));
 return value;
}
export function checkAdapterPath(p){
 if(!p.adapter)return;
 const path=resolve(ROOT,p.adapter);let actual;
 try{actual=realpathSync(path);}catch{fail('WEB_CONFIG_INVALID');}
 if(actual!==path||!actual.startsWith(resolve(ROOT,'adapters')+sep))fail('WEB_CONFIG_INVALID');
}
const adapterCache=new Map();
async function adapterFor(p){
 if(!p.adapter)return null;
 const path=resolve(ROOT,p.adapter);let actual,bytes;
 try{actual=realpathSync(path);if(!actual.startsWith(resolve(ROOT,'adapters')+sep)||actual!==path)fail('WEB_CONFIG_INVALID');bytes=readFileSync(actual);}catch{fail('WEB_CONFIG_INVALID');}
 const id=actual+':'+createHash('sha256').update(bytes).digest('hex');
 if(!adapterCache.has(id)){
  const mod=await import(pathToFileURL(actual).href+'?v='+id.split(':').at(-1));
  if(typeof mod.buildRequest!=='function'||typeof mod.parseResponse!=='function')fail('WEB_CONFIG_INVALID');
  adapterCache.set(id,mod);if(adapterCache.size>8)adapterCache.delete(adapterCache.keys().next().value);
 }
 return adapterCache.get(id);
}
export function configIdentity(config){return createHash('sha256').update(JSON.stringify(config)).digest('hex');}
export async function buildSearchJob(p,query,timeoutMs){
 if(p.legacy)return {op:'search',provider:p.id,query,key:p.key,network:p.network,timeout_ms:timeoutMs};
 let request={url:p.endpoint,method:p.method,headers:render(p.headers,query,p.secrets),params:render(p.params,query,p.secrets),body:p.body===undefined?null:render(p.body,query,p.secrets),body_format:p.body_format,response_format:p.response_format,allow_loopback:p.allow_loopback};
 const adapter=await adapterFor(p);
 if(adapter){
  const out=await adapter.buildRequest({query,request:clone(request),credentials:{...p.secrets}});
  if(!plain(out)||Object.keys(out).some(k=>!['url','method','headers','params','body','body_format','response_format','allow_loopback'].includes(k)))fail('WEB_CONFIG_INVALID');
  request={...request,...out};
  // A signer cannot silently move credentials to a different API origin or path.
  if(request.url!==p.endpoint||request.allow_loopback!==p.allow_loopback)fail('WEB_CONFIG_INVALID');
 }
 if(JSON.stringify(request).length>65536)fail('WEB_CONFIG_INVALID');
 return {op:'search_custom',...request,network:p.network,timeout_ms:timeoutMs};
}
function at(value,path){for(const key of pointer(path)){if(value===null||typeof value!=='object'||!Object.hasOwn(value,key))return undefined;value=value[key];}return value;}
function text(row,path){
 if(path===undefined)return '';
 for(const p of Array.isArray(path)?path:[path]){const v=at(row,p);if(v===null||v===undefined)continue;if(typeof v!=='string')fail('WEB_PROVIDER_INVALID');if(v)return v;}
 return '';
}
export async function parseSearchResponse(p,out){
 if(p.legacy){if(!Array.isArray(out?.rows))fail('WEB_PROVIDER_INVALID');return out.rows;}
 let rows;
 const adapter=await adapterFor(p);
 if(adapter)rows=await adapter.parseResponse(out?.payload);
 else{
  const items=at(out?.payload,p.response.items);
  if(!Array.isArray(items))fail('WEB_PROVIDER_INVALID');
  if(items.length>1000)fail('WEB_PROVIDER_INVALID');
  rows=items.slice(0,32).map(row=>{if((!plain(row)&&typeof row!=='string')||typeof at(row,p.response.url)!=='string')fail('WEB_PROVIDER_INVALID');return {url:text(row,p.response.url),title:text(row,p.response.title),excerpt:text(row,p.response.excerpt)};});
 }
 if(!Array.isArray(rows)||rows.length>1000)fail('WEB_PROVIDER_INVALID');
 return rows.slice(0,32).map(r=>{if(!plain(r)||typeof r.url!=='string'||typeof r.title!=='string'||typeof r.excerpt!=='string')fail('WEB_PROVIDER_INVALID');return {url:r.url,title:r.title.slice(0,160),excerpt:r.excerpt.slice(0,1200),evidence_kind:'SEARCH_SNIPPET'};});
}
