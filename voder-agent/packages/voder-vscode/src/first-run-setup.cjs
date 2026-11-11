 'use strict';
// VODER_GLM_REQUIRED_SETTINGS_V1: keep setup output compatible with model-catalog.
function applyRequiredModelSettings(item){
 if(String(item.id).toLowerCase()!=='glm-5.3-flash')return item;
 item.reasoning=true;
 if(!['low','high','max'].includes(item.thinkingLevel))item.thinkingLevel='low';
 item.compat={...(item.compat||{}),thinkingFormat:'zai',supportsReasoningEffort:true};
 return item;
}
// First-run operator configuration. This module does not grant VGR write authority.
const fs=require('node:fs/promises');
const path=require('node:path');
const os=require('node:os');
const crypto=require('node:crypto');
const HOME=path.join(os.homedir(),'.config','voder');
const APIs=['openai-completions','openai-responses','anthropic-messages','google-generative-ai'];
const SEARCH=['brave','tavily','serper','serpapi','bigmodel','deepseek','searxng'];
const SEARCH_OPTIONS=[...SEARCH,'custom'];
function assertSecret(s){if(typeof s!=='string'||s.length<8||s.length>4096||/[\r\n\0]/.test(s))throw Error('INVALID_SECRET');return s;}
function validEndpoint(s){const u=new URL(s);if(!((u.protocol==='https:')||(u.protocol==='http:'&&['localhost','127.0.0.1','[::1]'].includes(u.hostname)))||u.username||u.password||u.search||u.hash)throw Error('INVALID_MODEL_ENDPOINT');return u.href.replace(/\/$/,'');}
async function privateDir(){await fs.mkdir(HOME,{recursive:true,mode:0o700});let s=await fs.lstat(HOME);if(!s.isDirectory()||s.isSymbolicLink()||(s.mode&0o077)|| (process.getuid&&s.uid!==process.getuid()))throw Error('UNSAFE_CONFIG_DIRECTORY');}
async function readPrivate(name,defaultValue){await privateDir();const target=path.join(HOME,name);try{let s=await fs.lstat(target);if(!s.isFile()||s.isSymbolicLink()||(s.mode&0o077)||(process.getuid&&s.uid!==process.getuid())||s.size>65536)throw Error('UNSAFE_CONFIG_FILE');return JSON.parse(await fs.readFile(target,'utf8'));}catch(e){if(e.code==='ENOENT')return defaultValue;throw e;}}
async function savePrivate(name,data){await privateDir();const target=path.join(HOME,name);try{let s=await fs.lstat(target);if(!s.isFile()||s.isSymbolicLink()||(s.mode&0o077)||(process.getuid&&s.uid!==process.getuid()))throw Error('UNSAFE_CONFIG_FILE');}catch(e){if(e.code!=='ENOENT')throw e;}
 const temp=path.join(HOME,'.setup-'+crypto.randomUUID());try{await fs.writeFile(temp,JSON.stringify(data,null,2)+'\n',{flag:'wx',mode:0o600});await fs.rename(temp,target);}finally{await fs.rm(temp,{force:true});}}
async function ask(vscode,title,{password=false,prompt='',validate}={}){let v=await vscode.window.showInputBox({title,prompt,password,ignoreFocusOut:true,validateInput:validate});if(v===undefined)throw Object.assign(Error('Setup cancelled'),{code:'SETUP_CANCELLED'});return v.trim();}
async function choose(vscode,title,items){let v=await vscode.window.showQuickPick(items,{title,ignoreFocusOut:true});if(!v)throw Object.assign(Error('Setup cancelled'),{code:'SETUP_CANCELLED'});return typeof v==='string'?v:v.value;}
async function modelSpec(vscode){const id=await ask(vscode,'Voder · Exact model ID');if(!/^[-\w./:+]{1,128}$/.test(id))throw Error('INVALID_MODEL_ID');const api=await choose(vscode,'Voder · Model API protocol',APIs);const baseUrl=validEndpoint(await ask(vscode,'Voder · Inference API endpoint',{prompt:'Hugging Face weights must first be served through a compatible endpoint'}));const key=assertSecret(await ask(vscode,'Voder · API key',{password:true}));const tokens=Number(await ask(vscode,'Voder · Context window',{prompt:'Use the model provider’s documented token limit'}));if(!Number.isSafeInteger(tokens)||tokens<1024||tokens>1048576)throw Error('INVALID_CONTEXT_WINDOW');return {id,api,baseUrl,key,tokens};}
async function setupPrimary(vscode){const m=await modelSpec(vscode);let vendor=await ask(vscode,'Voder · Model vendor label',{prompt:'e.g. deepseek, openai, custom'});if(!/^[a-z0-9_-]{1,40}$/i.test(vendor))throw Error('INVALID_VENDOR');await savePrivate('model-provider.json',{VODER_MODEL_ID:m.id,VODER_MODEL_VENDOR:vendor,VODER_MODEL_PROTOCOL:m.api,VODER_MODEL_ENDPOINT:m.baseUrl,VODER_MODEL_API_KEY:m.key,VODER_MODEL_CONTEXT_WINDOW:String(m.tokens),VODER_MODEL_MAX_TOKENS:String(Math.min(m.tokens,8192))});return 'Primary model saved';}
async function setupSecondary(vscode){const m=await modelSpec(vscode);const name=await ask(vscode,'Voder · Additional model alias',{prompt:'Lowercase letters, numbers, dots and hyphens; e.g. glm-5.3-flash'});if(!/^[a-z0-9.-]{1,48}$/.test(name))throw Error('INVALID_MODEL_ALIAS');const provider='voder-secondary-'+name;const catalog=await readPrivate('additional-models.json',{version:1,models:[]});if(catalog.version!==1||!Array.isArray(catalog.models))throw Error('INVALID_MODEL_CATALOG');const item={provider,id:m.id,api:m.api,baseUrl:m.baseUrl,apiKey:m.key,contextWindow:m.tokens,maxTokens:Math.min(m.tokens,8192),reasoning:false,thinkingLevel:'off',input:['text'],compat:{}};applyRequiredModelSettings(item);await savePrivate('additional-models.json',{version:1,models:[...catalog.models.filter(x=>x.provider!==provider),item]});return 'Additional model saved';}
async function setupModal(vscode){const id=assertSecret(await ask(vscode,'Voder · Modal token ID',{password:true}));const secret=assertSecret(await ask(vscode,'Voder · Modal token secret',{password:true}));const app=await ask(vscode,'Voder · Modal app name',{prompt:'Required by the existing Voder native runtime'});if(!/^[a-z0-9][a-z0-9-]{0,62}$/i.test(app))throw Error('INVALID_MODAL_APP');await savePrivate('modal-provider.json',{MODAL_TOKEN_ID:id,MODAL_TOKEN_SECRET:secret,VODER_MODAL_APP:app});return 'Modal credentials saved (not live verified)';}
// Native search accepts up to eight independently configured providers.
// Save declarative definitions in the private user profile: never into the extension install tree.
const SEARCH_ENV_PREFIX='VODER_SEARCH_';
const SEARCH_CONFIG='search-config.json';
function validSearchUrl(value){
 const u=new URL(value);
 if(u.username||u.password||u.hash||u.search||u.protocol!=='https:' && !(u.protocol==='http:'&&['127.0.0.1','[::1]'].includes(u.hostname)))throw Error('INVALID_SEARCH_ENDPOINT');
 return u.href;
}
function validateSearchConfig(obj){
 if(!obj||obj.version!==1||!Array.isArray(obj.providers)||obj.providers.length>8)throw Error('INVALID_SEARCH_CONFIG');
 const seen=new Set();
 for(const p of obj.providers){
  if(!p||typeof p.id!=='string'||!/^[-a-zA-Z0-9_]{1,64}$/.test(p.id)||seen.has(p.id))throw Error('INVALID_SEARCH_PROVIDER');
  seen.add(p.id);
  if(p.preset!==undefined&&!SEARCH.includes(p.preset))throw Error('INVALID_SEARCH_PRESET');
  if(p.preset===undefined){if(typeof p.endpoint!=='string'||!p.endpoint||!['GET','POST'].includes(p.method)||!p.response||typeof p.response!=='object'||Array.isArray(p.response))throw Error('INVALID_CUSTOM_SEARCH_PROVIDER');}
  if(p.preset==='searxng'&&(typeof p.endpoint!=='string'||!p.endpoint))throw Error('INVALID_SEARCH_ENDPOINT');
 }
 return obj;
}
// Custom providers use the original declarative GET/POST + JSON Pointer contract.
// This does not execute user-supplied JavaScript or allow arbitrary host commands.
function validPointer(value){
 if(typeof value!=='string'||!value.startsWith('/')||value.length>256||value.split('/').length>17||/(?:^|\/)(?:__proto__|constructor|prototype)(?:\/|$)/.test(value)||/~(?![01])/.test(value))throw Error('INVALID_JSON_POINTER');
 return value;
}
function validParamName(value){if(!/^[a-zA-Z_][a-zA-Z0-9_.-]{0,63}$/.test(value))throw Error('INVALID_QUERY_PARAMETER');return value;}
function validHeaderName(value){if(!/^[A-Za-z0-9-]{1,64}$/.test(value)||['host','content-length','connection','transfer-encoding','proxy-authorization','cookie'].includes(value.toLowerCase()))throw Error('INVALID_HEADER_NAME');return value;}
async function customSearch(vscode,alias){
 const endpoint=validSearchUrl(await ask(vscode,'Voder · Custom search API endpoint',{prompt:'HTTPS URL of your search API; local 127.0.0.1 HTTP is also supported'}));
 const method=await choose(vscode,'Voder · Search HTTP method',['GET','POST']);
 const queryField=validParamName(await ask(vscode,'Voder · Search query parameter / JSON body field',{prompt:'e.g. q, query, search_query'}));
 const auth=await choose(vscode,'Voder · Search API authentication',['No API key','Bearer token','Custom API key header']);
 const entry={id:alias,endpoint,method,response_format:'json',body_format:'json',response:{}};
 if(method==='GET')entry.params={[queryField]:'{{query}}'};
 else entry.body={[queryField]:'{{query}}'};
 let secret=null;
 if(auth!=='No API key'){
  const header=auth==='Bearer token'?'Authorization':validHeaderName(await ask(vscode,'Voder · API key header name',{prompt:'e.g. X-API-Key'}));
  const key=assertSecret(await ask(vscode,'Voder · Custom search API key',{password:true}));
  const envName=SEARCH_ENV_PREFIX+'CUSTOM_'+alias.replace(/[^A-Za-z0-9]/g,'_').toUpperCase()+'_API_KEY';
  entry.headers={[header]:auth==='Bearer token'?'Bearer {{secret:api_key}}':'{{secret:api_key}}'};
  entry.credentials={api_key:{env:envName}};
  secret={[envName]:key};
 }
 for(const [field,label] of [['items','results list'],['url','result URL'],['title','result title'],['excerpt','result snippet']]){
  entry.response[field]=validPointer(await ask(vscode,'Voder · JSON Pointer: '+label,{prompt:field==='items'?'e.g. /results':field==='url'?'/url':field==='title'?'/title':'/snippet'}));
 }
 if(new URL(endpoint).protocol==='http:')entry.allow_loopback=true;
 return {entry,secret};
}
async function setupSearch(vscode){
 const selection=await choose(vscode,'Voder · Search provider',SEARCH_OPTIONS.map(x=>({label:x==='custom'?'Custom API — enter provider and endpoint':x,value:x})));
 const alias=await ask(vscode,'Voder · Search provider name',{prompt:'Unique name; up to eight configured search providers'});
 if(!/^[-a-zA-Z0-9_]{1,64}$/.test(alias))throw Error('INVALID_SEARCH_PROVIDER_ID');
 const current=validateSearchConfig(await readPrivate(SEARCH_CONFIG,{version:1,providers:[]}));
 let entry={id:alias,preset:selection};let credentials=null;
 if(selection==='custom'){
  const result=await customSearch(vscode,alias);
  entry=result.entry;credentials=result.secret;
 }else if(selection==='searxng'){
  const endpoint=validSearchUrl(await ask(vscode,'Voder · SearXNG search endpoint',{prompt:'A running instance returning JSON search results'}));
  entry.endpoint=endpoint;
  const u=new URL(endpoint);
  if(u.protocol==='http:')entry.allow_loopback=true;
 }else{
  const key=assertSecret(await ask(vscode,`Voder · ${selection} search API key`,{password:true}));
  const envName=SEARCH_ENV_PREFIX+selection.toUpperCase()+'_API_KEY';
  entry.credentials={api_key:{env:envName}};
  credentials={[envName]:key};
 }
 const providers=[...current.providers.filter(p=>p.id!==alias),entry];
 if(providers.length>8)throw Error('SEARCH_PROVIDER_LIMIT_8');
 const next=validateSearchConfig({...current,providers});
 // Preserve every existing credential and provider. Save secrets before activating an entry.
 if(credentials){
  const existing=await readPrivate('search-provider-credentials.json',{});
  if(!existing||typeof existing!=='object'||Array.isArray(existing))throw Error('INVALID_SEARCH_CREDENTIALS');
  await savePrivate('search-provider-credentials.json',{...existing,...credentials});
 }
 await savePrivate(SEARCH_CONFIG,next);
 return 'Search provider '+alias+' configured (live connectivity not yet tested)';
}

async function searchStatus(vscode){
 const config=validateSearchConfig(await readPrivate(SEARCH_CONFIG,{version:1,providers:[]}));
 await vscode.window.showInformationMessage('Voder: '+config.providers.length+' search provider(s) configured. Keyless search remains available where supported.');
}
async function run(vscode){const options=[{label:'1 · Primary model',value:'primary'},{label:'2 · Add or replace another model',value:'secondary'},{label:'3 · Modal sandbox credentials',value:'modal'},{label:'4 · Configure external search provider',value:'search'},{label:'5 · Search configuration status',value:'searchStatus'},{label:'Done',value:'done'}];while(true){let which=await choose(vscode,'Voder · First-run configuration',options);if(which==='done')break;try{let result=await ({primary:setupPrimary,secondary:setupSecondary,modal:setupModal,search:setupSearch,searchStatus}[which])(vscode);await vscode.window.showInformationMessage('Voder: '+result);}catch(e){if(e.code==='SETUP_CANCELLED')break;throw e;}}}
module.exports={run,savePrivate,readPrivate,validEndpoint};
