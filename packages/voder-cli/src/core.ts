/** Voder operator CLI; daemon/VGR authorization remains exclusively in Rust. */
import {existsSync, lstatSync} from 'node:fs';
import {chmod, mkdir, open, readFile, rename, stat, unlink, readdir, copyFile} from 'node:fs/promises';
import {homedir} from 'node:os';
import {dirname, isAbsolute, join, resolve} from 'node:path';
import {randomBytes, createHash} from 'node:crypto';
import {spawn, spawnSync} from 'node:child_process';
import globalProcess from 'node:process';

export const VENDORS = Object.freeze(['openai','anthropic','gemini','kimi','glm','deepseek','grok','qwen','openrouter','custom'] as const);
export const RUNTIMES = Object.freeze(['vllm','sglang','ollama','llamacpp','compatible'] as const);
export type Vendor = typeof VENDORS[number];
export type Runtime = typeof RUNTIMES[number];
export type CloudSelection = {kind:'cloud';vendor:Vendor;model:string;protocol:'chat'|'responses';endpoint?:string};
export type OpenSelection = {kind:'open';alias:string};
export type Selection = CloudSelection|OpenSelection;
export type OpenProfile = {runtime:Runtime;endpoint:string;model:string;allowLoopbackHttp:boolean};
export type State = {version:1;plugins:{cloud:boolean;open:boolean};active:Selection|null;openProfiles:Record<string,OpenProfile>};
export const EMPTY_STATE:State={version:1,plugins:{cloud:false,open:false},active:null,openProfiles:{}};
export const validVendor = (v:string):v is Vendor=>VENDORS.some(x=>x===v);
export const validRuntime = (v:string):v is Runtime=>RUNTIMES.some(x=>x===v);
const validAlias=(v:string)=>/^[a-z][a-z0-9_-]{0,47}$/u.test(v) && !['cloud','open'].includes(v);
export const validModel=(v:string)=>typeof v==='string' && v.length>0 && v.length<=256 && !/[\u0000-\u001f\u007f]/u.test(v) && v.trim()===v;
export function validateEndpoint(s:string, allowLoopbackHttp=false):string {
  let u:URL;try{u=new URL(s);}catch{throw new Error('invalid endpoint URL');}
  const loopback=['localhost','127.0.0.1','[::1]'].includes(u.hostname);
  if (!(u.protocol==='https:'||(allowLoopbackHttp&&loopback&&u.protocol==='http:'))) throw new Error('endpoint requires HTTPS (loopback HTTP requires explicit opt-in)');
  if (u.username||u.password||u.search||u.hash||u.pathname.split('/').includes('..'))throw new Error('endpoint must not contain credentials/query/hash/..');
  return u.toString().replace(/\/$/u,'');
}
export function parseOpts(args:string[]):{pos:string[];opts:Record<string,string|true>} {
  const pos:string[]=[];const opts:Record<string,string|true>={};
  const flags=new Set(['stdin','allow-loopback-http','live','help']);
  for(let i=0;i<args.length;i++){
    const a=args[i]!;
    if(a.startsWith('--')){
      const key=a.slice(2);if(!/^[a-z][a-z0-9-]*$/u.test(key)||key in opts)throw new Error('invalid/duplicate option');
      if(flags.has(key))opts[key]=true;
      else {const val=args[++i];if(!val||val.startsWith('--'))throw new Error(`missing value --${key}`);opts[key]=val;}
    }else if(a.startsWith('-'))throw new Error('unknown short option');else pos.push(a);
  }
  return {pos,opts};
}
export function requireOpts(opts:Record<string,string|true>, allowed:readonly string[]):void{
  const unknown=Object.keys(opts).filter(k=>!allowed.includes(k));if(unknown.length)throw new Error(`unsupported option(s): ${unknown.join(',')}`);
}
export function strOpt(opts:Record<string,string|true>,key:string):string|undefined {const v=opts[key];if(v===undefined)return undefined;if(typeof v!=='string')throw new Error(`option --${key} requires value`);return v;}
export function privateDir():string {return resolve(globalProcess.env.VODER_CONFIG_DIR||join(homedir(),'.config','voder'));}
export function repoRoot():string {const root=globalProcess.env.VODER_ROOT; if(!root || !isAbsolute(root)) throw new Error('VODER_ROOT must be an absolute trusted Voder project directory');return resolve(root);}
async function assertPrivateDir(dir:string):Promise<void>{
  if(existsSync(dir)) {if(lstatSync(dir).isSymbolicLink())throw new Error('configuration directory must not be symlink');const st=await stat(dir);if(!st.isDirectory()||(st.mode&0o077)!==0)throw new Error('insecure configuration directory permissions');}
  else await mkdir(dir,{recursive:true,mode:0o700});
  await chmod(dir,0o700);
}
export async function ensurePrivate():Promise<string>{const dir=privateDir();await assertPrivateDir(dir);return dir;}
async function secureRead(path:string):Promise<string|undefined>{
  if(!existsSync(path))return undefined;
  const st=lstatSync(path);if(st.isSymbolicLink()||!st.isFile()||(st.mode&0o077)!==0||st.size>128*1024)throw new Error('insecure or oversized private configuration');
  return readFile(path,'utf8');
}
async function atomicWrite(path:string,data:string):Promise<void>{
  const tmp=join(dirname(path),`.tmp-${randomBytes(16).toString('hex')}`);
  const f=await open(tmp,'wx',0o600);let success=false;
  try{await f.writeFile(data,'utf8');await f.sync();await f.close();await rename(tmp,path);await chmod(path,0o600);success=true;}
  finally{if(!success){await f.close().catch(()=>{});await unlink(tmp).catch(()=>{});}}
}
export function validateState(x:unknown):State{
 if(!x||typeof x!=='object'||Array.isArray(x))throw new Error('invalid Voder state');
 const s=x as Record<string,unknown>;if(s.version!==1)throw new Error('unsupported Voder configuration version');
 const p=s.plugins as Record<string,unknown>;
 if(!p||typeof p.cloud!=='boolean'||typeof p.open!=='boolean')throw new Error('invalid plugins state');
 if(!s.openProfiles||typeof s.openProfiles!=='object'||Array.isArray(s.openProfiles))throw new Error('invalid open-model profiles');
 const profiles:Record<string,OpenProfile>={};
 for(const [alias,raw] of Object.entries(s.openProfiles)){
   if(!validAlias(alias)||!raw||typeof raw!=='object'||Array.isArray(raw))throw new Error('invalid model alias');
   const o=raw as Record<string,unknown>;
   if(typeof o.runtime!=='string'||!validRuntime(o.runtime)||typeof o.endpoint!=='string'||typeof o.model!=='string'||!validModel(o.model)||typeof o.allowLoopbackHttp!=='boolean')throw new Error('invalid inference profile');
   profiles[alias]={runtime:o.runtime,endpoint:validateEndpoint(o.endpoint,o.allowLoopbackHttp),model:o.model,allowLoopbackHttp:o.allowLoopbackHttp};
 }
 let active:Selection|null=null;
 if(s.active!==null){
   if(!s.active||typeof s.active!=='object'||Array.isArray(s.active))throw new Error('invalid active model');
   const a=s.active as Record<string,unknown>;
   if(a.kind==='cloud'){
     if(typeof a.vendor!=='string'||!validVendor(a.vendor)||typeof a.model!=='string'||!validModel(a.model)||!['chat','responses'].includes(String(a.protocol)))throw new Error('invalid cloud model selection');
     if(a.protocol==='responses'&&a.vendor!=='openai'&&a.vendor!=='custom')throw new Error('Responses protocol requires OpenAI/custom provider');
     active={kind:'cloud',vendor:a.vendor,model:a.model,protocol:a.protocol as 'chat'|'responses',...(typeof a.endpoint==='string'?{endpoint:validateEndpoint(a.endpoint)}:{})};
   } else if(a.kind==='open'&&typeof a.alias==='string'&&profiles[a.alias])active={kind:'open',alias:a.alias};
   else throw new Error('invalid open model selection');
 }
 return {version:1,plugins:{cloud:p.cloud,open:p.open},active,openProfiles:profiles};
}
export async function loadState():Promise<State>{const dir=await ensurePrivate();const raw=await secureRead(join(dir,'state.json'));return raw===undefined?structuredClone(EMPTY_STATE):validateState(JSON.parse(raw));}
export async function saveState(s:State):Promise<void>{const dir=await ensurePrivate();const valid=validateState(s);await atomicWrite(join(dir,'state.json'),JSON.stringify(valid,null,2)+'\n');}
export async function credentialExists(vendor:string):Promise<boolean>{if(!validVendor(vendor))return false;const dir=await ensurePrivate();const d=join(dir,'credentials');if(!existsSync(d))return false;await assertPrivateDir(d);return !!(await secureRead(join(d,`${vendor}.json`)));}
export async function storeCredential(vendor:string,key:string):Promise<void>{
 if(!validVendor(vendor))throw new Error('unknown cloud vendor');
 if(!key||key.length>16000||/[\r\n\u0000]/u.test(key))throw new Error('invalid key');
 const dir=await ensurePrivate();const d=join(dir,'credentials');await assertPrivateDir(d);await atomicWrite(join(d,`${vendor}.json`),JSON.stringify({version:1,key})+'\n');
}
export async function removeCredential(target:string):Promise<void>{
 if(!(validVendor(target)||(/^open:[a-z][a-z0-9_-]{0,47}$/u.test(target))))throw new Error('invalid credential reference');
 const dir=await ensurePrivate();const d=join(dir,'credentials');if(!existsSync(d))return;
 await assertPrivateDir(d);const name=target.startsWith('open:')?`open-${target.slice(5)}`:target;
 const path=join(d,`${name}.json`);if(await secureRead(path)!==undefined)await unlink(path);
}
export async function storeOpenCredential(alias:string,key:string):Promise<void>{
 if(!/^[a-z][a-z0-9_-]{0,47}$/u.test(alias)||!key||key.length>16000||/[\r\n\u0000]/u.test(key))throw new Error('invalid inference credential');
 const dir=await ensurePrivate();const d=join(dir,'credentials');await assertPrivateDir(d);
 await atomicWrite(join(d,`open-${alias}.json`),JSON.stringify({version:1,key})+'\n');
}
export async function loadOpenCredential(alias:string):Promise<string|undefined>{
 if(!/^[a-z][a-z0-9_-]{0,47}$/u.test(alias))throw new Error('invalid open alias');
 const dir=await ensurePrivate();const d=join(dir,'credentials');if(!existsSync(d))return undefined;
 await assertPrivateDir(d);const raw=await secureRead(join(d,`open-${alias}.json`));
 if(!raw)return undefined;const j=JSON.parse(raw) as {version?:number;key?:string};
 if(j.version!==1||typeof j.key!=='string'||!j.key)throw new Error('invalid inference credential');return j.key;
}
export async function loadCredential(vendor:string):Promise<string>{if(!validVendor(vendor))throw new Error('unknown vendor');const dir=await ensurePrivate();const d=join(dir,'credentials');await assertPrivateDir(d);const raw=await secureRead(join(d,`${vendor}.json`));if(!raw)throw new Error(`missing credential: voder model login ${vendor}`);const j=JSON.parse(raw) as {version?:number;key?:string};if(j.version!==1||typeof j.key!=='string'||!j.key)throw new Error('invalid credential file');return j.key;}
export function assertInstalled(root:string,id:'cloud'|'open'):void{
 const path=join(root,'packages',`voder-provider-${id}`,'dist','index.js');
 if(!existsSync(path)||lstatSync(path).isSymbolicLink()||!lstatSync(path).isFile())throw new Error(`model plugin not installed: ${id}`);
}
export async function pluginAvailable(root:string,id:'cloud'|'open'):Promise<boolean>{try{assertInstalled(root,id);return true;}catch{return false;}}
export async function modelEnvironment(s:State,root:string):Promise<Record<string,string>>{
 if(!s.active)throw new Error('no model selected; use voder model use');
 const a=s.active;const bridge=join(root,'packages/voder-providers/bin/daemon-provider-bridge.mjs');
 if(!existsSync(bridge))throw new Error('provider-daemon bridge missing');
 if(a.kind==='cloud'){
   if(!s.plugins.cloud)throw new Error('cloud plugin disabled');assertInstalled(root,'cloud');
   return {VODER_MODEL_PLUGIN:'cloud',VODER_CLOUD_VENDOR:a.vendor,VODER_CLOUD_MODEL:a.model,VODER_CLOUD_PROTOCOL:a.protocol,
      VODER_CLOUD_API_KEY:await loadCredential(a.vendor),...(a.endpoint?{VODER_CLOUD_ENDPOINT:a.endpoint}:{})};
 }
 if(!s.plugins.open)throw new Error('open plugin disabled');assertInstalled(root,'open');
 const p=s.openProfiles[a.alias];if(!p)throw new Error('open profile missing');
 return {VODER_MODEL_PLUGIN:'open',VODER_OPEN_RUNTIME:p.runtime,VODER_OPEN_ENDPOINT:p.endpoint,VODER_OPEN_MODEL:p.model,
 ...(p.allowLoopbackHttp?{VODER_OPEN_ALLOW_LOOPBACK_HTTP:'1'}:{})};
}
export function nodeExecutable():string {return globalProcess.execPath;}
export function bridgePath(root:string):string {return join(root,'packages/voder-providers/bin/daemon-provider-bridge.mjs');}
export function safeStateDir(root:string):string {return join(root,'.voder-state');}
export async function runProviderDoctor(root:string,env:Record<string,string>):Promise<{providerId:string;model:string}>{
 const tmpRoot=await ensurePrivate();const d=join(tmpRoot,`doctor-${randomBytes(12).toString('hex')}`);await mkdir(d,{mode:0o700});
 const input=join(d,'input.json'),out=join(d,'result.json');
 try {
   await atomicWrite(input,JSON.stringify({protocol:'voder-provider-bridge-v1',runId:`r-${BigInt('0x'+randomBytes(6).toString('hex'))+1n}`,instruction:'Respond with the single word READY.'}));
   const child=spawn(nodeExecutable(),[bridgePath(root),input,out],{cwd:root,env:{...globalProcess.env,...env,NODE_OPTIONS:''},stdio:['ignore','ignore','pipe']});
   let stderrBytes=0;
   child.stderr?.on('data',(chunk:Buffer)=>{stderrBytes+=chunk.length;});
   const code=await new Promise<number>((res,rej)=>{const t=setTimeout(()=>{child.kill('SIGKILL');rej(new Error('model request timed out'))},190000);child.on('error',(err)=>{clearTimeout(t);rej(err)});child.on('close',(n)=>{clearTimeout(t);res(n??1)});});
   if(code!==0)throw new Error(`model bridge failed (exit ${code}); check endpoint, model access, credentials and protocol`);
   const raw=await secureRead(out);if(!raw||Buffer.byteLength(raw)>256*1024)throw new Error('missing or excessive model receipt');
   const receipt=JSON.parse(raw) as Record<string,unknown>;
   if(receipt.protocol!=='voder-provider-bridge-v1'||receipt.executionAuthorized!==false||receipt.deliveryAllowed!==false||typeof receipt.providerId!=='string'||!receipt.providerId||typeof receipt.model!=='string'||!receipt.model)throw new Error('model receipt failed authorization invariant');
   return {providerId:receipt.providerId,model:receipt.model};
 }finally{
   for(const f of [input,out])await unlink(f).catch(()=>{});
   const names=await readdir(d).catch(()=>[]);if(names.length===0)await (await import('node:fs/promises')).rmdir(d).catch(()=>{});
 }
}
export async function assertWorkspace(root:string):Promise<void>{
 if(!isAbsolute(root)||!existsSync(join(root,'Cargo.toml'))||!existsSync(join(root,'crates/voderd/src/provider_bridge.rs'))||!existsSync(join(root,'packages/voder-providers/bin/daemon-provider-bridge.mjs')))throw new Error('not an integrated Voder module-seven workspace');
}
export function printHelp():string {return `Voder model management (Node >=22)\n\n  voder plugin list\n  voder plugin install provider-cloud|provider-open\n  voder plugin remove provider-cloud|provider-open  # disable only, retains source\n  voder model login openai|anthropic|gemini|kimi|glm|deepseek|grok|qwen|openrouter|custom [--stdin]\n  voder model logout VENDOR|open:ALIAS\n  voder model auth open:ALIAS [--stdin] # optional authenticated inference endpoint\n  voder model add ALIAS --runtime vllm|sglang|ollama|llamacpp|compatible --endpoint URL --model ID [--allow-loopback-http] [--replace]\n  voder model use VENDOR/MODEL [--protocol chat|responses] [--endpoint HTTPS_URL]\n  voder model use open:ALIAS\n  voder model status\n  voder doctor [--live]        # --live makes one billable model request\n  voder daemon start [--state-dir ABSOLUTE_DIR] # stdio transport\n\nSecrets are read interactively (no echo) or from stdin; never pass keys as command arguments.\n`}
export function sha256(buf:Buffer|string):string{return createHash('sha256').update(buf).digest('hex')}
