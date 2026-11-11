import {createHash,randomUUID} from 'node:crypto';
import {readFile,writeFile,mkdir,open,lstat,realpath} from 'node:fs/promises';
import {resolve,join,dirname} from 'node:path';
import {spawn} from 'node:child_process';
import {fileURLToPath} from 'node:url';
export const moduleRoot=resolve(dirname(fileURLToPath(import.meta.url)),'..');
export const sha=x=>createHash('sha256').update(x).digest('hex');
export const time=()=>new Date().toISOString();
export const readJson=async p=>JSON.parse(await readFile(p,'utf8'));
export const fileHash=async p=>sha(await readFile(p));
export function requireThat(cond,code){if(!cond){const e=new Error(code);e.code=code;throw e;}}
export async function privateDir(path){await mkdir(path,{recursive:true,mode:0o700});const s=await lstat(path);requireThat(s.isDirectory()&&!s.isSymbolicLink()&&(s.mode&0o077)===0,'OUTPUT_NOT_PRIVATE');return path;}
export async function persist(path,value){const f=await open(path,'wx',0o600);try{await f.writeFile(JSON.stringify(value,null,2)+'\n');await f.sync();}finally{await f.close()}}
export function processRun(program,args,{cwd,timeoutMs=120000,env=process.env}={}){return new Promise((ok)=>{
 const start=Date.now();let out='',err='',killed=false;let child;
 try{child=spawn(program,args,{cwd,env,shell:false,stdio:['ignore','pipe','pipe']});}
 catch(e){return ok({exit_code:null,error_code:e.code??'SPAWN_ERROR',duration_ms:Date.now()-start});}
 const limit=65536;const cap=(s,x)=>(s+x.toString('utf8')).slice(-limit);
 child.stdout.on('data',d=>out=cap(out,d));child.stderr.on('data',d=>err=cap(err,d));
 const timeout=setTimeout(()=>{killed=true;child.kill('SIGKILL')},timeoutMs);
 child.on('error',e=>{err=cap(err,e.code??'PROCESS_ERROR')});
 child.on('close',(code,signal)=>{clearTimeout(timeout);ok({exit_code:code,signal,timeout:killed,duration_ms:Date.now()-start,stdout:out,stderr:err})});
 });}
export function taskRunId(){return time().replace(/[:.]/g,'-')+'-'+randomUUID().slice(0,8);}
export function safeRuntimeEnv(){const e={...process.env,VODER_ALLOW_MODAL_CHARGES:'',VODER_ALLOW_MODEL_API:'',VODER_MODEL_API_KEY:'',OPENAI_API_KEY:'',ANTHROPIC_API_KEY:''};return e;}
export async function canonicalAgent(input){const r=await realpath(resolve(input));requireThat(r===resolve(input),'AGENT_ROOT_NONCANONICAL');return r;}
