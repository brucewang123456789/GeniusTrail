/** Multi-project candidate admission only. Native verification, commit and delivery remain upstream. */
import {createRequire} from 'node:module';
import {randomUUID} from 'node:crypto';
import {mkdir,lstat,open,realpath} from 'node:fs/promises';
import {join,resolve,parse,relative,sep} from 'node:path';
const require=createRequire(import.meta.url);
const {inspectSourceRows,hash}=require('./workspace-tools.cjs');
const LIMITS=Object.freeze({maxPreflightCalls:12,maxSandboxCalls:3,maxFileBytes:null,maxProjectBytes:null});
const codeOf=e=>typeof e?.code==='string'&&/^[A-Z][A-Z0-9_]{0,95}$/.test(e.code)?e.code:'TOOLKIT_VERIFIER_EXCEPTION';
const redacted=s=>String(s??'').replace(/Bearer\s+\S+|\b(?:sk-|sk_)[A-Za-z0-9_-]+/gi,'[REDACTED]').replace(/((?:api[_-]?key|token|secret|authorization)["'\s]*[:=]["'\s]*)[^\s,;}]+/gi,'$1[REDACTED]');
const failure=(code,message)=>Object.assign(new Error(message),{code});
export function privateDiagnosticWriter(attemptStore){
 const root=resolve(attemptStore),dir=join(root,'verification-diagnostics');
 return async record=>{
  // A diagnostic path never follows a linked ancestor or uses a model-supplied filename.
  let current=parse(root).root;
  for(const part of relative(current,root).split(sep).filter(Boolean)){
   current=join(current,part);const s=await lstat(current);if(s.isSymbolicLink()||!s.isDirectory())throw failure('TOOLKIT_DIAGNOSTIC_PATH','Diagnostic ancestor is not a regular directory');
  }
  if(await realpath(root)!==root)throw failure('TOOLKIT_DIAGNOSTIC_PATH','Diagnostic root identity changed');
  await mkdir(dir,{mode:0o700}).catch(e=>{if(e.code!=='EEXIST')throw e;});
  const st=await lstat(dir);if(st.isSymbolicLink()||!st.isDirectory()||(st.mode&0o077)!==0||(process.getuid&&st.uid!==process.getuid()))throw failure('TOOLKIT_DIAGNOSTIC_PATH','Diagnostic directory is not private');
  const body=JSON.stringify({...record,schema:'voder.candidate.diagnostic.v1',authority:'DIAGNOSTIC_ONLY_NOT_VGR_PROOF',time:new Date().toISOString()});
  if(Buffer.byteLength(body)>24576)throw failure('TOOLKIT_DIAGNOSTIC_LIMIT','Diagnostic record exceeds its bound');
  const fd=await open(join(dir,randomUUID()+'.json'),'wx',0o600);try{await fd.writeFile(body+'\n');await fd.sync();}finally{await fd.close();}
 };
}
export class SyntaxThenSandboxVerifier{
 constructor(base,{targets,node,report}={}){
  this.base=base;this.targets=Object.freeze([...targets]);this.node=node;this.report=report;this.counts=new Map();this.preflightCounts=new Map();
 }
 describe(id){
  return {schema:'voder.candidate.contract.v1',...LIMITS,exact_sorted_targets:[...this.targets],full_batch_required:true,
   ...(id!==undefined?{preflight_calls_used:this.preflightCounts.get(id)||0,sandbox_calls_used:this.counts.get(id)||0}:{}),
   local_rejection_consumes_sandbox_budget:false,transport_attempt_is_never_refunded_or_replayed:true,
   success_meaning:'PASS proves only pinned validator acceptance, never task completeness, visual quality or delivery.',
   completion:'Propose a complete solution, not an empty probe. After exact task-satisfying bytes PASS, explicitly call voder_commit_verified_candidate with that digest. Never commit a probe to manufacture success.',
   recovery:'Fix the exact reported issue within the same scope. On exhausted budget, an earlier PASS may be committed explicitly only if those exact bytes satisfy the task. Otherwise report the blocker; never promote FAIL, partial output or uncertain execution.'};
 }
 async _record(record,{preserveOutcome=false}={}){
  if(!this.report)return;
  try{await this.report(record);}catch{
   // Observability failure must not discard an already obtained authoritative receipt.
   if(preserveOutcome)process.stderr.write('VODER_CANDIDATE_DIAGNOSTIC_WRITE_FAILED\n');
   if(!preserveOutcome)throw failure('TOOLKIT_DIAGNOSTIC_WRITE_FAILED','Cannot persist candidate diagnostics; no sandbox call started. Inspect private storage before retrying.');
  }
 }
 async verifyCandidate(candidate,options={}){
  const id=options.sessionId??'host';let phase='admission';
  const attempt=(this.preflightCounts.get(id)||0)+1;
  const members=candidate?.patches;
  const summary={session_id:id,candidate_digest:options.candidate_digest??null,attempt,
   files:Array.isArray(members)?members.map(p=>({path:typeof p?.relative_path==='string'?p.relative_path.slice(0,240):null,bytes:typeof p?.new_content==='string'?Buffer.byteLength(p.new_content):null})).slice(0,16):[],
   limits:LIMITS};
  try{
   if(attempt>LIMITS.maxPreflightCalls)throw failure('TOOLKIT_PREFLIGHT_BUDGET','Candidate admission budget exhausted; use an earlier exact task-satisfying PASS only through explicit verified commit, otherwise report the blocker.');
   this.preflightCounts.set(id,attempt);
   if(!Array.isArray(members)||JSON.stringify(members.map(p=>p?.relative_path))!==JSON.stringify(this.targets))throw failure('TOOLKIT_CANDIDATE_SCOPE','Submit the complete exact sorted file list from voder_ext_project_capabilities. Do not add, omit, reorder or rename approved paths. No sandbox budget was spent.');
   const bad=members.find(p=>typeof p.new_content!=='string'||Buffer.byteLength(p.new_content)<1||p.new_content.includes('\0'));
   if(bad)throw failure('TOOLKIT_CANDIDATE_SIZE','Each approved UTF-8 file must be nonempty and contain no NUL. No sandbox budget was spent.');
   phase='parser';
   const rows=members.map(p=>({relative:p.relative_path,bytes:Buffer.from(p.new_content),sha256:hash(p.new_content)}));
   const checks=await inspectSourceRows(rows,{node:this.node});
   const missing=checks.filter(c=>c.status==='UNAVAILABLE');
   if(missing.length)throw failure('TOOLKIT_PARSER_UNAVAILABLE','Host parser is unavailable for '+missing.map(c=>c.file).join(', ').slice(0,200)+'. Do not rewrite working source to fix host environment; no sandbox budget was spent.');
   const failed=checks.filter(c=>c.status==='FAIL');
   if(failed.length){
    const result={status:'FAIL',backend:'host-passive-parser',sandboxed:false,checks:failed.map(c=>({name:c.file,passed:false})),diagnostic:failed.map(c=>c.file+': '+c.diagnostic).join('\n').slice(-4000)};
    await this._record({...summary,phase,outcome:'FAIL',sandbox_calls:this.counts.get(id)||0,checks:failed.map(c=>({file:c.file,status:c.status,diagnostic:redacted(c.diagnostic).slice(0,1000)}))});
    return result;
   }
   phase='sandbox-admission';
   const used=this.counts.get(id)||0;
   if(used>=LIMITS.maxSandboxCalls)throw failure('TOOLKIT_CANDIDATE_BUDGET','Three independent sandbox attempts are already reserved. Do not reverify or submit changed bytes. Explicitly commit a previous exact PASS only if it satisfies the task; otherwise report the blocker.');
   await this._record({...summary,phase:'sandbox-intent',outcome:'REQUESTED',sandbox_calls:used});
   // Reserve atomically after all asynchronous local checks/diagnostic writes.
   const now=this.counts.get(id)||0;
   if(now>=LIMITS.maxSandboxCalls)throw failure('TOOLKIT_CANDIDATE_BUDGET','Sandbox budget exhausted by another reserved call; never replay.');
   this.counts.set(id,now+1);phase='sandbox';
   const result=await this.base.verifyCandidate(candidate,options);
   await this._record({...summary,phase,outcome:result?.status??'INVALID',sandbox_calls:this.counts.get(id),backend:result?.backend??null},{preserveOutcome:true});
   // Return the original receipt, including PASS, unchanged. Only the SDK and Rust may grant commit/delivery.
   return result;
  }catch(error){
   const code=codeOf(error);
   await this._record({...summary,phase,outcome:'ERROR',code,message:redacted(error?.message).slice(0,2000),stack:redacted(error?.stack).slice(0,6000),sandbox_calls:this.counts.get(id)||0},{preserveOutcome:true});
   if(typeof error?.code==='string'&&/^[A-Z][A-Z0-9_]{0,95}$/.test(error.code))throw error;
   throw failure(code,'Unexpected host candidate verifier failure; inspect private verification-diagnostics. Do not blindly retry or claim delivery.');
  }
 }
}
