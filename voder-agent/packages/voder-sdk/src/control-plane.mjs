/** Voder SDK control-plane v1: single live owner, durable read-only replay, no native authority impersonation. */
import {randomUUID, createHash} from 'node:crypto';
import {open, readFile, writeFile, mkdir, rename, lstat, readdir, unlink, rmdir} from 'node:fs/promises';
import {join, resolve, dirname, parse, relative, sep} from 'node:path';
import {VoderError} from '../../voder-agent-core/src/contracts.mjs';
import {NativeSdkContinuation} from './native-continuation.mjs';
import {readNativeSessionDiff} from './native-diff.mjs';

const UUID=/^[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$/i;
const SAFE_EVENT_KEYS=['summary','image_count','images_sha256','model','stop_reason','output_tokens','tool_call_count','tool_calls_digest','upstream_error_sha256','request_sequence','recovery_count','effective_max_tokens','registered_max_tokens','context_window','limit_source','capability_verified','tool','provider','model_id','model_epoch','context_sha256','code','action_id','verdict','status','count','files','bytes','policy_digest','candidate_digest','reason','context_bytes_upper_bound','context_estimate_tokens','context_source','usage_tokens','trailing_estimate_tokens','context_remaining_tokens','context_reserve_tokens','finalization_reserve_tokens','guard_branch','ledger_unchanged','status','reason','usage_input','usage_cacheRead','usage_cacheWrite','usage_totalTokens'];
const BUDGET_KEYS=['tool_calls','read_bytes','verification_calls','evidence_revisions','no_progress','positive_sandbox_checks','attempted_native_actions','exploration_calls','remaining_tool_calls','reserved_finalization_calls','remaining_verification_calls','repair_read_credits','repair_reads'];
function budgetSummary(value){
 if(!value||typeof value!=='object')return undefined;
 const data={};
 for(const k of BUDGET_KEYS)if(Number.isSafeInteger(value[k])&&value[k]>=0&&value[k]<=16777216)data[k]=value[k];
 if(['EXPLORING','FINALIZING'].includes(value.phase))data.phase=value.phase;
 if(typeof value.phase_reason==='string'&&/^[A-Z][A-Z0-9_]{0,95}$/.test(value.phase_reason))data.phase_reason=value.phase_reason;
 return Object.keys(data).length?data:undefined;
}
const digest=v=>createHash('sha256').update(v).digest('hex');
const fail=(code,msg)=>{throw new VoderError(code,msg)};
const terminal=new Set(['VERIFIED','VERIFIED_NOT_DELIVERED','DELIVERED','CANCELLED','FAILED','BLOCKED','DRAFT_UNVERIFIED','WAITING_FOR_INPUT']);
const MAX_LIVE_DRAFT_BYTES=131072;
function userVisibleResult(result,summary){
 // Only the live call owner can read a model draft. No text is written to the journal.
 if(summary.status!=='DRAFT_UNVERIFIED'||result?.verified!==false||typeof result.draft!=='string')return null;
 if(!result.draft.trim()||Buffer.byteLength(result.draft,'utf8')>MAX_LIVE_DRAFT_BYTES)return null;
 return Object.freeze({draft:result.draft,verified:false,output_authority:'UNVERIFIED_MODEL_TEXT_NOT_VGR_PROOF'});
}

function eventProjection(event){
 const data={sequence:event.sequence,type:event.type,state:event.state};
 for(const k of SAFE_EVENT_KEYS)if(['string','number','boolean'].includes(typeof event[k])){
   const val=event[k];if(typeof val!=='string'||val.length<=160)data[k]=val;
 }
 const budget=budgetSummary(['orchestration.phase','orchestration.budget'].includes(event.type)?event:event.orchestration);
 if(budget)data.orchestration=budget;
 return data;
}
async function assertNoLinkAncestors(path){
 const absolute=resolve(path);const root=parse(absolute).root;let cursor=root;
 for(const part of relative(root,absolute).split(sep).filter(Boolean)){
  cursor=join(cursor,part);const st=await lstat(cursor).catch(e=>{if(e.code==='ENOENT')return null;throw e;});
  if(st?.isSymbolicLink())fail('UNSAFE_PATH','Symlink path components forbidden');
 }
}
async function privateDirectory(dir){
 await assertNoLinkAncestors(dir);
 await mkdir(dir,{recursive:true,mode:0o700});
 const s=await lstat(dir);
 if(!s.isDirectory()||s.isSymbolicLink()||(s.mode&0o077))fail('UNSAFE_JOURNAL_DIR','Journal must be a private, non-symlink directory');
}
async function durableWrite(path,data){
 const tmp=join(dirname(path),`.tmp-${randomUUID()}`);let fd;
 try{
  fd=await open(tmp,'wx',0o600);await fd.writeFile(data);await fd.sync();await fd.close();fd=null;
  await rename(tmp,path);
  const dirFd=await open(dirname(path),'r');try{await dirFd.sync();}finally{await dirFd.close();}
 }finally{await fd?.close().catch(()=>{});await unlink(tmp).catch(()=>{});}
}
function validateSession(id){if(typeof id!=='string'||!UUID.test(id))fail('INVALID_SESSION_ID','Session identity must be a UUID');return id.toLowerCase();}
// User task bodies never enter the SDK journal. The trusted host supplies the
// exact task again on recovery, and its digest must match the durable checkpoint.
function recoveryTask(task){
 if(!task||typeof task!=='object'||Array.isArray(task)||Object.keys(task).some(k=>!['goal','requiredFiles','maxActions'].includes(k)))fail('RECOVERY_TASK_POLICY','Only bounded declarative tasks are resumable');
 if(typeof task.goal!=='string'||!task.goal.trim()||Buffer.byteLength(task.goal)>16384)fail('RECOVERY_TASK_POLICY','Invalid recoverable goal');
 const files=task.requiredFiles??[],budget=task.maxActions??12;
 if(!Array.isArray(files)||files.length>32||files.some(x=>typeof x!=='string'||x.length>240)||!Number.isInteger(budget)||budget<1||budget>64)
   fail('RECOVERY_TASK_POLICY','Invalid required files or action budget');
 return {goal:task.goal,requiredFiles:files,maxActions:budget};
}
const taskDigest=task=>digest(JSON.stringify(recoveryTask(task)));
async function ownerIdentity(pid=process.pid){
 // A Linux PID alone is recyclable; require same boot + kernel process start ticks.
 const [boot,stat]=await Promise.all([readFile('/proc/sys/kernel/random/boot_id','utf8'),readFile(`/proc/${pid}/stat`,'utf8')]);
 const end=stat.lastIndexOf(')');if(end<0)fail('RECOVERY_HOST_IDENTITY','Cannot parse Linux task identity');
 const ticks=stat.slice(end+2).trim().split(/\s+/)[19];
 if(!/^\d+$/.test(ticks??''))fail('RECOVERY_HOST_IDENTITY','Missing Linux process start time');
 return {pid,boot:boot.trim(),ticks};
}
async function oldOwnerAlive(owner){
 if(!owner||!Number.isInteger(owner.pid)||typeof owner.boot!=='string'||typeof owner.ticks!=='string')fail('RECOVERY_OWNER_UNKNOWN','Malformed crashed-task owner');
 const boot=(await readFile('/proc/sys/kernel/random/boot_id','utf8')).trim();
 if(owner.boot!==boot)fail('RECOVERY_DIFFERENT_BOOT','Cross-boot automatic recovery is not authorized');
 try{return (await ownerIdentity(owner.pid)).ticks===owner.ticks;}
 catch(e){if(e.code==='ENOENT'||e.code==='ESRCH')return false;throw e;}
}
function safeSummary(result){
 const status=String(result?.status??'UNKNOWN');
 return {status,verified:(status==='VERIFIED'||status==='VERIFIED_NOT_DELIVERED'||status==='DELIVERED') && result?.nativeCommit?.status==='COMMITTED',
   ...(typeof result?.blocked_reason==='string'&&/^[A-Z][A-Z0-9_]{0,95}$/.test(result.blocked_reason)?{blocked_reason:result.blocked_reason}:{}),
   ...(budgetSummary(result?.orchestration)?{orchestration:budgetSummary(result.orchestration)}:{}),
   ...(typeof result?.model_error_code==='string'&&/^[A-Z][A-Z0-9_]{0,95}$/.test(result.model_error_code)?{model_error_code:result.model_error_code}:{}),
   ...(result?.reporting_error_code==='POST_COMMIT_CHECKPOINT_FAILED'?{reporting_error_code:result.reporting_error_code}:{}),
   ...(result?.cancel_requested===true?{cancel_requested:true}:{}),
   ...(result?.delivery?.status?{delivery_status:result.delivery.status}:{}),
   ...(result?.delivery?.code?{delivery_error_code:result.delivery.code}:{}),
   ...(result?.nativeCommit?.commit_digest?{commit_digest:result.nativeCommit.commit_digest}:{}),
   ...(result?.nativeCommit?.run_id?{run_id:result.nativeCommit.run_id}:{}),
   ...(result?.task_acceptance?{task_acceptance:result.task_acceptance}:{}),
   ...(result?.missing?{missing:result.missing}:{}),
   ...(result?.draft?{draft_available:true}:{}),
   ...(result?.skills?{skills:result.skills}:{}),
   ...(result?.orchestration?.status?{orchestration_status:result.orchestration.status}:{})};
}
/** Journal is untrusted task history; SHA256 detects corruption, not signature or VGR proof. */
export class VoderJournal {
 constructor({dir,workspace}){
  if(typeof dir!=='string'||!dir.trim()||typeof workspace!=='string'||!workspace.trim())fail('INVALID_JOURNAL','Explicit journal and workspace required');
  this.dir=resolve(dir);this.workspace=resolve(workspace);
 }
 async initialize(){await privateDirectory(this.dir);await privateDirectory(join(this.dir,'sessions'));return this;}
 _path(id){return join(this.dir,'sessions',validateSession(id));}
 async _read(id){
  const d=this._path(id);const st=await lstat(d).catch(e=>{if(e.code==='ENOENT')fail('SESSION_NOT_FOUND','Session journal does not exist');throw e;});
  if(st.isSymbolicLink()||!st.isDirectory()||(st.mode&0o077))fail('UNSAFE_JOURNAL','Insecure session directory');
  const fp=join(d,'snapshot.json');const fs=await lstat(fp);
  if(fs.isSymbolicLink()||!fs.isFile()||(fs.mode&0o077)||fs.size>1024*1024)fail('UNSAFE_JOURNAL','Invalid snapshot file');
  const parsed=JSON.parse(await readFile(fp,'utf8'));
  if(parsed?.schema!=='voder.session.journal.v1'||parsed?.session_id!==id||parsed?.workspace!==this.workspace)fail('JOURNAL_WORKSPACE_MISMATCH','Untrusted journal identity/workspace');
  const {sha256,...body}=parsed;
  if(typeof sha256!=='string'||digest(JSON.stringify(body))!==sha256)fail('JOURNAL_INTEGRITY','Journal snapshot was modified');
  if(!Array.isArray(body.events)||body.events.length>10000||body.events.some((e,i)=>e.sequence!==i+1))fail('JOURNAL_INTEGRITY','Event sequence invalid');
  return body;
 }
 async create(id){
  await this.initialize();const d=this._path(id);
  await mkdir(d,{mode:0o700});
  const st=await lstat(d);if(st.isSymbolicLink()||(st.mode&0o077))fail('UNSAFE_JOURNAL','Unsafe session directory');
  const fp=join(d,'snapshot.json');
  const initial={schema:'voder.session.journal.v1',session_id:id,workspace:this.workspace,status:'CREATED',events:[],result:null,started_at:new Date().toISOString(),ended_at:null,abandoned:false};
  // No-clobber session ownership.
  const fd=await open(fp,'wx',0o600);await fd.close();
  await this.write(id,initial);return initial;
 }
 async write(id,body){
  if(body.session_id!==id||body.workspace!==this.workspace)fail('JOURNAL_IDENTITY','Session identity changed');
  if(body.events?.length>10000)fail('JOURNAL_LIMIT','Event limit exceeded; cannot silently truncate');
  const str=JSON.stringify(body);if(Buffer.byteLength(str)>950000)fail('JOURNAL_LIMIT','Journal exceeds limit');
  await durableWrite(join(this._path(id),'snapshot.json'),JSON.stringify({...body,sha256:digest(str)},null,2)+'\n');
 }
 async get(id){return this._read(validateSession(id));}
 async list(){
  await this.initialize();const names=await readdir(join(this.dir,'sessions'));
  const out=[];for(const id of names.filter(x=>UUID.test(x)).sort()){
   try{const r=await this.get(id);out.push({session_id:id,status:r.status,started_at:r.started_at});}catch(e){out.push({session_id:id,status:'JOURNAL_UNREADABLE',error_code:e.code??'ERROR'});}
  }
  return out;
 }
 async events(id,{after=0}={}){
  if(!Number.isSafeInteger(after)||after<0)fail('INVALID_CURSOR','Invalid event cursor');
  const state=await this.get(id);return state.events.filter(e=>e.sequence>after);
 }
 async export(id,output){
  const state=await this.get(id);
  if(typeof output!=='string'||!output.trim())fail('INVALID_EXPORT','Explicit output path required');
  const target=resolve(output);
  // Never overwrite or follow symlinks; the journal contains metadata, never source content/keys.
  const existing=await lstat(target).catch(e=>{if(e.code==='ENOENT')return null;throw e;});
  if(existing)fail('EXPORT_EXISTS','Export destination already exists');
  await assertNoLinkAncestors(dirname(target));
  const st=await lstat(dirname(target));if(st.isSymbolicLink()||!st.isDirectory())fail('UNSAFE_EXPORT','Export parent must be a real directory');
  const fd=await open(target,'wx',0o600);
  try{await fd.writeFile(JSON.stringify({...state,exported_at:new Date().toISOString(),authority:'SESSION_METADATA_ONLY_NOT_VGR_PROOF'},null,2)+'\n');await fd.sync();}
  finally{await fd.close();}
  return {path:target,status:'EXPORTED_UNVERIFIED_SESSION_METADATA',sha256:digest(await readFile(target))};
 }
}

export class VoderControlPlane {
 constructor({sdk,journal,nativeContinuation}={}){
  if(!sdk||!journal)fail('INVALID_HOST','SDK and private journal required');
  if(nativeContinuation!==undefined&&!(nativeContinuation instanceof NativeSdkContinuation))fail('INVALID_RECOVERY_AUTHORITY','Recovery must be owned by the independent Rust adapter');
  this.sdk=sdk;this.journal=journal;this.nativeContinuation=nativeContinuation;
  this.running=new Map();this.maintenance=new Set();this.records=new Map();this.writes=new Map();this.liveOutputs=new Map();this.eventPrefixes=new Map();this.checkpoints=new Map();this.journalErrors=new Map();
 }
 _installDispatchFence(id){
  this.sdk.setNativeDispatchHooks(id,{
   before:async input=>{
    if(this.journalErrors.has(id))fail('JOURNAL_CHECKPOINT_FAILED','Durable event checkpoint failed; native dispatch is disabled');
    const rec=this.records.get(id);if(!rec)fail('RECOVERY_NO_JOURNAL','No durable journal bound to native request');
    if(rec.native_actions?.some(a=>a.action_id===input.action_id||a.idempotency_digest===digest(input.idempotency_key)))fail('RECOVERY_DUPLICATE','Reused native action identity');
    rec.native_actions??=[];
    if(rec.native_actions.length>=64)fail('RECOVERY_ACTION_LIMIT','Too many native actions');
    rec.native_actions.push({schema:'voder.dispatch.intent.v1',session_id:id,action_id:input.action_id,run_id:`run-${input.action_id}`,idempotency_digest:digest(input.idempotency_key),status:'DISPATCH_INTENT'});
    // No gateway spawn or mutation is possible until the journal is fsynced.
    await this._flush(id);
    if(this.journalErrors.has(id))fail('JOURNAL_CHECKPOINT_FAILED','A concurrent durable checkpoint failed; native dispatch is disabled');
   },
   after:async(input,receipt)=>{
    const rec=this.records.get(id);const a=rec?.native_actions?.find(x=>x.action_id===input.action_id);
    if(!a)fail('RECOVERY_INTENT_MISSING','Native action has no durable intent');
    a.status=receipt.status==='COMMITTED'?'VERIFIED_NATIVE_COMMIT':'NATIVE_NONCOMMIT';
    await this._flush(id);
   }
  });
 }
 async createSession({memoryFiles=[],enableMemoryProposals=false,autoContinue=false}={}){
  if(typeof autoContinue!=='boolean')fail('RECOVERY_POLICY','Explicit boolean autoContinue required');
  if(autoContinue && (memoryFiles.length||enableMemoryProposals||this.sdk.toolRegistry||this.sdk.skillStore))
    fail('RECOVERY_CONTEXT_UNSUPPORTED','Automatic recovery cannot silently omit imported memory, enabled proposals or host-installed tools');
  await this.journal.initialize();const id=randomUUID();
  await this.journal.create(id);
  this.sdk.createSession({id,memoryFiles,enableMemoryProposals});
  this.records.set(id,await this.journal.get(id));this._installDispatchFence(id);
  if(autoContinue){const rec=this.records.get(id);rec.auto_continue=true;await this._flush(id);}
  return {sessionId:id,status:'CREATED'};
 }
 async submitTask(id,task){
  validateSession(id);if(this.running.has(id)||this.maintenance.has(id))fail('SESSION_BUSY','Task is already running');
  const current=this.records.get(id);if(!current)fail('SESSION_NOT_LIVE','Historical sessions cannot submit');
  if(!['CREATED','WAITING_FOR_INPUT','BLOCKED'].includes(current.status))fail('SESSION_STATE','Cannot re-submit settled session');
  if(current.auto_continue){
   // Do not save the goal, only its identity. A new process needs trusted taskResolver.
   current.recovery={schema:'voder.sdk.recovery.v1',task_sha256:taskDigest(task),restart_count:0};
   current.owner=await ownerIdentity();
  }
  current.status='INSPECTING';await this._flush(id);
  return this._launch(id,task);
 }
 async _launch(id,task,{oldEvents=[],continueExisting=false}={}){
  const current=this.records.get(id);
  if(!this.eventPrefixes.has(id))this.eventPrefixes.set(id,[...oldEvents]);
  const collect=()=>{const prefix=this.eventPrefixes.get(id)??[];current.events=[...prefix,...this.sdk.getEvents(id).map(event=>({...eventProjection(event),sequence:prefix.length+event.sequence}))];};
  // Metadata events only, coalesced while a durable write is pending. Never checkpoint token deltas.
  let dirty=false,off;
  const checkpoint=()=>{dirty=true;if(this.checkpoints.has(id))return;
   const work=Promise.resolve().then(async()=>{while(dirty){dirty=false;collect();await this._flush(id);}}).catch(error=>{this.journalErrors.set(id,error?.code??'ERROR');void this.sdk.cancel(id).catch(()=>{});}).finally(()=>{this.checkpoints.delete(id);});
   this.checkpoints.set(id,work);
  };
  off=this.sdk.subscribeEvents(id,checkpoint);
  const promise=Promise.resolve().then(async()=>{
   try{const result=await (continueExisting?this.sdk.continueTask(id):this.sdk.submitTask(id,task));current.result=safeSummary(result);current.status=result.status;
    const output=userVisibleResult(result,current.result);if(output)this.liveOutputs.set(id,output);else this.liveOutputs.delete(id);}
   catch(e){this.liveOutputs.delete(id);current.status='FAILED';current.result={status:'FAILED',error_code:e?.code??'ERROR'};}
   finally{
    off?.();await this.checkpoints.get(id);collect();
    current.ended_at=new Date().toISOString();current.owner=null;
    const annotateJournalError=()=>{if(this.journalErrors.has(id)){const code=this.journalErrors.get(id);current.write_error=code;current.result={...current.result,journal_error_code:code};if(!current.result?.verified&&current.status!=='FAILED'){current.status='JOURNAL_FAILED';current.result.status='JOURNAL_FAILED';}}};
    annotateJournalError();
    await this._flush(id).catch(e=>{this.journalErrors.set(id,e?.code??'ERROR');});
    annotateJournalError();
    this.running.delete(id);
   }
   return current.result;
  });this.running.set(id,promise);
  return {sessionId:id,status:'ACCEPTED',note:'Accepted is not VGR commitment or delivery'};
 }
 /** After supervisor restart: never replay prior native action; only re-plan
  * after Rust has authoritatively recorded rollback of EVERY prior action. */
 async autoRecover({taskResolver,sessionId}={}){
  if(sessionId!==undefined)validateSession(sessionId);
  if(typeof taskResolver!=='function')fail('RECOVERY_TASK_RESOLVER','Trusted taskResolver is required; no goal is journaled');
  const candidates=await this.journal.list(),out=[];
  for(const candidate of candidates){
   if(candidate.status==='INSPECTING'&&(sessionId===undefined||candidate.session_id===sessionId)){
    const r=await this._recoverOne(candidate.session_id,taskResolver).catch(e=>({sessionId:candidate.session_id,status:'RECOVERY_BLOCKED',reason:e?.code??'ERROR'}));
    out.push(r);
   }
  }
  if(sessionId!==undefined&&!out.length)fail('RECOVERY_NOT_INTERRUPTED','Target session is not an interrupted task');
  return out;
 }
 async _recoverOne(id,taskResolver){
  if(this.running.has(id)||this.records.has(id))return {sessionId:id,status:'RECOVERY_BLOCKED',reason:'SESSION_LIVE'};
  const lock=join(this.journal._path(id),'recovery.lock');
  try{await mkdir(lock,{mode:0o700});}catch(e){return {sessionId:id,status:'RECOVERY_BLOCKED',reason:'RECOVERY_LOCK_HELD'};}
  try{
   let rec=await this.journal.get(id);
   if(rec.status!=='INSPECTING'||rec.auto_continue!==true||rec.recovery?.schema!=='voder.sdk.recovery.v1'||rec.recovery.restart_count!==0)
     return {sessionId:id,status:'RECOVERY_BLOCKED',reason:'NO_AUTHORIZED_CHECKPOINT'};
   if(rec.cancel_requested)return {sessionId:id,status:'RECOVERY_BLOCKED',reason:'USER_CANCELLATION_PRECEDES_CRASH'};
   if(await oldOwnerAlive(rec.owner))return {sessionId:id,status:'RECOVERY_BLOCKED',reason:'ORIGINAL_PROCESS_STILL_ALIVE'};
   if(!Array.isArray(rec.native_actions??[])||rec.native_actions?.some(a=>a.schema!=='voder.dispatch.intent.v1'||a.session_id!==id||a.run_id!==`run-${a.action_id}`))
     return {sessionId:id,status:'RECOVERY_BLOCKED',reason:'INVALID_ACTION_LOG'};
   const old=rec.native_actions??[];
   if(old.length&&!this.nativeContinuation)return {sessionId:id,status:'RECOVERY_BLOCKED',reason:'NO_RUST_RECOVERY_ADAPTER'};
   for(const action of old){
    // Any unreadable, ambiguous, committed, or active Rust outcome blocks auto-resubmission.
    const verdict=await this.nativeContinuation.reconcile(action);
    if(verdict?.status!=='ROLLED_BACK'||verdict.authority!=='VODER_NATIVE_VGR_RECOVERY'||verdict.action_id!==action.action_id||verdict.delivery_allowed!==false){
     rec.status='RECOVERY_REVIEW_REQUIRED';rec.result={status:'RECOVERY_REVIEW_REQUIRED',reason:verdict.status==='COMMITTED'?'NATIVE_ACTION_COMMITTED_NO_AUTOREPLAY':'RUST_OUTCOME_UNKNOWN'};
     await this.journal.write(id,rec);
     return {sessionId:id,status:rec.status,reason:rec.result.reason};
    }
   }
   // Task cannot be replayed from untrusted journal text: the host supplies
   // an exact task (possibly from encrypted vault) and a matching SHA256.
   const original=await taskResolver(id);
   if(taskDigest(original)!==rec.recovery.task_sha256)return {sessionId:id,status:'RECOVERY_BLOCKED',reason:'TASK_IDENTITY_MISMATCH'};
   const remaining=(original.maxActions??12)-old.length;
   if(remaining<1)return {sessionId:id,status:'RECOVERY_BLOCKED',reason:'ACTION_BUDGET_EXHAUSTED'};
   // Previously attempted native actions still count against the approved task budget.
   const resumedTask={...recoveryTask(original),maxActions:remaining};
   this.sdk.createSession({id});this.records.set(id,rec);this._installDispatchFence(id);
   rec.recovery.restart_count=1;rec.owner=await ownerIdentity();rec.status='INSPECTING';
   await this._flush(id);
   await this._launch(id,resumedTask,{oldEvents:rec.events});
   return {sessionId:id,status:'AUTO_REPLANNING',prior_actions:old.length,authority:old.length?'RUST_ROLLBACK_VERIFIED':'NO_NATIVE_DISPATCH_INTENT'};
  } finally {await rmdir(lock).catch(()=>{});}
 }
 async _flush(id){
  const prev=this.writes.get(id)??Promise.resolve();
  const next=prev.catch(()=>{}).then(()=>this.journal.write(id,this.records.get(id)));
  this.writes.set(id,next);return next;
 }
 async wait(id){validateSession(id);if(!this.running.has(id)){
   if(this.records.has(id))return {sessionId:id,...(this.records.get(id).result??{status:this.records.get(id).status}),...(this.liveOutputs.get(id)??{})};
   const rec=await this.journal.get(id);return {sessionId:id,status:rec.status,result:rec.result};
  }
  const summary=await this.running.get(id);return {sessionId:id,...summary,...(this.liveOutputs.get(id)??{})};
 }
 async status(id){
  validateSession(id);let record=this.records.get(id);
  if(!record)return {...await this.journal.get(id),events:undefined};
  return {session_id:id,status:this.running.has(id)?this.sdk.getStatus(id):record.status,started_at:record.started_at,ended_at:record.ended_at,result:record.result};
 }
 async events(id,{after=0}={}){
  validateSession(id);
  if(!Number.isSafeInteger(after)||after<0)fail('INVALID_CURSOR','Invalid event cursor');
  if(this.records.has(id)){const record=this.records.get(id);const prior=record.events??[];
   if(this.running.has(id)){const prefix=this.eventPrefixes.get(id)??[];const live=this.sdk.getEvents(id).map(e=>({...eventProjection(e),sequence:prefix.length+e.sequence}));return [...prefix,...live].filter(e=>e.sequence>after);}
   return prior.filter(e=>e.sequence>after);}
  return this.journal.events(id,{after});
 }
 subscribe(id,callback,{after=0}={}){
  validateSession(id);if(!this.records.has(id))fail('SESSION_NOT_LIVE','Only a live session can be subscribed to');
  return this.sdk.subscribeEvents(id,event=>callback(eventProjection(event)),{after});
 }
 async cancel(id){
  validateSession(id);
  if(!this.records.has(id))fail('SESSION_NOT_LIVE','Historical session cannot be cancelled');
  if(!this.running.has(id))fail('SESSION_NOT_ACTIVE','No active task');
  const rec=this.records.get(id);rec.cancel_requested=true;await this._flush(id);
  await this.sdk.cancel(id);return {sessionId:id,status:'CANCELLATION_REQUESTED',nativeOutcome:'REQUIRES_RECONCILIATION_IF_IN_FLIGHT'};
 }
 async resume(id){
  validateSession(id);
  const rec=await this.journal.get(id);
  return {sessionId:id,status:rec.status,mode:'READ_ONLY_REATTACH',
    canContinue: false,reason:'Native VGR reconciliation required before resuming interrupted side effects',
    events:rec.events,result:rec.result};
 }
 async reconcileBatchDelivery(id,actionId){
  validateSession(id);if(this.running.has(id)||this.maintenance.has(id))fail('SESSION_BUSY','Cannot reconcile a live writer');
  const rec=await this.journal.get(id);if(!rec.native_actions?.some(a=>a.session_id===id&&a.action_id===actionId&&a.status==='VERIFIED_NATIVE_COMMIT'))fail('BATCH_RECOVERY_IDENTITY','Action must belong to this journaled session');
  const result=await this.sdk.reconcileBatchDelivery(id,actionId);
  if(['ROLLED_BACK_DELIVERY','ALREADY_RESTORED'].includes(result.status)){rec.status='VERIFIED_NOT_DELIVERED';rec.result={...rec.result,status:rec.status,delivery_status:'ROLLED_BACK_DELIVERY'};await this.journal.write(id,rec);if(this.records.has(id))this.records.set(id,rec);}
  return result;
 }
 async continueSession(id){
  validateSession(id);const rec=this.records.get(id);if(this.running.has(id)||this.maintenance.has(id)||!['BLOCKED','DRAFT_UNVERIFIED'].includes(rec?.status))fail('SESSION_CONTINUE_DENIED','Only an idle blocked live task can continue');
  rec.status='INSPECTING';await this._flush(id);return this._launch(id,undefined,{continueExisting:true});
 }
 async _maintain(id,kind,options){
  validateSession(id);if(this.running.has(id)||this.maintenance.has(id)||!this.records.has(id))fail('SESSION_BUSY','Live idle session required');
  this.maintenance.add(id);try{const result=await this.sdk[kind](id,options);const rec=this.records.get(id);if(kind==='switchModel'&&result?.provider&&result?.id)rec.model_selection={provider:result.provider,id:result.id,model_epoch:result.model_epoch};const prefix=this.eventPrefixes.get(id)??[];rec.events=[...prefix,...this.sdk.getEvents(id).map(e=>({...eventProjection(e),sequence:prefix.length+e.sequence}))];await this._flush(id);return result;}finally{this.maintenance.delete(id);}
 }
 compactSession(id,options){return this._maintain(id,'compactSession',options);}
 switchModel(id,options){return this._maintain(id,'switchModel',options);}
 listModels(id){return this._maintain(id,'listModels');}
 async getDiff(id){
  validateSession(id);
  return readNativeSessionDiff({journal:this.journal,id,
    attemptStore:this.sdk.verifiedDelivery?.attemptStore,policyFile:this.sdk.verifiedDelivery?.policyFile,commitVerifier:this.sdk.commitVerifier});
 }
 async export(id,output){return this.journal.export(id,output);}
 async memoryProposals(id){validateSession(id);return this.sdk.getMemoryProposals(id);}
 async approveMemory(id,proposalId,options){validateSession(id);return this.sdk.approveMemoryProposal(id,proposalId,options);}
 async shutdown(){for(const id of this.running.keys())await this.sdk.cancel(id).catch(()=>{});await Promise.allSettled([...this.running.values()]);this.sdk.dispose();}
 dispose(){this.liveOutputs.clear();this.sdk.dispose();}
}
