'use strict';
const {referenceTask}=require('./reference-transport.cjs');
/** UI-only orchestration. Every turn receives a fresh SDK session; native VGR alone controls writes. */
const {classifyStatus}=require('./terminal-mapping.cjs');
const {inspectionGoal}=require('./entry-orchestrator.cjs');
const {planningGoal}=require('./task-plan.cjs');
const UUID=/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
const MAX_GOAL_BYTES=16384;
const EVENT_TYPES=new Set(['task.browser_stage','model.tool_execution_start','model.tool_execution_end','sandbox.candidate_requested','sandbox.candidate_settled','vgr.action_requested','vgr.action_settled','task.settled','task.failed','task.cancel_requested','model.progress_summary','model.request_start','model.message_end','model.image_request','model.vision_selection']);
function validateGoal(goal){
 if(typeof goal!=='string'||!goal.trim()||Buffer.byteLength(goal,'utf8')>MAX_GOAL_BYTES||goal.includes('\0'))throw Object.assign(new Error('Enter a nonempty task (maximum 16 KiB).'),{code:'CHAT_INVALID_GOAL'});
 return goal;
}
function scopedGoal(goal,history){
 validateGoal(goal);
 if(!Array.isArray(history)||!history.length)return goal;
 const previous=history.slice(-3).filter(h=>h&&typeof h.goal==='string'&&typeof h.status==='string'&&h.goal.length<=16384&&/^[A-Z_]{2,64}$/.test(h.status));
 while(previous.length){
  const prefix='[PRIOR VODER CHAT TURNS — CONTEXT ONLY; DO NOT REEXECUTE]\n'+previous.map((h,i)=>`${i+1}. Goal: ${JSON.stringify(h.goal.slice(0,2500))}\n   Result: ${h.status}`).join('\n')+'\n[/PRIOR VODER CHAT TURNS]\n\nCURRENT USER TASK (ONLY THIS TASK IS TO BE EXECUTED):\n';
  const combined=prefix+goal;
  if(Buffer.byteLength(combined,'utf8')<=MAX_GOAL_BYTES)return combined;
  previous.shift();
 }
 return goal;
}
function safeUpdate(e){
 if(!e||!EVENT_TYPES.has(e.type))return null;
 const out={type:e.type};
 if(e.type==='model.progress_summary'&&typeof e.summary==='string'&&e.summary.length>=12&&e.summary.length<=160&&Buffer.byteLength(e.summary,'utf8')<=480&&!/[\u0000-\u001f\u007f<>`\\]/u.test(e.summary)&&!/(?:^|\s)\/|@|:\/\/|\b(?:key|token|secret|password|credential|authorization|bearer|api[_-]?key)\b/iu.test(e.summary))out.summary=e.summary;
 for(const k of ['status','state','tool','code','verdict'])if(typeof e[k]==='string'&&e[k].length<=80&&/^[\w. -]+$/.test(e[k]))out[k]=e[k];
 if(Number.isSafeInteger(e.image_count))out.imageCount=e.image_count;
 if(typeof e.images_sha256==='string'&&/^[a-f0-9]{64}$/.test(e.images_sha256))out.imagesSha256=e.images_sha256;
 if(Number.isSafeInteger(e.sequence)&&e.sequence>=0)out.sequence=e.sequence;
 return out;
}
function safeResult(result,id){
 const state=classifyStatus(result||{}),out={sessionId:id,status:state.status,label:state.label,verified:state.verified};
 if(state.verified&&typeof result?.commit_digest==='string'&&/^[a-f0-9]{64}$/i.test(result.commit_digest))out.commitDigest=result.commit_digest;
 if(result?.task_acceptance?.schema==='voder.task-acceptance.v1')out.taskAcceptance=result.task_acceptance;
 if(result?.status==='DRAFT_UNVERIFIED'&&result.verified===false&&typeof result.draft==='string'&&Buffer.byteLength(result.draft,'utf8')<=131072)out.draft=result.draft;
 if(Array.isArray(result?.missing))out.missing=result.missing.filter(x=>typeof x==='string'&&x.length<=240).slice(0,32);
 if(typeof result?.model_error_code==='string'&&/^[A-Z0-9_]+$/.test(result.model_error_code))out.errorCode=result.model_error_code;
 if(typeof result?.error_code==='string'&&/^REFERENCE_[A-Z_]+$/.test(result.error_code))out.errorCode=result.error_code;
 return out;
}
class ChatRunService {
 constructor(open){if(typeof open!=='function')throw Error('Open SDK function required');this.open=open;this.active=new Map();}
 async start(folder,goal,notify=()=>{},{requiredFiles=[],history=[],nativeProfile=null,operatorTarget=null,readOnly=false,planning=false,referenceContext=[],conversationContext=""}={}){
  if(!Array.isArray(requiredFiles)||requiredFiles.length>32||requiredFiles.some(f=>typeof f!=='string'||!f.trim()||f.length>240||f.startsWith('/')||f.split('/').includes('..')))throw Object.assign(Error('Invalid required file list'),{code:'CHAT_INVALID_REQUIRED_FILES'});
  if(readOnly&&nativeProfile!==null)throw Object.assign(Error('Read-only investigation cannot receive a native profile'),{code:'ENTRY_READ_ONLY_PROFILE'});
  if(planning&&!readOnly)throw Object.assign(Error('Task planning must be read-only preparation'),{code:'TASK_PLAN_POLICY'});
  let scoped=scopedGoal(planning?planningGoal(validateGoal(goal)):readOnly?inspectionGoal(validateGoal(goal),{hasReferences:Array.isArray(referenceContext)&&referenceContext.length>0}):goal,history);
  validateGoal(scoped);
  // Current native Rust delivery requires an existing Git-committed preimage; stop unmistakable new-file requests before paid calls.
  if(nativeProfile!==null&&/(?:新建|创建|新增)(?:一个|一份|新的|个)?(?:.{0,28})?(?:文件|\.py|\.js|\.ts)|(?:create|add|write)\s+(?:a\s+)?new\s+(?:source\s+)?file/i.test(goal))throw Object.assign(Error('Creating new files is not supported by the installed native VGR delivery; choose a modification of an existing tracked file.'),{code:'CHAT_NEW_FILE_UNSUPPORTED'});
  if(nativeProfile!==null){
   if(typeof nativeProfile!=='string'||!nativeProfile.startsWith('/'))throw Object.assign(Error('Native profile required'),{code:'CHAT_NATIVE_PROFILE_REQUIRED'});
   if(typeof operatorTarget!=='string'||!operatorTarget||!requiredFiles.includes(operatorTarget))throw Object.assign(Error('Native operator scope missing'),{code:'CHAT_NATIVE_SCOPE_REQUIRED'});
   const guard='\n\nOPERATOR-PINNED SCOPE: Only propose a replacement for the EXISTING Git-tracked file '+JSON.stringify(operatorTarget)+'. No new files, additional file writes, altered tests, or unapproved paths. Use Voder native candidate verification and request native VGR commit of the exact passing candidate. If the user asks for a new file or anything outside the pinned scope, report BLOCKED instead; do not claim delivery.';
   if(Buffer.byteLength(scoped+guard,'utf8')>MAX_GOAL_BYTES)throw Object.assign(Error('Task exceeds native bounded prompt after scope guard'),{code:'CHAT_GOAL_TOO_LARGE'});
   scoped+=guard;
  }
  if(!folder?.uri?.fsPath)throw Object.assign(Error('Choose an explicit project workspace'),{code:'CHAT_WORKSPACE_REQUIRED'});
  const key=folder.uri.fsPath;
  // Synchronous reservation avoids two parallel submissions when a double click races with initialize.
  if(this.active.has(key))throw Object.assign(Error('A Voder chat task is already running in this workspace'),{code:'CHAT_BUSY'});
  const run={folder,sessionId:null,settled:false,rpc:null};this.active.set(key,run);
  let onEvent=()=>{},ackPending=false;
  try{
   const rpc=await this.open(folder,nativeProfile,{readOnly});run.rpc=rpc;
   const task=referenceTask(rpc,{goal:scoped,...(requiredFiles.length?{requiredFiles}:{}),maxActions:1},referenceContext);
   if(conversationContext){if(typeof conversationContext!=='string'||Buffer.byteLength(conversationContext,'utf8')>24576)throw Object.assign(Error('Invalid bounded conversation context'),{code:'CHAT_CONTEXT_BOUNDS'});task.conversationContext=conversationContext;}
   const created=await rpc.request('session.create',{autoContinue:false},12000);
   const id=created?.sessionId;
   if(!UUID.test(id||''))throw Object.assign(Error('SDK returned an invalid session identity'),{code:'CHAT_BAD_SESSION'});
   run.sessionId=id;notify({kind:'created',sessionId:id});
   onEvent=p=>{if(p?.sessionId!==id)return;const event=safeUpdate(p.event);if(event)notify({kind:'event',sessionId:id,event});};
   rpc.on('event',onEvent);
   await rpc.request('session.subscribe',{sessionId:id,after:0},12000);
   // The UI cannot issue native actions. Only the real SDK task gate can do so.
   ackPending=true;
   const ack=await rpc.request('task.submit',{sessionId:id,task},15000);
   ackPending=false;
   if(ack?.status!=='ACCEPTED')throw Object.assign(Error('Task not accepted by SDK'),{code:'CHAT_NOT_ACCEPTED'});
   notify({kind:'accepted',sessionId:id,status:'ACCEPTED'});
   const completion=(async()=>{
    try{
     const result=await rpc.request('task.wait',{sessionId:id},3000*1000);
     if(readOnly&&result?.verified===true)throw Object.assign(Error('Read-only session returned native verification'),{code:'ENTRY_READ_ONLY_CONTRACT'});
     const status={...safeResult(result,id),...(readOnly?{phase:'inspect'}:{})};run.settled=true;notify({kind:'settled',...status});return status;
    }catch(e){notify({kind:'uncertain',sessionId:id,code:String(e?.code||'RPC_UNKNOWN').slice(0,64)});throw e;}
    finally{rpc.off('event',onEvent);if(this.active.get(key)===run)this.active.delete(key);}
   })();
   // A completion is always observed by the caller; attaching a handler prevents late unhandled rejections.
   completion.catch(()=>{});
   return {sessionId:id,completion};
  }catch(e){if(ackPending&&run.sessionId)notify({kind:'uncertain',sessionId:run.sessionId,code:String(e?.code||'RPC_UNKNOWN').slice(0,64)});if(run.rpc)run.rpc.off('event',onEvent);if(this.active.get(key)===run)this.active.delete(key);throw e;}
 }
 async inspect(folder,goal,notify=()=>{},{history=[],referenceContext=[]}={}){return this.start(folder,goal,notify,{history,referenceContext,readOnly:true});}
 async cancel(folder,expectedSessionId){
  const run=this.active.get(folder?.uri?.fsPath);
  if(!run||!run.sessionId||run.sessionId!==expectedSessionId||run.settled)throw Object.assign(Error('No matching active chat task'),{code:'CHAT_NOT_ACTIVE'});
  // Request only: the VGR native outcome may still be in flight or unknown.
  return run.rpc.request('session.cancel',{sessionId:run.sessionId},15000);
 }
 activeSession(folder){return this.active.get(folder?.uri?.fsPath)?.sessionId??null;}
}
module.exports={ChatRunService,validateGoal,scopedGoal,safeUpdate,safeResult};
