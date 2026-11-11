'use strict';
const {referenceTask}=require('./reference-transport.cjs');
const {validateGoal,safeUpdate,safeResult}=require('./chat-core.cjs');
const {responsePresentation}=require('./response-presentation.cjs');
const UUID=/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
class GeneralRunService {
 constructor(open){this.open=open;this.active=new Map();}
 async start(folder,goal,notify=()=>{},{launch,onCreated=async()=>{},referenceContext=[],conversationContext="",requestWriteApproval=async()=>false}={}){
  validateGoal(goal);const key=folder?.uri?.fsPath;
  if(!key||launch?.root!==key)throw Error('GENERAL_WORKSPACE_IDENTITY');
  if(this.active.has(key))throw Object.assign(Error('Another task is active'),{code:'CHAT_BUSY'});
  const run={sessionId:null,rpc:null,cancelled:false,settled:false,approvalAbort:new AbortController(),decisions:new Set(),renderFailureCode:null};this.active.set(key,run);let onEvent=()=>{},ackPending=false;
  try{
   const rpc=await this.open(folder,launch);run.rpc=rpc;
   const task=referenceTask(rpc,{goal,maxActions:1},referenceContext); // Original user prompt is unchanged.
   if(conversationContext){if(typeof conversationContext!=='string'||Buffer.byteLength(conversationContext,'utf8')>24576)throw Object.assign(Error('Invalid bounded conversation context'),{code:'CHAT_CONTEXT_BOUNDS'});task.conversationContext=conversationContext;}
   const created=await rpc.request('session.create',{autoContinue:false},12000),id=created?.sessionId;
   if(!UUID.test(id||''))throw Error('GENERAL_SESSION_IDENTITY');
   run.sessionId=id;await onCreated(id);notify({kind:'created',sessionId:id});
   onEvent=p=>{if(p?.sessionId!==id)return;
    const e=p.event;
    if(e?.type==='task.workspace_approval_requested'){
     const valid=e.sessionId===id&&UUID.test(e.requestId||'')&&/^[a-f0-9]{64}$/.test(e.scopeDigest||'')&&e.root===key&&Array.isArray(e.files)&&e.files.length>0&&e.files.length<=16&&e.files.every(f=>typeof f==='string'&&f.length<=240&&!f.startsWith('/')&&!f.split('/').some(x=>!x||x==='.'||x==='..'))&&Array.isArray(e.checks)&&e.checks.length<=8;
     if(!valid){notify({kind:'uncertain',sessionId:id,code:'INVALID_WRITE_APPROVAL_EVENT'});return;}
     if(run.decisions.has(e.requestId))return;run.decisions.add(e.requestId);
     void(async()=>{
      let approved=false;try{approved=await requestWriteApproval(structuredClone(e),{signal:run.approvalAbort.signal})===true;}catch{}
      if(run.cancelled||run.settled||this.active.get(key)!==run)approved=false;
      try{await rpc.request('task.authorizeWorkspace',{sessionId:id,requestId:e.requestId,scopeDigest:e.scopeDigest,approved},15000);}catch(error){if(!run.settled)notify({kind:'uncertain',sessionId:id,code:error.code||'WRITE_APPROVAL_OUTCOME_UNKNOWN'});}
     })();return;
    }
    if(e?.type==='model.tool_execution_end'&&e.tool==='voder_task_render_svg'){if(e.status==='RENDERED')run.renderFailureCode=null;else if(['DENIED','FAIL','ERROR'].includes(e.status))run.renderFailureCode=e.code||'SVG_RENDER_FAILED';}
    const event=safeUpdate(e);if(event)notify({kind:'event',sessionId:id,event});};rpc.on('event',onEvent);
   await rpc.request('session.subscribe',{sessionId:id,after:0},12000);
   ackPending=true;const ack=await rpc.request('task.submit',{sessionId:id,task},15000);ackPending=false;
   if(ack?.status!=='ACCEPTED')throw Error('GENERAL_TASK_NOT_ACCEPTED');notify({kind:'accepted',sessionId:id});
   const completion=(async()=>{try{const raw=await rpc.request('task.wait',{sessionId:id},3600*1000),result={...safeResult(raw,id),presentation:responsePresentation(raw,{goal,readOnly:launch.readOnly===true,renderFailureCode:run.renderFailureCode})};run.settled=true;notify({kind:'settled',...result});return result;}catch(e){notify({kind:'uncertain',sessionId:id,code:e.code||'RPC_UNKNOWN'});throw e;}finally{run.settled=true;run.approvalAbort.abort();rpc.off('event',onEvent);if(this.active.get(key)===run)this.active.delete(key);}})();completion.catch(()=>{});return {sessionId:id,completion};
  }catch(e){run.settled=true;run.approvalAbort.abort();if(ackPending&&run.sessionId)notify({kind:'uncertain',sessionId:run.sessionId,code:e.code||'RPC_UNKNOWN'});run.rpc?.off('event',onEvent);if(this.active.get(key)===run)this.active.delete(key);throw e;}
 }
 activeSession(folder){return this.active.get(folder?.uri?.fsPath)?.sessionId??null;}
 async cancel(folder,id){const run=this.active.get(folder?.uri?.fsPath);if(!run||run.sessionId!==id)throw Error('No matching active task');run.cancelled=true;run.approvalAbort.abort();return run.rpc.request('session.cancel',{sessionId:id},15000);}
}
module.exports={GeneralRunService};
