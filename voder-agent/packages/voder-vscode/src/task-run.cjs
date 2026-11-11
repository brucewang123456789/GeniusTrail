'use strict';
const {referenceTask}=require('./reference-transport.cjs');
/** General reviewed task workflow for single-file or batch native delivery. */
const {validateGoal,safeUpdate,safeResult}=require('./chat-core.cjs');
const {validateTask:validateProject}=require('./task-workspace.cjs');
const UUID=/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
class TaskRunService {
 constructor(open,storageRoot){this.storageRoot=storageRoot;if(typeof open!=='function')throw Error('Open SDK function required');this.open=open;this.active=new Map();}
 async start(folder,goal,notify=()=>{},{project,onCreated=async()=>{},referenceContext=[]}={}){
  validateGoal(goal);
  if(!project || project.root!==folder?.uri?.fsPath)throw Error('TASK_PROJECT: scope mismatch');
  await validateProject(project.recordPath,this.storageRoot);
  const TARGETS=project.targets;const nativeProfile=project.profile,readOnly=false,requiredFiles=[...TARGETS];
  const guard='\n\nHOST-APPROVED GENERAL TASK: Work in the SELECTED task directory. The exact approved UTF-8 paths are listed in requiredFiles. Existing source preimages and new-file placeholders are Git tracked. Implement the ORIGINAL user task, without changing its artifact type, filenames, framework or objective to fit a web demo. Read the current files and SHA256 values. Submit a complete sorted candidate through voder_verify_candidate, using {patches:[{relative_path,old_sha256,new_content},...]} for multiple paths, or {relative_path,old_sha256,new_content} for one path. Repair based on real checks before committing the exact passing candidate. Only Rust VGR and verified native delivery may save changes. The immutable host validator parses Python/JSON/XML/SVG and runs only the explicitly reviewed sandbox argv checks. SVG checks do not certify visual quality. No arbitrary HOST shell, no fake renderer results, no invented image inspection or quality claims. Use voder_ext_task_capabilities for the exact capabilities, approved checks and limits. Use voder_ext_task_content_hash on the exact final SVG/source bytes before writing a verification record; never guess a SHA256. Native output is UTF-8 source only; optional PNG/binary render delivery and autonomous visual inspection are not currently supported by this path. Preserve the best editable source, record this specific limitation in the requested verification report and do not claim complete visual acceptance. Do not execute another task or read unrelated directories. No fixed whole-file byte ceiling. Read large files in pages and prefer exact hash-pinned edits, including edit members in a complete sorted batch; do not send the whole source to the model unnecessarily. Never rewrite a solution merely to compensate for an unavailable sandbox tool. Original user task follows:\n';
  const scoped=guard+goal;validateGoal(scoped);
  if(!folder?.uri?.fsPath)throw Object.assign(Error('Choose an explicit project workspace'),{code:'CHAT_WORKSPACE_REQUIRED'});
  const key=folder.uri.fsPath;
  // Synchronous reservation avoids two parallel submissions when a double click races with initialize.
  if(this.active.has(key))throw Object.assign(Error('A Voder chat task is already running in this workspace'),{code:'CHAT_BUSY'});
  const run={folder,sessionId:null,settled:false,rpc:null};this.active.set(key,run);
  let onEvent=()=>{},ackPending=false;
  try{
   const rpc=await this.open(folder,nativeProfile,{readOnly});run.rpc=rpc;
   const task=referenceTask(rpc,{goal:scoped,...(requiredFiles.length?{requiredFiles}:{}),maxActions:1},referenceContext);
   const created=await rpc.request('session.create',{autoContinue:false},12000);
   const id=created?.sessionId;
   if(!UUID.test(id||''))throw Object.assign(Error('SDK returned an invalid session identity'),{code:'CHAT_BAD_SESSION'});
   run.sessionId=id;await onCreated(id);notify({kind:'created',sessionId:id});
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
 async cancel(folder,expectedSessionId){
  const run=this.active.get(folder?.uri?.fsPath);
  if(!run||!run.sessionId||run.sessionId!==expectedSessionId||run.settled)throw Object.assign(Error('No matching active chat task'),{code:'CHAT_NOT_ACTIVE'});
  // Request only: the VGR native outcome may still be in flight or unknown.
  return run.rpc.request('session.cancel',{sessionId:run.sessionId},15000);
 }
 activeSession(folder){return this.active.get(folder?.uri?.fsPath)?.sessionId??null;}
}
module.exports={TaskRunService};
