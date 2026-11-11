'use strict';
const {referenceTask}=require('./reference-transport.cjs');
/** Isolated batch workflow; never used by the existing single-file ChatRunService. */
const {validateGoal,safeUpdate,safeResult}=require('./chat-core.cjs');
const {validateExtended:validateProject}=require('./multi-workspace.cjs');
const UUID=/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
class MultiRunService {
 constructor(open,storageRoot){this.storageRoot=storageRoot;if(typeof open!=='function')throw Error('Open SDK function required');this.open=open;this.active=new Map();}
 async start(folder,goal,notify=()=>{},{project,onCreated=async()=>{},referenceContext=[]}={}){
  validateGoal(goal);
  if(!project || project.root!==folder?.uri?.fsPath)throw Error('WEB_PROJECT: scope mismatch');
  await validateProject(project.recordPath,this.storageRoot);
  const TARGETS=project.targets;const nativeProfile=project.profile,readOnly=false,requiredFiles=[...TARGETS];
  const guard='\n\nHOST-OWNED NEW WEB PROJECT: The host has created blank Git-tracked preimages in this isolated directory. Implement the requested web animation by replacing ALL APPROVED files listed in requiredFiles in ONE batch candidate {patches:[{relative_path,old_sha256,new_content},...]}, sorted by relative_path. Read the current source SHA values. Use voder_verify_candidate, then voder_commit_verified_candidate for the exact passing batch. Do not submit files individually. verify_web.py is the immutable independent structural validator: it requires a full HTML5 document, a title, exactly one external app.js script, one style.css link, a canvas or inline SVG, and requestAnimationFrame in app.js. Use no network, dependencies, inline scripts, inline event handlers, frames, forms, or new files. Core files must contain at least 32 UTF-8 bytes; auxiliary files must be nonempty. No fixed whole-file byte ceiling. Use paginated reads and hash-pinned exact edits for large files. Design the whole scene in native Canvas/SVG/CSS code; do not read another project. A sandbox PASS certifies structure only, NOT visual correctness or JavaScript execution. Clearly disclose that limit. Auxiliary files may use local modules, JSON or Python source; no Python program runs in the browser. No external dependencies or new paths beyond the exact file list. No other paths are authorized. Original user task follows:\n';
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
module.exports={MultiRunService};
