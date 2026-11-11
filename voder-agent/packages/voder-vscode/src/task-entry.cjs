'use strict';
const vscode=require('vscode'),fs=require('node:fs/promises'),path=require('node:path');
const {parsePlanWithDiagnostics}=require('./task-plan.cjs');
const {planTask,createTask,validateTask,hash}=require('./task-workspace.cjs');
const {TaskRunService}=require('./task-run.cjs');
const {nativeEnv}=require('./native-env.cjs');
const {validateVerifiedDiff}=require('./verified-diff.cjs');
class TaskEntry {
 constructor(app){this.app=app;this.storage=app.context.globalStorageUri.fsPath;this.records=new Map();this.reserved=new Set();this.runs=new TaskRunService(async(folder,profile)=>app._open(folder,{mode:'native',gateway:profile,generalTask:true,nodeExecutable:JSON.parse(await fs.readFile(profile,'utf8')).nodeBinary}),this.storage);}
 async start(provider,view,msg,folder){
  const app=this.app,root=folder.uri.fsPath;
  if(this.reserved.has(root)||this.runs.activeSession(folder)||app.chatRuns.activeSession(folder))throw Object.assign(Error('Another task is running in this directory.'),{code:'CHAT_BUSY'});
  this.reserved.add(root);const reservedRoots=[root];let executionStarted=false;
  const track=id=>{app._trackChatSession(id,folder);const p=provider.pending.get(view);if(p){p.folder=folder;p.sessionId=id;}};
  try{
   const agentRoot=await app._agentRootForChat(folder);await app.context.globalState.update('voder.chat.agentRoot',agentRoot);
   provider.post(view,{type:'phase_started',phase:'plan'});
   const prep=await app.chatRuns.start(folder,msg.goal,update=>{
    if(['created','accepted'].includes(update.kind))track(update.sessionId);
    if(update.kind!=='settled')provider.post(view,{...update,type:update.kind,phase:'plan'});
   },{readOnly:true,planning:true,...(msg.referenceContext?.length?{referenceContext:msg.referenceContext}:{})});
   const proposal=await parsePlanWithDiagnostics(await prep.completion,{storageRoot:this.storage,workspace:root,goal:msg.goal});
   if(!provider.views.has(view))return;
   if(proposal.action==='answer'){
    provider.post(view,{type:'inspection_complete',answer:proposal.answer,scope:null,continuing:false,status:'INSPECTED'});return;
   }
   provider.post(view,{type:'task_planned',answer:proposal.answer,files:proposal.files});
   nativeEnv(process.env);
   const plan=await planTask({agentRoot,storageRoot:this.storage,root,scope:proposal});
   if(plan.root!==root){
    const actual={uri:vscode.Uri.file(plan.root),name:path.basename(plan.root)};
    if(this.reserved.has(plan.root)||this.runs.activeSession(actual)||app.chatRuns.activeSession(actual))throw Object.assign(Error('Another task is running in the actual target repository.'),{code:'CHAT_BUSY'});
    this.reserved.add(plan.root);reservedRoots.push(plan.root);
   }
   const checkText=plan.scope.checks.length?plan.scope.checks.map(x=>JSON.stringify(x)).join('\n'):'Text/syntax validation only; runtime/visual checks not selected.';
   const approved=await provider.requestNativeApproval(view,{files:plan.scope.files,operation:'Prepare selected project files under the requested write scope.',checksSummary:'Independent sandbox checks: '+plan.scope.checks.length+' (maximum 20 seconds each). Independent authorization remains required.',target:plan.root+'\n'+plan.scope.files.join('\n'),validator:'Create only selected missing source preimages, initialize Git if needed and snapshot the current selected files (including any existing edits) plus a host validator into a Git baseline commit. Unrelated files and staged changes are excluded. Sandbox commands (20s each):\n'+checkText+'\nRust VGR remains required for final writes. Binary delivery and visual inspection are unavailable.'});
   if(!approved){provider.post(view,{type:'not_started',goal:msg.goal,message:'Task scope declined. No files or execution were created.'});return;}
   if(!provider.views.has(view))return;
   const executionFolder={uri:vscode.Uri.file(plan.root),name:path.basename(plan.root)};
   const project=await createTask(plan);
   provider.post(view,{type:'workspace_selected',path:project.root});
   const current=provider.pending.get(view);if(current)current.folder=executionFolder;
   if(!provider.views.has(view))return;
   provider.post(view,{type:'phase_started',phase:'native'});
   executionStarted=true;
   const running=await this.runs.start(executionFolder,msg.goal,update=>{
    if(['created','accepted'].includes(update.kind)){
     const id=update.sessionId;this.records.set(id,project);app._trackChatSession(id,executionFolder,project.profile);const p=provider.pending.get(view);if(p){p.folder=executionFolder;p.sessionId=id;}
    }
    provider.post(view,{...update,type:update.kind});
   },{project,...(msg.referenceContext?.length?{referenceContext:msg.referenceContext}:{}),onCreated:async id=>{const records={...(app.context.globalState.get('voder.tasks',{})||{}),[id]:project.recordPath};await app.context.globalState.update('voder.tasks',records);}});
   const result=await running.completion;
   if(result?.status==='DELIVERED'&&result.verified===true)provider.post(view,{type:'task_delivered',sessionId:running.sessionId,path:project.root,files:project.targets});
  }catch(error){
   if(executionStarted)throw error;
   provider.post(view,{type:'not_started',status:'START_BLOCKED',goal:msg.goal,code:String(error.code||'TASK_PREPARATION_FAILED'),message:'Task preparation blocked: '+String(error.message||error).slice(0,500)});
  }finally{for(const reserved of reservedRoots)this.reserved.delete(reserved);}
 }
 async checked(id){const p=this.records.get(id)?.recordPath||this.app.context.globalState.get('voder.tasks',{})?.[id];return p?validateTask(p,this.storage):null;}
 async action(id,command){
  const project=await this.checked(id);if(!project)return false;
  if(command==='open-project'){await vscode.commands.executeCommand('vscode.openFolder',vscode.Uri.file(project.root),{forceNewWindow:true});return true;}
  if(command!=='preview')return false;
  const folder={uri:vscode.Uri.file(project.root),name:path.basename(project.root)};
  const rpc=await this.app._open(folder,{mode:'native',gateway:project.profile,generalTask:true,nodeExecutable:project.nodeBinary});
  if(!rpc.diffCapable)throw Error('Verified diff capability required before opening artifacts');
  const files=validateVerifiedDiff(await rpc.request('session.diff',{sessionId:id},30000),id);
  if(JSON.stringify(files.map(x=>x.relative_path))!==JSON.stringify(project.targets))throw Error('Delivered target list differs');
  for(const f of files)if(hash(await fs.readFile(path.join(project.root,f.relative_path)))!==f.after_sha256)throw Error('Artifact changed since delivery');
  const artifact=project.targets.find(f=>f.endsWith('.svg'))||project.targets.find(f=>f.endsWith('.html'))||project.targets[0];
  // Open the verified file in the editor. No generated script runs in the extension host.
  await vscode.commands.executeCommand('vscode.open',vscode.Uri.file(path.join(project.root,artifact)));return true;
 }
}
module.exports={TaskEntry};
