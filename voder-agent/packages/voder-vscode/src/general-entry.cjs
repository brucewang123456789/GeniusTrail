'use strict';
const vscode=require('vscode'),fs=require('node:fs/promises'),path=require('node:path'),crypto=require('node:crypto');
const {launchGeneral,readLaunch}=require('./general-workspace.cjs');
const {GeneralRunService}=require('./general-run.cjs');
const {ordinaryWorkspace}=require('./ordinary-workspace.cjs');
const {publishOriginalProject}=require('./original-project-publication.cjs');
const {readOnlyRequest,writeRequest}=require('./entry-orchestrator.cjs');
const {planTask}=require('./task-workspace.cjs');
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
const escapeHtml=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const imageTypes={'.svg':'image/svg+xml','.png':'image/png','.jpg':'image/jpeg','.jpeg':'image/jpeg','.gif':'image/gif','.webp':'image/webp','.avif':'image/avif'};
function previewHtml({content,base,source,label,image}){
 // Generated scripts run only in an opaque-origin frame, without a VS Code bridge.
 const csp=`default-src 'none'; script-src 'unsafe-inline' ${source}; style-src 'unsafe-inline' ${source}; img-src data: blob: ${source}; font-src data: ${source}; connect-src ${source}; frame-src 'self'; object-src 'none'; base-uri ${source}; form-action 'none'; worker-src 'none';`;
 let body;
 if(image)body=`<img class="image" alt="${escapeHtml(label)}" src="${escapeHtml(image)}">`;
 else{
  const tag=`<base href="${escapeHtml(base)}">`;
  const html=/<head\b[^>]*>/i.test(content)?content.replace(/<head\b[^>]*>/i,m=>m+tag):content.replace(/<html\b[^>]*>/i,m=>m+'<head>'+tag+'</head>');
  body=`<iframe sandbox="allow-scripts" title="${escapeHtml(label)}" srcdoc="${escapeHtml(html)}"></iframe>`;
 }
 return `<!DOCTYPE html><html><head><meta charset="utf-8"><meta http-equiv="Content-Security-Policy" content="${escapeHtml(csp)}"><style>html,body{margin:0;height:100%;overflow:hidden;background:#fff;color:#555;font:12px system-ui}body{display:flex;flex-direction:column}.note{padding:5px 10px;background:#f3f3f3;flex:none}iframe{width:100%;flex:1;border:0;min-height:0}.image{width:100%;height:calc(100% - 28px);object-fit:contain}</style></head><body><div class="note">${escapeHtml(label)}</div>${body}</body></html>`;
}
class TaskEntry {
 constructor(app){this.app=app;this.storage=app.context.globalStorageUri.fsPath;this.reserved=new Set();this.previews=new Map();this.runs=new GeneralRunService((folder,launch)=>app._open(folder,{mode:'native',gateway:launch.profile,generalTask:true,nodeExecutable:launch.nodeBinary}));}
 async start(provider,view,msg,folder){
  const app=this.app,selectedRoot=folder.uri.fsPath;let root=selectedRoot;
  if(this.reserved.has(root)||this.runs.activeSession(folder)||app.chatRuns.activeSession?.(folder))throw Object.assign(Error('Another task is active'),{code:'CHAT_BUSY'});
  this.reserved.add(root);let executionStarted=false;
  try{
   const agentRoot=await app._agentRootForChat(folder);await app.context.globalState.update('voder.chat.agentRoot',agentRoot);
    // A normal interactive task must start from an explicitly selected private project.
    // Reject shared-writable paths before any copying, host preparation, or model call.
    const ordinaryProjectStat=await require('node:fs/promises').lstat(root);
    if(!ordinaryProjectStat.isDirectory()||ordinaryProjectStat.isSymbolicLink())
      throw Object.assign(Error('Open a project folder before using Voder.'),{code:'ORDINARY_WORKSPACE_INVALID_DIRECTORY'});
   const location=await ordinaryWorkspace({root,storageRoot:this.storage,readOnly:readOnlyRequest(msg.goal)});
   if(location.isolated){root=location.root;folder={...folder,name:folder.name+' (private task copy)',uri:vscode.Uri.file(root)};provider.post(view,{type:'workspace_selected',path:selectedRoot});provider.post(view,{type:'workspace_isolated'});}
   if(!readOnlyRequest(msg.goal)&&writeRequest(msg.goal))await planTask({root,agentRoot,storageRoot:this.storage,scope:{action:'execute',answer:'Host preflight only; no file preparation or model request',files:['voder-entry-preflight-'+crypto.randomUUID()+'.txt'],checks:[]}});
   const launch=await launchGeneral({root,agentRoot,storageRoot:this.storage,memoryRoot:selectedRoot,approvalMode:'interactive',readOnly:readOnlyRequest(msg.goal)});
   if(!provider.views.has(view))return;
   provider.post(view,{type:'phase_started',phase:'respond'});executionStarted=true;
   const conversationContext=require('./dialogue-model-tool.cjs').boundContext(await provider.memoryContext?.(view,selectedRoot,msg.goal)??'');
   const running=await this.runs.start(folder,msg.goal,u=>{
    if(['created','accepted'].includes(u.kind)){app._trackChatSession(u.sessionId,folder);const p=provider.pending.get(view);if(p){p.folder=folder;p.sessionId=u.sessionId;}}
    // A private-copy Rust commit is not yet delivery to the user's selected project.
    if(u.kind==='settled'&&location.isolated&&u.status==='DELIVERED'&&u.verified===true){provider.post(view,{type:'publication_pending'});return;}
    const {taskAcceptance,commitDigest,...publicUpdate}=u;provider.post(view,{...publicUpdate,type:u.kind});
   },{launch,referenceContext:msg.referenceContext||[],conversationContext,requestWriteApproval:async(scope,{signal})=>{
    if(!provider.views.has(view)||signal.aborted)return false;
    return provider.requestNativeApproval(view,{files:scope.files,operation:'Prepare the selected files. Other project source files require separate approval.',checksSummary:'Independent sandbox checks: '+scope.checks.length+'. Configured checks do not certify appearance or interaction.',target:scope.root+'\n'+scope.files.join('\n'),validator:'Allow preparation and checked delivery for these files only. New files initially receive placeholder content; preparation also records a Git baseline and adds a validation script. Additional source files require another approval. Independent checks:\n'+scope.checks.map(x=>JSON.stringify(x)).join('\n')},{signal});
   },onCreated:async id=>{const records={...(app.context.globalState.get('voder.generalTasks',{})||{}),[id]:{root,profile:launch.profile,artifacts:[]}};await app.context.globalState.update('voder.generalTasks',records);}});
   let result=await running.completion;
   if(location.isolated&&result.status==='DELIVERED'&&result.verified===true){
    try{
     const published=await publishOriginalProject({originalRoot:selectedRoot,privateRoot:root,storageRoot:this.storage,sessionId:running.sessionId,result,
      approve:scope=>provider.views.has(view)&&provider.requestNativeApproval(view,{publication:true,files:scope.files,operation:'Deliver verified files to the selected project only if the originals are unchanged.',checksSummary:'Only previously verified and unchanged files may be delivered.',target:scope.root+'\n'+scope.files.join('\n'),validator:'Publish only these Rust-verified files to your original selected project. Existing files must match the original private-copy baseline. Exact SHA-256 values: '+scope.digests.map(d=>d.path+' '+d.sha256).join('; ')})});
     result={...result,projectDelivery:published};
    }catch(e){
     // Preserve private Rust evidence but do not call it a delivered user project.
     const code=String(e?.code||'PUBLICATION_FAILED').replace(/[^A-Z0-9_]/g,'').slice(0,64);
     result={...result,status:'VERIFIED_NOT_DELIVERED',projectDelivery:{status:'NOT_DELIVERED',code},presentation:{kind:'UNDELIVERED',title:'Verified changes not saved to project',footer:'Ready · original project delivery incomplete'}};
     provider.post(view,{type:'publication_incomplete',code});
    }
    const {taskAcceptance,commitDigest,...publicResult}=result;provider.post(view,{type:'settled',...publicResult});
   }
   if(provider.memoryFinish)await provider.memoryFinish(view,selectedRoot,msg.goal,result).catch(()=>{});
   if(result.status==='DELIVERED'&&result.verified){
    const record={root,projectRoot:location.isolated?selectedRoot:root,profile:launch.profile,artifacts:result.taskAcceptance?.artifacts||[]},records={...(app.context.globalState.get('voder.generalTasks',{})||{}),[running.sessionId]:record};await app.context.globalState.update('voder.generalTasks',records);
    provider.post(view,{type:'task_delivered',sessionId:running.sessionId,path:record.projectRoot,files:record.artifacts.map(x=>x.path)});
    try{await this.previewRecord({...record,root:record.projectRoot},running.sessionId,'Delivered artifact');}catch(e){provider.post(view,{type:'info',text:'Files saved; preview is unavailable.'});}
   }
   return result;
  }catch(e){if(provider.memoryFail)await provider.memoryFail(view,selectedRoot,e?.code||'OUTCOME_UNKNOWN').catch(()=>{});if(executionStarted)throw e;provider.post(view,{type:'not_started',status:'START_BLOCKED',goal:msg.goal,code:e.code||'GENERAL_START_FAILED',message:'Task start blocked: '+String(e.message).slice(0,600)});}
  finally{this.reserved.delete(selectedRoot);}
 }
 async participant(folder,goal,stream,token,referenceContext=[]){
  const view={},pending=new Map([[view,{folder,sessionId:null}]]);let settled,lastNotice,listener;
  const provider={views:new Set([view]),pending,post:(_,m)=>{if(m.type==='accepted'&&token?.isCancellationRequested){const id=this.runs.activeSession(folder);if(id)void this.runs.cancel(folder,id).catch(()=>{});}if(m.type==='settled')settled=m;if(m.type==='not_started')lastNotice=m.message;if(m.type==='event')stream.progress('Voder: '+String(m.event?.type||'task progress'));},requestNativeApproval:async (_,scope)=>{if(token?.isCancellationRequested)return false;return await vscode.window.showWarningMessage('Run this Voder task in '+scope.target+'? Model and sandbox usage apply. Final source writes still require Rust VGR.','Run task','Cancel')==='Run task';}};
  const cancel=()=>{const id=this.runs.activeSession(folder);if(id)void this.runs.cancel(folder,id).catch(()=>{});};
  if(token?.onCancellationRequested)listener=token.onCancellationRequested(cancel);
  try{
   const result=await this.start(provider,view,{goal,referenceContext},folder)||settled||{status:'NOT_STARTED',verified:false};
   stream.markdown('**Voder — '+String(result.presentation?.title||result.status)+'**\n\n'+(result.verified?'Configured checks passed and native source delivery was confirmed. Task correctness and visual review are separate.':String(lastNotice||(['ANSWER','RENDERED_RESPONSE'].includes(result.presentation?.kind)?'Response complete; no project files were changed.':result.presentation?.kind==='PARTIAL'?'Rendering or visual input is incomplete; see the reported limitations.':'No verified native delivery was established.'))));
   if(result.taskAcceptance?.artifacts?.length)stream.markdown('\n\n'+result.taskAcceptance.artifacts.map(x=>'`'+x.path+'`').join(', '));
   if(result.draft)stream.markdown('\n\n'+result.draft.slice(0,60000).replace(/[<>]/g,c=>c==='<'?'&lt;':'&gt;'));
   return {metadata:{status:result.status,verified:result.verified===true,sessionId:result.sessionId}};
  }finally{listener?.dispose();}
 }
 async reviewClient({id,folder}){
  const record=this.app.context.globalState?.get?.('voder.generalTasks',{})?.[id];if(!record)return null;
  if(record.root!==folder.uri.fsPath)throw Error('Review workspace differs from the task');
  const launch=await readLaunch(record.profile);if(launch.root!==record.root)throw Error('Review launch identity differs');
  return this.app._open(folder,{mode:'native',gateway:record.profile,generalTask:true,nodeExecutable:launch.nodeBinary});
 }
 async action(id,command){
  const r=this.app.context.globalState?.get?.('voder.generalTasks',{})?.[id];if(!r)return false;
  if(command==='open-project'){await vscode.commands.executeCommand('vscode.openFolder',vscode.Uri.file(r.projectRoot||r.root),{forceNewWindow:true});return true;}
  if(command!=='preview')return false;
  await this.previewRecord({...r,root:r.projectRoot||r.root},id,'Delivered artifact');return true;
 }
 async previewCurrentFile(){
  const editor=vscode.window.activeTextEditor,doc=editor?.document;
  if(!doc||!['file','vscode-remote'].includes(doc.uri?.scheme))throw Error('Open a saved local or remote file first');
  if(doc.isDirty)throw Error('Save the current file before previewing it');
  const p=doc.uri.fsPath,b=await fs.readFile(p),root=path.dirname(p),name=path.basename(p);
  return this.previewRecord({root,artifacts:[{path:name,sha256:sha(b)}]},'file:'+p,'Manual preview · delivery not verified');
 }
 async previewRecord(record,key,label){
  const artifacts=record.artifacts||[];if(!artifacts.length)return false;
  const selected=artifacts.find(x=>/(^|\/)index\.html?$/i.test(x.path))||artifacts.find(x=>/\.html?$/i.test(x.path))||artifacts.find(x=>imageTypes[path.extname(x.path).toLowerCase()])||artifacts[0];
  const root=record.root;if(await fs.realpath(root)!==root)throw Error('Preview root changed');
  const contents=new Map();let total=0;
  for(const a of artifacts){
   if(typeof a.path!=='string'||path.isAbsolute(a.path)||a.path.includes('\\')||a.path.split('/').some(x=>!x||x==='.'||x==='..')||contents.has(a.path))throw Error('Invalid preview artifact path');
   const p=path.join(root,a.path),st=await fs.lstat(p);
   if(!st.isFile()||st.isSymbolicLink()||await fs.realpath(p)!==p||st.size>8*1024*1024||(total+=st.size)>32*1024*1024)throw Error('Preview artifact is unavailable or exceeds local preview capacity');
   const b=await fs.readFile(p);if(sha(b)!==a.sha256)throw Error('Delivered artifact changed');contents.set(a.path,b);
  }
  const html=/\.html?$/i.test(selected.path),image=imageTypes[path.extname(selected.path).toLowerCase()];
  if(!html&&!image){const uri=vscode.Uri.file(path.join(root,selected.path));if(/\.md$/i.test(selected.path))await vscode.commands.executeCommand('markdown.showPreviewToSide',uri);else await vscode.commands.executeCommand('vscode.open',uri,{viewColumn:vscode.ViewColumn.Beside,preview:true,preserveFocus:true});return true;}
  const digest=sha(Buffer.from(JSON.stringify(artifacts))),old=this.previews.get(key);
  if(old?.digest===digest){old.panel.reveal(vscode.ViewColumn.Beside,true);return true;}
  const temp=path.join(this.storage,'task-preview-'+crypto.randomUUID());await fs.mkdir(temp,{mode:0o700});let panel;
  try{
   // Snapshot only recorded artifact bytes. No arbitrary workspace resource root.
   for(const [name,b]of contents){const dest=path.join(temp,name);await fs.mkdir(path.dirname(dest),{recursive:true,mode:0o700});await fs.writeFile(dest,b,{mode:0o600,flag:'wx'});}
   panel=vscode.window.createWebviewPanel('voder.taskPreview','Voder · '+path.basename(selected.path),{viewColumn:vscode.ViewColumn.Beside,preserveFocus:true},{enableScripts:html,localResourceRoots:[vscode.Uri.file(temp)],retainContextWhenHidden:true});
   const resource=panel.webview.asWebviewUri(vscode.Uri.file(path.join(temp,selected.path))).toString();
   panel.webview.html=previewHtml({content:contents.get(selected.path).toString('utf8'),base:resource.replace(/[^/]*$/,''),source:panel.webview.cspSource,label:label+' · '+selected.path,...(image?{image:resource}:{})});
   this.previews.set(key,{panel,digest});panel.onDidDispose(()=>{if(this.previews.get(key)?.panel===panel)this.previews.delete(key);void fs.rm(temp,{recursive:true,force:true});});old?.panel.dispose();return true;
  }catch(e){panel?.dispose();await fs.rm(temp,{recursive:true,force:true});throw e;}

 }
}
module.exports={TaskEntry,previewHtml};
