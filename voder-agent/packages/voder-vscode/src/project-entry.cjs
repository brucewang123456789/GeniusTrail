'use strict';
const vscode=require('vscode'),fs=require('node:fs/promises'),path=require('node:path'),crypto=require('node:crypto');
const {execFileSync}=require('node:child_process');
const {planProject,createProject,validateProject,TARGETS,isProjectGoal,digest}=require('./project-workspace.cjs');
const {ProjectRunService}=require('./project-run.cjs');
const {nativeEnv}=require('./native-env.cjs');
const {validateVerifiedDiff}=require('./verified-diff.cjs');
class ProjectEntry {
 constructor(app){this.app=app;this.storage=app.context.globalStorageUri.fsPath;this.records=new Map();this.runs=new ProjectRunService(async(folder,profile)=>app._open(folder,{mode:'native',gateway:profile,nodeExecutable:JSON.parse(await fs.readFile(profile,'utf8')).nodeBinary}),this.storage);}
 async start(provider,view,msg){
  const app=this.app,anchor=app.chatWorkspace?.()||{uri:app.context.globalStorageUri,name:'Voder'};
  const agentRoot=await app._agentRootForChat(anchor);
  nativeEnv(process.env); // Preflight only: no model calls, sandbox calls or project writes.
  const plan=await planProject({agentRoot,storageRoot:this.storage});
  const approved=await provider.requestNativeApproval(view,{project:true,files:TARGETS,operation:'Create an isolated web project with only the listed editable files.',checksSummary:'Pinned structural checks only; browser behavior and visual quality require separate review.',target:plan.root+' / '+TARGETS.join(', '),validator:'pinned web structure checks (not browser or visual tests)'});
  if(!approved){provider.post(view,{type:'not_started',goal:msg.goal,message:'New project approval declined. No project or task was created.'});return;}
  if(!provider.views.has(view))return;
  // Ensure source/runtime discovery and the exact workspace identity survive the separate generated folder.
  await app.context.globalState.update('voder.chat.agentRoot',agentRoot);
  const project=await createProject(plan),folder={uri:vscode.Uri.file(project.root),name:path.basename(project.root)};
  const pending=provider.pending.get(view);if(pending)pending.folder=folder;
  provider.post(view,{type:'workspace_selected',path:project.root});provider.post(view,{type:'phase_started',phase:'native'});
  provider.post(view,{type:'info',text:'New isolated project: '+project.root+'. Only app.js, index.html and style.css are editable. Verification covers structure; preview is required for visual acceptance.'});
  const running=await this.runs.start(folder,msg.goal,update=>{
   if(['created','accepted'].includes(update.kind)){
    const id=update.sessionId;this.records.set(id,project);app._trackChatSession(id,folder,project.profile);const p=provider.pending.get(view);if(p)p.sessionId=id;
   }
   provider.post(view,{...update,type:update.kind});
  },{project,...(msg.referenceContext?.length?{referenceContext:msg.referenceContext}:{}),onCreated:async id=>{const index={...(app.context.globalState.get('voder.webProjects',{})||{}),[id]:project.recordPath};await app.context.globalState.update('voder.webProjects',index);}});
  const settled=await running.completion;
  if(settled?.status==='DELIVERED'&&settled.verified===true){
   provider.post(view,{type:'project_delivered',sessionId:running.sessionId,path:project.root});
   // A preview failure never changes the independently reported native delivery status.
   try{await this.preview(running.sessionId);provider.post(view,{type:'info',text:'Project preview opened. Inspect the animation visually; automatic checks validated structure only.'});}
   catch(e){provider.post(view,{type:'info',text:'Project delivered; preview unavailable: '+String(e.message).slice(0,220)});}
  }
 }
 async checked(id){const recordPath=this.records.get(id)?.recordPath||this.app.context.globalState.get('voder.webProjects',{})?.[id];if(!recordPath)throw Error('Unknown generated-project session; no task was rerun');return validateProject(recordPath,this.storage);}
 async verifiedFiles(id){
  const project=await this.checked(id),folder={uri:vscode.Uri.file(project.root),name:path.basename(project.root)};
  const rpc=await this.app._open(folder,{mode:'native',gateway:project.profile,nodeExecutable:project.nodeBinary});if(!rpc.diffCapable)throw Error('Verified diff capability is required before preview');
  const diff=await rpc.request('session.diff',{sessionId:id},30000);const files=validateVerifiedDiff(diff,id);
  if(JSON.stringify(files.map(x=>x.relative_path))!==JSON.stringify(TARGETS))throw Error('Project diff does not cover the approved files');
  for(const f of files)if(digest(await fs.readFile(path.join(project.root,f.relative_path)))!==f.after_sha256)throw Error('Project changed after delivery');
  return {project,files};
 }
 async preview(id){
  const {project,files}=await this.verifiedFiles(id),contents=Object.fromEntries(files.map(x=>[x.relative_path,x.after]));
  // Passive parser only: never execute generated code in the extension host.
  execFileSync(project.nodeBinary,['--check',path.join(project.root,'app.js')],{env:{PATH:'/usr/bin:/bin'},timeout:5000,stdio:'pipe',maxBuffer:8000});
  const temp=path.join(this.storage,'web-preview-'+crypto.randomUUID());await fs.mkdir(temp,{mode:0o700});
  for(const f of ['app.js','style.css'])await fs.writeFile(path.join(temp,f),contents[f],{flag:'wx',mode:0o600});
  const panel=vscode.window.createWebviewPanel('voder.webProject','Voder · Web project preview',vscode.ViewColumn.Beside,{enableScripts:true,localResourceRoots:[vscode.Uri.file(temp)],retainContextWhenHidden:false});
  const nonce=crypto.randomBytes(24).toString('hex'),js=panel.webview.asWebviewUri(vscode.Uri.file(path.join(temp,'app.js'))).toString(),css=panel.webview.asWebviewUri(vscode.Uri.file(path.join(temp,'style.css'))).toString();
  // Insert a restrictive CSP before model HTML; no bridge handler, network, frames, forms or navigation commands.
  const csp=`default-src 'none'; script-src 'nonce-${nonce}'; style-src ${panel.webview.cspSource} 'unsafe-inline'; img-src data:; font-src 'none'; connect-src 'none'; frame-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none';`;
  let html=contents['index.html'].replace(/<script\b[^>]*>[\s\S]*?<\/script\s*>/gi,'').replace(/<link\b[^>]*>/gi,'').replace(/<meta\b[^>]*http-equiv\s*=[^>]*>/gi,'');
  html=html.replace(/<head\b[^>]*>/i,`<head><meta http-equiv="Content-Security-Policy" content="${csp}"><link rel="stylesheet" href="${css}">`);
  if(!/<head>/i.test(html))throw Error('Preview requires an explicit head element');
  html=html.replace(/<\/body\s*>/i,`<script nonce="${nonce}" src="${js}"></script></body>`);
  panel.webview.html=html;panel.onDidDispose(()=>{void fs.rm(temp,{recursive:true,force:true});});
  return project.root;
 }
 async action(id,command){
  if(!this.records.has(id)&&!this.app.context.globalState.get('voder.webProjects',{})?.[id])return false;
  if(command==='preview')await this.preview(id);
  else if(command==='open-project'){const p=await this.checked(id);await vscode.commands.executeCommand('vscode.openFolder',vscode.Uri.file(p.root),{forceNewWindow:true});}
  else return false;
  return true;
 }
}
module.exports={ProjectEntry,isProjectGoal};
