'use strict';
const vscode=require('vscode'),fs=require('node:fs/promises'),path=require('node:path'),crypto=require('node:crypto');
const {inspectLanguages,inspectDependencies,readScope,hash}=require('./workspace-tools.cjs');
const {createObserver}=require('./browser-observer.cjs');
const {runPythonChecks}=require('./sandbox-checks.cjs');
const {validateProject}=require('./project-workspace.cjs');
const {validateExtended}=require('./multi-workspace.cjs');
const esc=x=>String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function reportHtml(report){
 const rows=Array.isArray(report.syntax?.checks)?report.syntax.checks:[];
 const runtime=report.runtime;
 const title=runtime?'Runtime observations':'Project file checks';
 const table=rows.length?'<table><thead><tr><th>File</th><th>Result</th></tr></thead><tbody>'+rows.map(r=>'<tr><td>'+esc(r.file||'')+'</td><td>'+esc(r.status==='PASS'?'Passed':r.status==='FAIL'?'Issues found':'Not checked')+'</td></tr>').join('')+'</tbody></table>':'';
 const observed=runtime?'<p>'+esc(runtime.status==='NO_TELEMETRY'?'No browser observations were received.':'Browser observations collected. Appearance and controls have not been certified.')+'</p>':'';
 return `<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline';"><title>Voder · ${title}</title><style>body{color-scheme:dark;background:#080808;color:#ddd;font:16px/1.7 'Times New Roman',serif;margin:40px auto;padding:0 24px;max-width:820px}header{letter-spacing:.2em;font-size:11px;color:#aaa}h1{font-weight:400;font-size:30px;line-height:1.2;margin:22px 0}p{color:#aaa;font-size:14px}table{border-collapse:collapse;width:100%;margin:24px 0;font-size:14px}th,td{text-align:left;border-bottom:1px solid #292929;padding:12px 8px;overflow-wrap:anywhere}th{font-size:12px;color:#aaa;font-weight:400}details{border-top:1px solid #292929;margin-top:28px;padding-top:14px}summary{cursor:pointer;color:#aaa;font-size:13px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.7 ui-monospace,Consolas,monospace;color:#aaa}</style></head><body><header>VODER · PROJECT REVIEW</header><h1>${title}</h1><p>Only the checks listed here were performed. File checks do not establish visual quality or correct interaction.</p>${table}${observed}<details><summary>Developer diagnostics</summary><pre>${esc(JSON.stringify(report,null,2))}</pre></details></body></html>`;
}
class ToolkitController{
 constructor(app){this.app=app;this.panels=new Set();this.latest=new Map();}
 source(id){if(this.app.multiProjects?.records.has(id)||this.app.context.globalState.get('voder.multiWebProjects',{})?.[id])return this.app.multiProjects;return this.app.webProjects;}
 async selectSession(){
  const state=this.app.context.globalState,index={...(state.get('voder.webProjects',{})||{}),...(state.get('voder.multiWebProjects',{})||{})};
  const entries=Object.entries(index).filter(([id,p])=>/^[a-f0-9-]{36}$/i.test(id)&&typeof p==='string').slice(-128);
  if(!entries.length)throw Error('No saved generated-project session. Submit a task first.');
  const items=[];for(const[id,p]of entries)try{const project=p.includes('/multi-project-profiles/')?await validateExtended(p,this.app.context.globalStorageUri.fsPath):await validateProject(p,this.app.context.globalStorageUri.fsPath);items.push({label:id,description:project.root,id,project});}catch{}
  if(!items.length)throw Error('Saved project identities could not be validated. No task was replayed.');
  return vscode.window.showQuickPick(items,{placeHolder:'Select exact saved session · read-only inspection, never replay'});
 }
 show(report,title='Voder · Project tools'){
  const panel=vscode.window.createWebviewPanel('voder.toolReport',title,vscode.ViewColumn.Beside,{enableScripts:false,localResourceRoots:[]});panel.webview.html=reportHtml(report);this.panels.add(panel);panel.onDidDispose(()=>this.panels.delete(panel));return panel;
 }
 async inspect(id){
  if(!this.app.trusted())throw Error('Workspace Trust required');
  const owner=this.source(id),{project,files}=await owner.verifiedFiles(id),names=files.map(f=>f.relative_path);
  const syntax=await inspectLanguages(project.root,names,{node:project.nodeBinary}),rows=await readScope(project.root,names),dependencies=inspectDependencies(rows);
  const report={schema:'voder.project-tools.report.v1',session_id:id,workspace:project.root,delivery:'RECHECKED_THROUGH_NATIVE_SESSION_DIFF',syntax,dependencies,browser:'Use Inspect runtime for bounded browser observations',services:'No host server or package manager started',visual_acceptance:'NOT_ESTABLISHED'};
  this.latest.set(id,report);this.show(report);return report;
 }
 async restore(){
  if(!this.app.trusted())throw Error('Workspace Trust required');const s=await this.selectSession();if(!s)return;
  const folder={uri:vscode.Uri.file(s.project.root),name:path.basename(s.project.root)};
  // Deliberately use a draft control connection: status inspection cannot receive native/Modal credentials or auto-recover actions.
  const rpc=await this.app._open(folder,{mode:'draft',gateway:'',nodeExecutable:s.project.nodeBinary});
  const status=await rpc.request('session.status',{sessionId:s.id},15000),events=await rpc.request('session.events',{sessionId:s.id,after:0},15000);
  const tail=Array.isArray(events)?events.slice(-25).map(e=>({sequence:e.sequence,type:e.type,state:e.state,status:e.status})):[];
  this.app.chatSessionFolder.set(s.id,folder);
  const report={session_id:s.id,workspace:s.project.root,authority:'READ_ONLY_JOURNAL_METADATA_NOT_NEW_NATIVE_PROOF',status:status.status??status.state??'UNKNOWN',result:status.result??null,recent_events:tail,automatic_replay:false,next:'Inspect verified diff for a delivered result; native reconciliation is required before retrying an uncertain action.'};this.show(report,'Voder · Restored session evidence');return report;
 }
 async sandboxPython(){
  if(!this.app.trusted())throw Error('Workspace Trust required');const folder=this.app.chatWorkspace?.();if(!folder)throw Error('Open one explicit project');
  const selected=await vscode.window.showOpenDialog({title:'Select Python source and independent test*.py files for isolated checks',defaultUri:folder.uri,canSelectFiles:true,canSelectFolders:false,canSelectMany:true,filters:{Python:['py']}});if(!selected?.length)return;
  const files=selected.map(f=>path.relative(folder.uri.fsPath,f.fsPath).split(path.sep).join('/'));await readScope(folder.uri.fsPath,files);
  const decision=await vscode.window.showWarningMessage('Run these selected files in a billed isolated Modal sandbox? No project files are modified: '+files.join(', '),{modal:true},'Run sandbox checks');if(decision!=='Run sandbox checks')return;
  const agentRoot=await this.app._agentRootForChat(folder);const result=await runPythonChecks({root:folder.uri.fsPath,files,agentRoot,storageRoot:this.app.context.globalStorageUri.fsPath});this.show(result,'Voder · Isolated Python checks');return result;
 }
 async inspectCurrent(){
  if(!this.app.trusted())throw Error('Workspace Trust required');const folder=this.app.chatWorkspace?.();if(!folder)throw Error('Open one explicit project or choose a generated session');
  const files=await vscode.window.showOpenDialog({title:'Select files for passive syntax/dependency inspection · no program execution',defaultUri:folder.uri,canSelectFiles:true,canSelectFolders:false,canSelectMany:true});if(!files?.length)return;
  const names=files.map(f=>path.relative(folder.uri.fsPath,f.fsPath).split(path.sep).join('/'));
  const syntax=await inspectLanguages(folder.uri.fsPath,names),dependencies=inspectDependencies(await readScope(folder.uri.fsPath,names));this.show({workspace:folder.uri.fsPath,syntax,dependencies,source_writes:0,model_calls:0});
 }
 async runtime(id){
  if(!this.app.trusted())throw Error('Workspace Trust required');const {project,files}=await this.source(id).verifiedFiles(id),contents=Object.fromEntries(files.map(f=>[f.relative_path,f.after]));
  const syntax=await inspectLanguages(project.root,files.map(f=>f.relative_path),{node:project.nodeBinary});if(syntax.checks.some(c=>c.status==='FAIL'))throw Error('Runtime inspection blocked by syntax errors; inspect Project tools report');
  const storage=this.app.context.globalStorageUri.fsPath,temp=path.join(storage,'runtime-observation-'+crypto.randomUUID());await fs.mkdir(temp,{mode:0o700});
  // Only exact verified snapshot members, never a repository mount, secrets or dependency directories.
  for(const f of files){const target=path.join(temp,f.relative_path);await fs.mkdir(path.dirname(target),{recursive:true,mode:0o700});await fs.writeFile(target,f.after,{flag:'wx',mode:0o600});}
  const panel=vscode.window.createWebviewPanel('voder.runtimeInspection','Voder · Runtime observations',vscode.ViewColumn.Beside,{enableScripts:true,localResourceRoots:[vscode.Uri.file(temp)]});
  const nonce=crypto.randomBytes(24).toString('hex'),uri=n=>panel.webview.asWebviewUri(vscode.Uri.file(path.join(temp,n))).toString(),source=panel.webview.cspSource;
  const multi=project.schema==='voder.multi-web-project.v1';
  const base=multi?uri('index.html').replace(/[^/]*$/, ''):'';
  const baseTag=multi?'<base href="'+esc(base)+'">':'';
  let previous=null;
  const observer=createObserver(panel,row=>{
   previous=row;this.latest.set(id,{...(this.latest.get(id)||{}),runtime:row});
   this.app.output.appendLine('[BROWSER OBSERVATION / NOT VGR PROOF] '+JSON.stringify({session_id:id,...row}));
  });
  const csp=`default-src 'none'; script-src 'nonce-${nonce}' ${source}; style-src ${source} 'unsafe-inline'; img-src data: ${source}; connect-src ${source}; font-src 'none'; frame-src 'none'; object-src 'none'; base-uri ${multi?source:"'none'"}; form-action 'none';`;
  let html=(multi?contents['index.html'].replace(/<base\b[^>]*>/gi,''):contents['index.html']).replace(/<script\b[^>]*>[\s\S]*?<\/script\s*>/gi,'').replace(/<link\b[^>]*>/gi,'').replace(/<meta\b[^>]*http-equiv\s*=[^>]*>/gi,'');
  if(!/<head\b[^>]*>/i.test(html)||!/<\/body\s*>/i.test(html))throw Error('Runtime inspection requires complete head/body');
  html=html.replace(/<head\b[^>]*>/i,`<head><meta http-equiv="Content-Security-Policy" content="${csp}">${baseTag}<link rel="stylesheet" href="${uri('style.css')}"><script nonce="${nonce}">${observer.source}</script>`);
  html=html.replace(/<\/body\s*>/i,`<script ${project.schema==='voder.multi-web-project.v1'?'type="module" ':''}nonce="${nonce}" src="${uri('app.js')}"></script></body>`);
  panel.webview.html=html;this.panels.add(panel);
  const timer=setTimeout(()=>{if(!this.panels.has(panel))return;this.latest.set(id,{...(this.latest.get(id)||{}),syntax,runtime:previous??{status:'NO_TELEMETRY',authority:'NO_ACCEPTANCE_ESTABLISHED'}});this.app.output.appendLine('[Runtime observation complete] '+JSON.stringify({session_id:id,observations_received:previous!==null,visual_acceptance:'NOT_ESTABLISHED'}));},12500);
  panel.onDidDispose(()=>{clearTimeout(timer);observer.dispose();this.panels.delete(panel);void fs.rm(temp,{recursive:true,force:true});});
 }
 dispose(){for(const panel of this.panels)panel.dispose();this.panels.clear();}
}
module.exports={ToolkitController,reportHtml};
