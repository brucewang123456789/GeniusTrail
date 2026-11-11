'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs/promises'),path=require('node:path'),os=require('node:os'),crypto=require('node:crypto'),Module=require('node:module');
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
async function fixture(fn){
 const top=await fs.mkdtemp(path.join(os.tmpdir(),'voder-preview-test-')),root=path.join(top,'project'),storage=path.join(top,'storage');await fs.mkdir(root);await fs.mkdir(storage);
 const panels=[],commands=[],updates=[],state=new Map();let failPanel=false;
 const vscode={Uri:{file:p=>({scheme:'file',fsPath:p})},ViewColumn:{Beside:2},window:{activeTextEditor:null,createWebviewPanel(type,title,column,options){if(failPanel)throw Error('Editor preview unavailable');let disposed;const p={type,title,column,options,webview:{cspSource:'https://preview.invalid',asWebviewUri:u=>({toString:()=> 'https://preview.invalid/assets/'+encodeURIComponent(u.fsPath)})},onDidDispose:f=>disposed=f,reveal(...args){this.revealed=args;},dispose(){disposed?.();}};panels.push(p);return p;}},commands:{executeCommand:async(...args)=>commands.push(args)}};
 const file=require.resolve('../src/general-entry.cjs'),load=Module._load;delete require.cache[file];
 Module._load=function(name,parent,...args){if(parent?.filename===file){if(name==='vscode')return vscode;if(name==='./general-workspace.cjs')return{launchGeneral:async()=>({root,profile:'/fixture/profile'})};if(name==='./ordinary-workspace.cjs')return{ordinaryWorkspace:async()=>({root,isolated:false})};if(name==='./task-workspace.cjs')return{planTask:async()=>({})};if(name==='./general-run.cjs')return{GeneralRunService:class{activeSession(){return null;}}};}return load.call(this,name,parent,...args);};
 let TaskEntry;try{TaskEntry=require(file).TaskEntry;}finally{Module._load=load;delete require.cache[file];}
 const app={context:{globalStorageUri:{fsPath:storage},globalState:{get:(k,d)=>state.get(k)??d,update:async(k,v)=>state.set(k,v)}},chatRuns:{activeSession:()=>null},_agentRootForChat:async()=>'/fixture/agent',_trackChatSession(){}};
 const entry=new TaskEntry(app),view={},provider={views:new Set([view]),pending:new Map([[view,{}]]),post:(_,m)=>updates.push(m)};
 const put=async(name,content)=>{await fs.mkdir(path.dirname(path.join(root,name)),{recursive:true});await fs.writeFile(path.join(root,name),content);return{path:name,sha256:sha(Buffer.from(content))};};
 try{await fn({root,storage,entry,vscode,panels,commands,put,updates,state,fail:()=>{failPanel=true;},async start(result){entry.runs.start=async()=>({sessionId:'delivered-fixture',completion:Promise.resolve(result)});return entry.start(provider,view,{goal:'Create an animation'},{uri:{fsPath:root},name:'Project'});}});}finally{for(const p of panels)p.dispose();await fs.rm(top,{recursive:true,force:true});}
}
test('verified delivery automatically opens an interactive preview beside the editor; preview click reuses it',()=>fixture(async f=>{
 const artifact=await f.put('index.html','<!DOCTYPE html><html><head></head><body><button onclick="this.textContent=2">1</button></body></html>');
 const result={status:'DELIVERED',verified:true,taskAcceptance:{artifacts:[artifact]}};assert.equal(await f.start(result),result);assert.equal(f.panels.length,1);assert.equal(f.panels[0].column.viewColumn,2);assert.equal(f.panels[0].options.enableScripts,true);assert.match(f.panels[0].webview.html,/sandbox="allow-scripts"/);assert.doesNotMatch(f.panels[0].webview.html,/allow-same-origin/);assert.equal(f.commands.length,0);
 assert.equal(await f.entry.action('delivered-fixture','preview'),true);assert.equal(f.panels.length,1);assert.deepEqual(f.panels[0].revealed,[2,true]);
}));
test('unverified task never opens automatically and preview failure cannot downgrade confirmed delivery',()=>fixture(async f=>{
 const a=await f.put('index.html','<!DOCTYPE html><html><head></head><body>Fixture</body></html>');await f.start({status:'DRAFT_UNVERIFIED',verified:false,taskAcceptance:{artifacts:[a]}});assert.equal(f.panels.length,0);
 f.fail();const result={status:'DELIVERED',verified:true,taskAcceptance:{artifacts:[a]}};assert.equal(await f.start(result),result);assert.ok(f.updates.some(m=>m.type==='info'&&m.text.includes('preview is unavailable')));
}));
test('only hash-matching delivered assets enter the private preview; changed files are refused',()=>fixture(async f=>{
 const a=await f.put('index.html','<!DOCTYPE html><html><head><script src="app.js"></script></head><body></body></html>'),b=await f.put('app.js','window.fixture=1;');await f.put('private.txt','Must never enter preview');
 await f.entry.previewRecord({root:f.root,artifacts:[a,b]},'assets','Delivered artifact');const dir=f.panels[0].options.localResourceRoots[0].fsPath;assert.deepEqual((await fs.readdir(dir)).sort(),['app.js','index.html']);assert.equal(await fs.readFile(path.join(dir,'app.js'),'utf8'),'window.fixture=1;');
 await fs.writeFile(path.join(f.root,'app.js'),'changed');await assert.rejects(f.entry.previewRecord({root:f.root,artifacts:[a,b]},'changed','Delivered artifact'),/changed/);assert.equal(f.panels.length,1);
}));
test('manual current-file preview opens recovered HTML without registering a verified delivery',()=>fixture(async f=>{
 await f.put('index.html','<!DOCTYPE html><html><head></head><body>Recovered</body></html>');f.vscode.window.activeTextEditor={document:{uri:{scheme:'file',fsPath:path.join(f.root,'index.html')},isDirty:false}};
 assert.equal(await f.entry.previewCurrentFile(),true);assert.match(f.panels[0].webview.html,/Manual preview/);assert.match(f.panels[0].webview.html,/delivery not verified/);assert.equal(f.state.has('voder.generalTasks'),false);assert.equal(f.commands.length,0);
}));
test('SVG/images use visual preview and other delivered formats open beside the editor',()=>fixture(async f=>{
 const a=await f.put('figure.svg','<svg xmlns="http://www.w3.org/2000/svg"/>');await f.entry.previewRecord({root:f.root,artifacts:[a]},'svg','Delivered artifact');assert.equal(f.panels[0].options.enableScripts,false);assert.match(f.panels[0].webview.html,/<img /);
 const b=await f.put('notes.txt','Delivered');await f.entry.previewRecord({root:f.root,artifacts:[b]},'text','Delivered artifact');assert.equal(f.commands[0][0],'vscode.open');assert.equal(f.commands[0][2].viewColumn,2);const c=await f.put('report.md','# Delivered');await f.entry.previewRecord({root:f.root,artifacts:[c]},'markdown','Delivered artifact');assert.equal(f.commands[1][0],'markdown.showPreviewToSide');
}));
