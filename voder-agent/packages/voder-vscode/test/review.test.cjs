'use strict';
const {TestChatViewProvider}=require('./chat-test-context.cjs');
const test=require('node:test'),assert=require('node:assert/strict'),Module=require('node:module');
const {EventEmitter}=require('node:events');
const original=Module._load;
const folder={name:'repo',uri:{fsPath:'/home/test/repo'}};
function disposable(){return {dispose(){}};}
const vscode={
 TreeItem:class {constructor(label){this.label=label;}},TreeItemCollapsibleState:{None:0},ThemeIcon:class {constructor(id){this.id=id;}},
 StatusBarAlignment:{Left:0},EventEmitter:class {constructor(){this.event=()=>disposable();}fire(){}dispose(){}},
 Uri:{file:fsPath=>({fsPath})},
 workspace:{isTrusted:true,workspaceFolders:[folder],getConfiguration:()=>({get:(k,d)=>d}),getWorkspaceFolder:()=>folder,onDidChangeConfiguration:disposable},
 commands:{registerCommand:()=>disposable(),executeCommand:async()=>{}},languages:{getDiagnostics:()=>[]},
 window:{terminals:[],activeTerminal:null,createOutputChannel:()=>({appendLine(){},show(){},dispose(){}}),createStatusBarItem:()=>({show(){},dispose(){}}),
 registerTreeDataProvider:()=>disposable(),onDidOpenTerminal:()=>disposable(),onDidCloseTerminal:()=>disposable(),onDidChangeActiveTerminal:()=>disposable(),
 showInformationMessage:async()=>{},showWarningMessage:async()=>{},showErrorMessage:async()=>{}}
};
Module._load=function(id,...args){if(id==='vscode')return vscode;return original.call(this,id,...args);};
const {VoderExtension}=require('../src/extension.cjs');Module._load=original;
const {createHash}=require('node:crypto');
const {VoderChatViewProvider}=require('../src/chat-view.cjs');
const hash=x=>createHash('sha256').update(x).digest('hex');
const sid='ca29c10d-46f3-430a-83be-5b4875d22aba';
function fixture(){const values=new Map();const app=new VoderExtension({subscriptions:[],globalStorageUri:{fsPath:'/tmp/voder-test'},globalState:{get:(k,d)=>values.get(k)??d,update:async(k,v)=>values.set(k,v)}});return {app,values};}
function proof(){const action='action-'+sid;return {schema:'voder.session.diff.v1',status:'VERIFIED_DELIVERED',authority:'VODER_NATIVE_VGR_STATE',sessionId:sid,action_id:action,run_id:'run-'+action,commit_digest:hash('commit'),relative_path:'calc.py',before:'old',after:'new',before_sha256:hash('old'),after_sha256:hash('new')};}
test('review reuses exact persisted task profile; root calc.py opens read-only editor snapshots',async()=>{const {app,values}=fixture();app._trackChatSession(sid,folder,'/private/exact.json');assert.equal(values.get('voder.chat.reviewBindings')[sid].profile,'/private/exact.json');app.chatReviewProfiles.clear();let opened=0;vscode.Uri.parse=s=>({toString:()=>s});vscode.commands.executeCommand=async(name)=>{assert.equal(name,'vscode.diff');opened++;};app._open=async(f,opts)=>{assert.equal(opts.gateway,'/private/exact.json');return {diffCapable:true,request:async(method,params)=>{assert.equal(method,'session.diff');assert.equal(params.sessionId,sid);return proof();}};};const r=await app.inspectDiff({folder,id:sid});assert.equal(r.editor_documents.length,1);assert.equal(opened,1);assert.equal(app.diffDocuments.size,2);});
test('legacy completed chat uses already approved profile without new authorization or submit',async()=>{const {app,values}=fixture();values.set(app._chatProfileKey(folder),'/private/approved.json');app._open=async(f,opts)=>{assert.equal(opts.gateway,'/private/approved.json');return {diffCapable:false,request(){throw Error('No RPC permitted');}};};assert.equal((await app.inspectDiff({folder,id:sid})).status,'NOT_AVAILABLE');});
test('review refuses mismatched workspace before opening any connection',async()=>{const {app}=fixture();app.chatReviewProfiles.set(sid,{workspace:'/other',profile:'/private/x.json'});app._open=()=>{throw Error('must not open')};await assert.rejects(app.inspectDiff({folder,id:sid}),/workspace/);});
test('review refuses corrupt bytes before invoking editor',async()=>{const {app}=fixture();let calls=0;vscode.commands.executeCommand=async()=>calls++;app._open=async()=>({diffCapable:true,request:async()=>({...proof(),after:'tampered'})});await assert.rejects(app.inspectDiff({folder,id:sid}),/digest/);assert.equal(calls,0);});
for(const outcome of ['unsupported','throws','opens','unknown'])test('chat diff feedback is explicit: '+outcome,async()=>{const messages=[];const view={webview:{postMessage:m=>messages.push(m)}};const app={chatSessionFolder:new Map(outcome==='unknown'?[]:[[sid,folder]]),inspectDiff:async()=>{if(outcome==='throws')throw Error('proof mismatch');if(outcome==='opens')return {status:'VERIFIED_DELIVERED',editor_documents:[{}]};return {status:'NOT_AVAILABLE',reason:'unsupported'};}};const provider=new TestChatViewProvider(app);provider.views.add(view);await provider.handle(view,{type:'action',command:'diff',sessionId:sid});assert.equal(messages.length,1);assert.equal(messages[0].type,'action_result');assert.equal(messages[0].status,outcome==='opens'?'OPENED':outcome==='throws'?'ERROR':'NOT_AVAILABLE');});

test('after reload, command palette discovers historical sessions through approved profile',async()=>{const {app,values}=fixture();values.set(app._chatProfileKey(folder),'/private/approved.json');vscode.window.showQuickPick=async rows=>rows[0];app._open=async(f,opts)=>{assert.equal(opts.gateway,'/private/approved.json');return {request:async name=>{assert.equal(name,'session.list');return [{session_id:sid,status:'DELIVERED'}];}};};const picked=await app._pickSession();assert.equal(picked.id,sid);});
