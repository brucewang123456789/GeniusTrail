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
test('one click ignores terminal draft and automatically selects newest delivered session',async()=>{const {app}=fixture();app._folder=async()=>folder;vscode.window.activeTerminal={};app.terminalMap.set(vscode.window.activeTerminal,{id:'wrong',folder});const newer='22345678-1234-4234-8234-123456789abc';app._reviewClient=async()=>({request:async method=>{assert.equal(method,'session.list');return [{session_id:sid,status:'DELIVERED',started_at:'2026-10-16T01:00:00Z'},{session_id:'32345678-1234-4234-8234-123456789abc',status:'CREATED',started_at:'2026-10-16T03:00:00Z'},{session_id:newer,status:'DELIVERED',started_at:'2026-10-16T02:00:00Z'}];}});assert.equal((await app._pickDiffSession()).id,newer);vscode.window.activeTerminal=null;});
test('explicit session review never silently switches to another delivered task',async()=>{const {app}=fixture();app._reviewClient=async()=>{throw Error('must not discover')};assert.equal((await app._pickDiffSession({folder,id:sid})).id,sid);});
test('no delivered task does not create or execute any task',async()=>{const {app}=fixture();app._folder=async()=>folder;app._reviewClient=async()=>({request:async method=>{assert.equal(method,'session.list');return [{session_id:sid,status:'VERIFIED_NOT_DELIVERED'}];}});assert.equal(await app._pickDiffSession(),null);});
test('settled delivered task opens exact verified diff automatically without submitting',async()=>{const out=[];let calls=0;const view={webview:{postMessage:m=>out.push(m)}};const provider=new TestChatViewProvider({inspectDiff:async args=>{calls++;assert.equal(args.id,sid);return {editor_documents:[{}]};}});provider.views.add(view);await provider.openDeliveredDiff(view,folder,{sessionId:sid,status:'DELIVERED',verified:true});assert.equal(calls,1);assert.equal(out[0].status,'OPENING');assert.equal(out.at(-1).status,'OPENED');});
test('native commit alone never triggers automatic workspace diff',async()=>{let calls=0;const view={};const provider=new TestChatViewProvider({inspectDiff:async()=>calls++});provider.views.add(view);await provider.openDeliveredDiff(view,folder,{sessionId:sid,status:'VERIFIED_NOT_DELIVERED',verified:true});assert.equal(calls,0);});
