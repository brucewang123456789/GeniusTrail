'use strict';
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
test('same terminal concurrent bind produces exactly one new session, no action',async()=>{
 const context={subscriptions:[],globalStorageUri:{fsPath:'/tmp/voder-test'}},app=new VoderExtension(context);
 let creates=0,subscribes=0;const id='ca29c10d-46f3-430a-83be-5b4875d22aba';
 app._open=async()=>({request:async method=>{if(method==='session.subscribe'){subscribes++;return {status:'SUBSCRIBED'};}assert.equal(method,'session.create');creates++;await new Promise(r=>setTimeout(r,5));return {sessionId:id};}});
 const terminal={creationOptions:{cwd:'/home/test/repo'}};
 const [a,b]=await Promise.all([app._bind(terminal,folder),app._bind(terminal,folder)]);
 assert.equal(creates,1);assert.equal(subscribes,1);assert.equal(a.id,b.id);assert.equal(a.id,id);
 app.onTerminalClose(terminal);assert.equal(app.terminalMap.has(terminal),false);
});
test('onTerminalOpen never binds in untrusted workspace',async()=>{
 const app=new VoderExtension({subscriptions:[],globalStorageUri:{fsPath:'/tmp/voder-test'}});let called=0;
 app._bind=async()=>called++;
 vscode.workspace.isTrusted=false;
 await app.onTerminalOpen({creationOptions:{cwd:'/home/test/repo'}});
 assert.equal(called,0);vscode.workspace.isTrusted=true;
});

test('recovery SecretStorage checkpoint retains the exact submitted task and action budget',async()=>{
 const app=new VoderExtension({subscriptions:[],globalStorageUri:{fsPath:'/tmp/voder-test'}});
 const id='ca29c10d-46f3-430a-83be-5b4875d22aba';let saved,sent;
 app._pickBinding=async()=>({id,folder,autoContinue:true});
 app._open=async()=>({recoveryCapable:true,request:async(method,params)=>{if(method==='task.submit'){sent=params.task;return {status:'ACCEPTED'};}return {status:'WAITING_FOR_INPUT'};}});
 app._saveRecoveryTask=async(f,s,task)=>{saved=structuredClone(task);};
 const result=await app.submitTask(false,{goal:'Keep exact original task',requiredFiles:['docs/contract.md'],maxActions:1});
 assert.equal(result.sessionId,id);assert.deepEqual(saved,sent);assert.deepEqual(saved.requiredFiles,['docs/contract.md']);assert.equal(saved.maxActions,1);
});

test('retains explicit operator source discovery without requiring workspace JSON',async()=>{
 const app=new VoderExtension({subscriptions:[],globalStorageUri:{fsPath:'/tmp/voder-test'}});const old=process.env.VODER_AGENT_ROOT;process.env.VODER_AGENT_ROOT=process.env.VODER_AGENT_TEST_ROOT;
 try{const cli=await app._validateAgent('');assert.equal(cli,require('node:path').resolve(process.env.VODER_AGENT_TEST_ROOT,'packages/voder-cli/bin/voder-cli.mjs'));}finally{if(old!==undefined)process.env.VODER_AGENT_ROOT=old;else delete process.env.VODER_AGENT_ROOT;}
});
test('retains negotiated diff capability refusal with no RPC or UI diff execution',async()=>{
 const app=new VoderExtension({subscriptions:[],globalStorageUri:{fsPath:'/tmp/voder-test'}});let calls=0;app._open=async()=>({diffCapable:false,request:async()=>{calls++;throw Error('should not call');}});
 const r=await app.inspectDiff({folder,id:'ca29c10d-46f3-430a-83be-5b4875d22aba'});assert.equal(r.status,'NOT_AVAILABLE');assert.equal(calls,0);
});
test('retains disposal of all private in-memory diff snapshots',()=>{
 const app=new VoderExtension({subscriptions:[],globalStorageUri:{fsPath:'/tmp/voder-test'}});app.diffDocuments.set('test','owned snapshot');app.eventCounts.set('test',1);app.deactivate();assert.equal(app.diffDocuments.size,0);assert.equal(app.eventCounts.size,0);
});

test('actual CLI completion barrier precedes inactive cancellation refusal',async()=>{
 const fs=require('node:fs/promises'),path=require('node:path');
 const dir=await fs.mkdtemp(path.join(require('node:os').tmpdir(),'a10-settlement-'));
 const agent=process.env.VODER_AGENT_TEST_ROOT;
 const {RpcClient}=require('../src/rpc-client.cjs');
 const rpc=new RpcClient({command:process.execPath,args:[path.join(agent,'packages/voder-cli/bin/voder-cli.mjs'),'rpc','--draft','--workspace',dir,'--session-dir',path.join(dir,'journal')],cwd:dir});
 const app=new VoderExtension({subscriptions:[],globalStorageUri:{fsPath:dir}});app._open=async()=>rpc;
 try{
  await rpc.request('initialize');const {sessionId:id}=await rpc.request('session.create');
  await rpc.request('task.submit',{sessionId:id,task:{goal:'Need absent user input',requiredFiles:['missing.txt']}});
  const state=await app.showStatus({folder,id,quiet:true,waitForSettlement:true});assert.equal(state.status,'WAITING_FOR_INPUT');
  await assert.rejects(app.cancelTask({folder,id,userConfirmed:true}),e=>e.code==='SESSION_NOT_ACTIVE');
  assert.equal((await rpc.request('session.status',{sessionId:id})).status,'WAITING_FOR_INPUT');
 }finally{rpc.close();await fs.rm(dir,{recursive:true,force:true});}
});

test('delayed durable journal reproduces status-before-settlement window without paid services',async()=>{
 const fs=require('node:fs/promises'),path=require('node:path');
 const {pathToFileURL}=require('node:url');
 const {VoderSDK,VoderControlPlane,VoderJournal}=await import(pathToFileURL(path.join(process.env.VODER_AGENT_TEST_ROOT,'packages/voder-sdk/src/index.mjs')));
 const dir=await fs.mkdtemp(path.join(require('node:os').tmpdir(),'a10-race-'));
 const sdk=new VoderSDK({cwd:dir,native:{async executeAuthorized(){throw Error('NO_NATIVE_ALLOWED');}},modelHost:{async createSession(){throw Error('NO_MODEL_ALLOWED');}},allowUnverifiedDraft:true});
 const journal=await new VoderJournal({dir:path.join(dir,'journal'),workspace:dir}).initialize();const host=new VoderControlPlane({sdk,journal});let release;
 try{
  const {sessionId}=await host.createSession();let entered;const hit=new Promise(r=>entered=r),gate=new Promise(r=>release=r),original=journal.write.bind(journal);
  journal.write=async(id,record)=>{if(record.ended_at&&!record.cancel_requested){entered();await gate;}return original(id,record);};
  await host.submitTask(sessionId,{goal:'Need absent user input',requiredFiles:['missing.txt']});await hit;
  assert.equal((await host.status(sessionId)).status,'WAITING_FOR_INPUT');
  const ack=host.cancel(sessionId);release();assert.equal((await ack).status,'CANCELLATION_REQUESTED');
  await host.wait(sessionId);await assert.rejects(host.cancel(sessionId),e=>e.code==='SESSION_NOT_ACTIVE');
 }finally{release?.();host.dispose();await fs.rm(dir,{recursive:true,force:true});}
});
