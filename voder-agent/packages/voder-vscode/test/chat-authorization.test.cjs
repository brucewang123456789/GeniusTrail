'use strict';
const {TestChatViewProvider}=require('./chat-test-context.cjs');
/** Production extension auth-control tests, not mere source-string assertions. */
const test=require('node:test'),assert=require('node:assert/strict');
const fs=require('node:fs'),fsp=require('node:fs/promises'),os=require('node:os'),path=require('node:path'),crypto=require('node:crypto'),Module=require('node:module');
const {execFileSync}=require('node:child_process');
const {resolveNativeIntent}=require('../src/native-intent.cjs');
function git(dir,...args){return execFileSync('/usr/bin/git',['-C',dir,...args],{encoding:'utf8'});}
async function repository(){
 const root=await fsp.mkdtemp(path.join(os.tmpdir(),'voder-chat-auth-'));git(root,'init','-q');
 for(const [p,content] of [['calc.py','def add(a,b):return a-b\n'],['test_calc.py','from calc import add\nassert add(1,2)==3\n'],['other.py','pass\n'],['test_other.py','pass\n']])await fsp.writeFile(path.join(root,p),content);
 git(root,'add','.');git(root,'-c','user.name=Voder','-c','user.email=test@invalid.local','commit','-qm','baseline');
 return {root,folder:{uri:{fsPath:root},name:'fixture'},finish:()=>fsp.rm(root,{force:true,recursive:true})};
}
test('auto scope from Chinese task identifies exact existing source and validator',async()=>{
 const f=await repository();try{
  const goal='修复已有calc.py中的add函数，使test_calc.py的3个断言通过；只能修改calc.py，不修改测试脚本。';
  const scope=resolveNativeIntent(f.root,goal);
  assert.equal(scope.target,'calc.py');assert.equal(scope.test,'test_calc.py');
  assert.equal(resolveNativeIntent(f.root,'修复calc.py').test,'test_calc.py');
 }finally{await f.finish();}
});
test('goal cannot silently choose a missing or unsafe or ambiguous tracked source',async()=>{
 const f=await repository();try{
  assert.throws(()=>resolveNativeIntent(f.root,'请创建 brand_new.py'),e=>e.code==='CHAT_SCOPE_FILE_NOT_TRACKED');
  assert.throws(()=>resolveNativeIntent(f.root,'修复 calc.py 和 other.py'),e=>e.code==='CHAT_SCOPE_AMBIGUOUS_TARGET');
  assert.throws(()=>resolveNativeIntent(f.root,'修复 test_calc.py'),e=>e.code==='CHAT_SCOPE_AMBIGUOUS_TARGET');
  assert.equal(resolveNativeIntent(f.root,'修复已有函数'),null);
  await fsp.mkdir(path.join(f.root,'src'));await fsp.mkdir(path.join(f.root,'extras'));
  await fsp.writeFile(path.join(f.root,'src/foo.py'),'pass\n');await fsp.writeFile(path.join(f.root,'extras/foo.py'),'pass\n');
  git(f.root,'add','src/foo.py','extras/foo.py');
  assert.throws(()=>resolveNativeIntent(f.root,'修复 foo.py'),e=>e.code==='CHAT_SCOPE_AMBIGUOUS_PATH');
 }finally{await f.finish();}
});
test('git worktree scope never follows source symlinks',async()=>{
 const f=await repository();try{
  await fsp.unlink(path.join(f.root,'calc.py'));await fsp.symlink('/etc/passwd',path.join(f.root,'calc.py'));
  assert.throws(()=>resolveNativeIntent(f.root,'修复 calc.py'),e=>e.code==='CHAT_SCOPE_UNSAFE_FILE');
 }finally{await f.finish();}
});
// Load actual VoderExtension class with a simulated editor host, not a new authorization implementation.
const filename=path.resolve(__dirname,'../src/extension.cjs');
let scenario;
const fakeVscode={
 TreeItem:class {},TreeItemCollapsibleState:{None:0},ThemeIcon:class {},StatusBarAlignment:{Left:0},
 EventEmitter:class {event(){}fire(){}dispose(){}},
 workspace:{isTrusted:true,getConfiguration(){return {get:(key,def)=>key==='nativeGatewayConfig'?(scenario?.configured||''):def};}},
 window:{createOutputChannel(){return {appendLine(text){scenario?.logs.push(text);},dispose(){}};},createStatusBarItem(){return {show(){},dispose(){}};},
  async showQuickPick(){scenario.picks++;return scenario.quickChoice?{value:scenario.quickChoice}:undefined;},
  async showOpenDialog(){scenario.dialogs++;return scenario.dialogs===1?[{fsPath:path.join(scenario.root,'calc.py')}]:[{fsPath:path.join(scenario.root,'test_calc.py')}];},
  async showWarningMessage(){scenario.approvals++;return scenario.approve?'Approve scoped native':undefined;}}
};
const fakeProfile={
 async validateProfile(file,folder){scenario.validations++;
  if(file==='invalid')throw Object.assign(Error('scope mismatch'),{code:'CHAT_PROFILE_POLICY_MISMATCH'});
  const scope=scenario.created.get(file);if(!scope||scope.workspace!==folder.uri.fsPath)throw Object.assign(Error('workspace mismatch'),{code:'CHAT_PROFILE_POLICY_MISMATCH'});
  return {...scope};
 },
 async createProfile({folder,targetFile,testFile}){scenario.creations++;
  const scope={path:'/private/'+crypto.randomUUID()+'.json',target:path.basename(targetFile),validator:'/usr/bin/env /usr/local/bin/python3.10 -B '+path.basename(testFile),workspace:folder.uri.fsPath};
  scenario.created.set(scope.path,scope);return scope;
 }
};
const originalLoad=Module._load;let VoderExtension;
try{
 Module._load=function(request,parent,...rest){if(request==='vscode')return fakeVscode;if(request==='./native-profile.cjs'&&parent?.filename===filename)return fakeProfile;return originalLoad.call(this,request,parent,...rest);};
 ({VoderExtension}=require(filename));
}finally{Module._load=originalLoad;}
function newApp(f){
 scenario={root:f.root,stored:new Map(),created:new Map(),approvals:0,creations:0,validations:0,picks:0,dialogs:0,approve:true,logs:[],quickChoice:null,configured:''};
 const context={subscriptions:[],globalStorageUri:{fsPath:path.join(f.root,'storage')},globalState:{get:k=>scenario.stored.get(k),async update(k,v){if(v===undefined)scenario.stored.delete(k);else scenario.stored.set(k,v);}}};
 const app=new VoderExtension(context);app._agentRootForChat=async()=>path.join(f.root,'voder-agent');return app;
}
const calc='修复已有calc.py中的add函数，使test_calc.py的3个断言通过；只能修改calc.py，不修改测试脚本。';
test('real extension auto-authorizes with one scoped modal approval and persists profile without file picker',async()=>{
 const f=await repository();try{
  const app=newApp(f),profile=await app.ensureNativeChat(f.folder,{goal:calc});
  assert.equal(profile.target,'calc.py');assert.equal(scenario.creations,1);assert.equal(scenario.approvals,1);
  assert.equal(scenario.picks,0);assert.equal(scenario.dialogs,0);
  assert.equal(scenario.stored.get(app._chatProfileKey(f.folder)),profile.path);
  assert.equal((await app.ensureNativeChat(f.folder,{goal:calc})).path,profile.path);
  assert.equal(scenario.creations,1);assert.equal(scenario.approvals,1);
 }finally{await f.finish();}
});
test('different user goal cannot reuse old target authorization; each new target needs explicit approval',async()=>{
 const f=await repository();try{
  const app=newApp(f);const key=app._chatProfileKey(f.folder),first=await app.ensureNativeChat(f.folder,{goal:calc});
  scenario.approve=false;
  await assert.rejects(app.ensureNativeChat(f.folder,{goal:'修复 other.py，由 test_other.py 验证'}),e=>e.code==='CHAT_NATIVE_NOT_APPROVED');
  assert.equal(scenario.stored.get(key),first.path);
  assert.equal(scenario.creations,2);assert.equal(scenario.approvals,2);
  scenario.approve=true;
  const next=await app.ensureNativeChat(f.folder,{goal:'修复 other.py，由 test_other.py 验证'});
  assert.equal(next.target,'other.py');assert.notEqual(next.path,first.path);
  assert.equal(scenario.stored.get(key),next.path);
 }finally{await f.finish();}
});
test('explicit test change cannot reuse approval even for same source',async()=>{
 const f=await repository();try{
  await fsp.writeFile(path.join(f.root,'test_alternative.py'),'pass\n');git(f.root,'add','test_alternative.py');
  const app=newApp(f);await app.ensureNativeChat(f.folder,{goal:calc});
  await app.ensureNativeChat(f.folder,{goal:'修复 calc.py，使 test_alternative.py 通过'});
  assert.equal(scenario.creations,2);
 }finally{await f.finish();}
});
test('unnamed new goal cannot silently reuse a previous file authorization',async()=>{
 const f=await repository();try{
  const app=newApp(f);const original=await app.ensureNativeChat(f.folder,{goal:calc});
  await assert.rejects(app.ensureNativeChat(f.folder,{goal:'请检查项目并做一些修改'}),e=>e.code==='CHAT_SCOPE_INPUT_REQUIRED');
  assert.equal(scenario.stored.get(app._chatProfileKey(f.folder)),original.path);
  assert.equal(scenario.picks,0);
 }finally{await f.finish();}
});
test('configure native profile cancel preserves old approval instead of deleting it',async()=>{
 const f=await repository();try{
  const app=newApp(f),previous=await app.ensureNativeChat(f.folder,{goal:calc});const key=app._chatProfileKey(f.folder);
  scenario.quickChoice='cancel';app._folder=async()=>f.folder;
  await assert.rejects(app.configureNativeChat(),e=>e.code==='CHAT_NATIVE_NOT_APPROVED');
  assert.equal(scenario.stored.get(key),previous.path);
 }finally{await f.finish();}
});
test('invalid cached profile never permits tool execution or skips approval',async()=>{
 const f=await repository();try{
  const app=newApp(f);scenario.stored.set(app._chatProfileKey(f.folder),'invalid');scenario.approve=false;
  await assert.rejects(app.ensureNativeChat(f.folder,{goal:calc}),e=>e.code==='CHAT_NATIVE_NOT_APPROVED');
  assert.equal(scenario.stored.get(app._chatProfileKey(f.folder)),'invalid');
  assert.ok(scenario.logs.some(x=>x.includes('Saved profile blocked')));
 }finally{await f.finish();}
});
test('webview passes exact user goal into native authorization before SDK session start',()=>{
 const source=fs.readFileSync(path.resolve(__dirname,'../src/chat-view.cjs'),'utf8');
 assert.match(source,/ensureNativeChat\(folder,\{goal:msg\.goal,requestApproval:/);
 const extension=fs.readFileSync(filename,'utf8');
 assert.match(extension,/_runChatParticipant\(request,context,stream,token\)/);
 assert.match(extension,/ensureNativeChat\(folder,\{goal\}\)/);
 assert.match(extension,/_runChatParticipant\(request,context,stream,token\)/g);
});
test('end-to-end chat-view submit -> real extension authorization -> bounded SDK start (mock transport)',async()=>{
 const f=await repository();try{
  const app=newApp(f);app.generalTasks=null; // Isolate the retained legacy native-profile API; ordinary chat has its own executing-session tests.
  let dispatched=[];
  app._trackChatSession=()=>{};
  app.chatRuns={start:async(folder,goal,notify,opts)=>{
   dispatched.push({folder,goal,opts});
   const id='796b77df-06d4-419a-af18-2d4b1ba9010a';
   notify({kind:'accepted',sessionId:id});notify({kind:'settled',sessionId:id,status:'BLOCKED',verified:false});
   return {sessionId:id,completion:Promise.resolve({status:'BLOCKED',verified:false})};
  }};
  app.chooseFolder=async()=>f.folder;
  const {VoderChatViewProvider}=require('../src/chat-view.cjs');const provider=new TestChatViewProvider(app);
  const events=[],view={webview:{cspSource:'vscode-resource:',options:{},postMessage(m){events.push(m);if(m.type==='approval_requested')queueMicrotask(()=>provider.handle(view,{type:'authorization_decision',requestId:m.requestId,decision:'approve'}));},onDidReceiveMessage(){}},onDidDispose(){}};
  provider.resolveWebviewView(view);
  await provider.handle(view,{type:'submit',goal:calc});
  assert.equal(dispatched.length,1);assert.equal(dispatched[0].opts.operatorTarget,'calc.py');
  assert.equal(dispatched[0].opts.requiredFiles[0],'calc.py');
  assert.equal(scenario.approvals,0);assert.equal(scenario.picks,0);
  await provider.handle(view,{type:'submit',goal:calc});
  assert.equal(dispatched.length,2);assert.equal(scenario.approvals,0);
  assert.ok(events.some(m=>m.type==='accepted'));
 }finally{await f.finish();}
});
test('declined authorization blocks before SDK session, without deleting previous authorization',async()=>{
 const f=await repository();try{
  const app=newApp(f),older=await app.ensureNativeChat(f.folder,{goal:calc});
  app.generalTasks=null; // This test exercises explicit legacy profile authorization.
  scenario.approve=false;let launches=0;app.chooseFolder=async()=>f.folder;
  app.chatRuns={start:async()=>{launches++;throw Error('SHOULD_NOT_RUN');}};
  const {VoderChatViewProvider}=require('../src/chat-view.cjs'),provider=new TestChatViewProvider(app);
  const view={webview:{cspSource:'vscode-resource:',options:{},postMessage(m){if(m.type==='approval_requested')queueMicrotask(()=>provider.handle(view,{type:'authorization_decision',requestId:m.requestId,decision:'decline'}));},onDidReceiveMessage(){}},onDidDispose(){}};
  provider.resolveWebviewView(view);
  await provider.handle(view,{type:'submit',goal:'修复 other.py，使 test_other.py 通过'});
  assert.equal(launches,0);
  assert.equal(scenario.stored.get(app._chatProfileKey(f.folder)),older.path);
 }finally{await f.finish();}
});

async function stateView(app){const {VoderChatViewProvider}=require('../src/chat-view.cjs');const provider=new TestChatViewProvider(app,{entryPlan:()=>({kind:'native',wantsWrite:true})}),events=[];let dispose;const view={webview:{cspSource:'vscode-resource:',options:{},postMessage:m=>events.push(m),onDidReceiveMessage(){}},onDidDispose:f=>{dispose=f;}};provider.resolveWebviewView(view);return {provider,view,events,dispose:()=>dispose()};}
for(const [label,goal,code] of [['placeholder','修复实际源码文件名.py，使实际测试文件名.py通过','CHAT_SCOPE_PLACEHOLDER'],['missing scope','请修复项目中的问题','CHAT_SCOPE_INPUT_REQUIRED'],['untracked path','修复missing_file.py','CHAT_SCOPE_FILE_NOT_TRACKED']])test('preparation '+label+' waits for input with zero dialogs, profiles and SDK calls',async()=>{const f=await repository();try{const app=newApp(f);app.chooseFolder=async()=>f.folder;let launches=0;app.chatRuns={start:async()=>launches++};const {provider,view,events}=await stateView(app);await provider.handle(view,{type:'submit',goal});const notice=events.find(m=>m.type==='input_required');assert.equal(notice.code,code);assert.equal(notice.goal,goal);assert.deepEqual(notice.choices,[]);assert.equal(notice.scopeContext.workspace,f.folder.uri.fsPath);assert.equal(launches,0);assert.equal(scenario.creations,0);assert.equal(scenario.picks+scenario.dialogs+scenario.approvals,0);assert.equal(events.some(m=>m.type==='error'),false);}finally{await f.finish();}});
test('inline approval requires exact pending token; stale messages cannot persist or start',async()=>{const f=await repository();try{const app=newApp(f);app.chooseFolder=async()=>f.folder;let launches=0;app._trackChatSession=()=>{};app.chatRuns={start:async()=>{launches++;return {sessionId:'796b77df-06d4-419a-af18-2d4b1ba9010a',completion:Promise.resolve({status:'BLOCKED'})};}};const {provider,view,events}=await stateView(app);const turn=provider.handle(view,{type:'submit',goal:calc});await new Promise(r=>setImmediate(r));const request=events.find(m=>m.type==='approval_requested');assert.ok(request);assert.equal(launches,0);await provider.handle(view,{type:'authorization_decision',requestId:'stale',decision:'approve'});assert.equal(launches,0);assert.equal(scenario.stored.get(app._chatProfileKey(f.folder)),undefined);await provider.handle(view,{type:'authorization_decision',requestId:request.requestId,decision:'approve'});await turn;assert.equal(launches,1);assert.ok(scenario.stored.get(app._chatProfileKey(f.folder)));assert.equal(scenario.approvals+scenario.picks+scenario.dialogs,0);}finally{await f.finish();}});
test('explicit inline decline is NOT_STARTED and preserves prior approval',async()=>{const f=await repository();try{const app=newApp(f),prior=await app.ensureNativeChat(f.folder,{goal:calc});app.chooseFolder=async()=>f.folder;let launches=0;app.chatRuns={start:async()=>launches++};const {provider,view,events}=await stateView(app);const turn=provider.handle(view,{type:'submit',goal:'修复other.py，使test_other.py通过'});await new Promise(r=>setImmediate(r));const request=events.find(m=>m.type==='approval_requested');await provider.handle(view,{type:'authorization_decision',requestId:request.requestId,decision:'decline'});await turn;assert.equal(launches,0);assert.ok(events.some(m=>m.type==='not_started'&&m.status==='NOT_STARTED'));assert.equal(events.some(m=>m.type==='error'||m.type==='uncertain'),false);assert.equal(scenario.stored.get(app._chatProfileKey(f.folder)),prior.path);}finally{await f.finish();}});
test('view disposal before approval cannot persist authorization or launch SDK',async()=>{const f=await repository();try{const app=newApp(f);app.chooseFolder=async()=>f.folder;let launches=0;app.chatRuns={start:async()=>launches++};const {provider,view,dispose}=await stateView(app);const turn=provider.handle(view,{type:'submit',goal:calc});await new Promise(r=>setImmediate(r));dispose();await turn;assert.equal(launches,0);assert.equal(provider.approvals.size,0);assert.equal(scenario.stored.get(app._chatProfileKey(f.folder)),undefined);}finally{await f.finish();}});
test('webview readiness restores pending approval without approving or dispatching',async()=>{const f=await repository();try{const app=newApp(f);app.chooseFolder=async()=>f.folder;app.chatRuns={start:async()=>{throw Error('must not start')}};const {provider,view,events}=await stateView(app);const turn=provider.handle(view,{type:'submit',goal:calc});await new Promise(r=>setImmediate(r));const request=events.find(m=>m.type==='approval_requested');await provider.handle(view,{type:'ready'});const repeats=events.filter(m=>m.type==='approval_requested');assert.equal(repeats.length,2);assert.equal(repeats[1].requestId,request.requestId);await provider.handle(view,{type:'authorization_decision',requestId:request.requestId,decision:'decline'});await turn;}finally{await f.finish();}});
test('preparation block is distinct from runtime uncertainty and leaves pending state clear',async()=>{const f=await repository();try{const app=newApp(f);app.chooseFolder=async()=>f.folder;app.ensureNativeChat=async()=>{throw Object.assign(Error('dirty source'),{code:'CHAT_FILE_NOT_TRACKED'})};app.chatRuns={start:async()=>{throw Error('must not run')}};const {provider,view,events}=await stateView(app);await provider.handle(view,{type:'submit',goal:calc});assert.ok(events.some(m=>m.type==='not_started'&&m.status==='START_BLOCKED'));assert.equal(events.some(m=>m.type==='uncertain'),false);assert.equal(provider.pending.size,0);}finally{await f.finish();}});
