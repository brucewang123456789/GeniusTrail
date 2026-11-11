'use strict';
const {TestChatViewProvider}=require('./chat-test-context.cjs');
const test=require('node:test'),assert=require('node:assert/strict');
const {EventEmitter}=require('node:events');
const {ChatRunService,validateGoal,safeUpdate,safeResult}=require('../src/chat-core.cjs');
const {chatHtml}=require('../src/chat-view.cjs');
const folder={uri:{fsPath:'/trusted/source'}};
const ID1='ca29c10d-46f3-430a-83be-5b4875d22aba',ID2='ca29c10d-46f3-430a-83be-5b4875d22abb';
class FakeRPC extends EventEmitter{
 constructor(){super();this.calls=[];this.ids=[ID1,ID2];this.waiters=[];}
 async request(method,params){this.calls.push({method,params});
  if(method==='session.create')return {sessionId:this.ids.shift()};
  if(method==='session.subscribe')return {status:'SUBSCRIBED'};
  if(method==='task.submit')return {status:'ACCEPTED'};
  if(method==='task.wait')return new Promise((resolve,reject)=>this.waiters.push({resolve,reject}));
  if(method==='session.cancel')return {status:'CANCELLATION_REQUESTED'};
  throw Error('UNEXPECTED_METHOD '+method);
 }
}
test('chat uses real SDK RPC, a new session for every independently verified turn',async()=>{
 const rpc=new FakeRPC(),service=new ChatRunService(async()=>rpc),updates=[];
 const first=await service.start(folder,'Fix a bug',u=>updates.push(u));
 assert.equal(first.sessionId,ID1);
 assert.deepEqual(rpc.calls.slice(0,3).map(x=>x.method),['session.create','session.subscribe','task.submit']);
 assert.deepEqual(rpc.calls[2].params.task,{goal:'Fix a bug',maxActions:1});
 await assert.rejects(service.start(folder,'Second turn'),e=>e.code==='CHAT_BUSY');
 rpc.emit('event',{sessionId:ID1,event:{type:'vgr.action_settled',status:'COMMITTED',tool:'safe',description:'secret'}});
 assert.equal(updates.find(x=>x.kind==='event').event.description,undefined);
 rpc.waiters[0].resolve({status:'DELIVERED',verified:true,commit_digest:'a'.repeat(64)});
 const settled=await first.completion;assert.equal(settled.verified,true);assert.equal(settled.commitDigest,'a'.repeat(64));
 assert.equal(service.activeSession(folder),null);
 const second=await service.start(folder,'Add tests');assert.equal(second.sessionId,ID2);
 rpc.waiters[1].resolve({status:'BLOCKED',verified:false});assert.equal((await second.completion).verified,false);
});
test('explicit cancellation is scoped to an exact running session, no auto-replay',async()=>{
 const rpc=new FakeRPC(),service=new ChatRunService(async()=>rpc),first=await service.start(folder,'Run task');
 await assert.rejects(service.cancel(folder,ID2),e=>e.code==='CHAT_NOT_ACTIVE');
 const ack=await service.cancel(folder,ID1);assert.equal(ack.status,'CANCELLATION_REQUESTED');
 assert.equal(rpc.calls.filter(c=>c.method==='session.cancel').length,1);
 rpc.waiters[0].resolve({status:'CANCELLED',verified:false});await first.completion;
 await assert.rejects(service.cancel(folder,ID1),e=>e.code==='CHAT_NOT_ACTIVE');
});
test('uncertain dispatch and connection failure cannot be reported as delivered',async()=>{
 const rpc=new FakeRPC(),service=new ChatRunService(async()=>rpc),updates=[];
 const task=await service.start(folder,'Run',u=>updates.push(u));
 rpc.waiters[0].reject(Object.assign(Error('connection loss'),{code:'RPC_TIMEOUT'}));
 await assert.rejects(task.completion,e=>e.code==='RPC_TIMEOUT');
 assert.equal(service.activeSession(folder),null);
 assert.equal(updates.at(-1).kind,'uncertain');
 assert.equal(updates.at(-1).code,'RPC_TIMEOUT');
});
test('guard input size and reject invalid UI-injected data',()=>{
 assert.throws(()=>validateGoal(' '),/nonempty/);
 assert.throws(()=>validateGoal('x'.repeat(16385)),/16 KiB/);
 assert.throws(()=>validateGoal('x\u0000'),/nonempty/);
 assert.equal(safeUpdate({type:'model.tool_execution_end',code:'DENIED',raw:'API_KEY=secret'}).raw,undefined);
 assert.equal(safeUpdate({type:'unknown'}),null);
 assert.equal(safeResult({status:'DELIVERED',verified:false},ID1).verified,false);
 assert.equal(safeResult({status:'DRAFT_UNVERIFIED',verified:false,draft:'text'},ID1).draft,'text');
});
test('webview has restrictive CSP and no HTML insertion of runtime tool/model content',()=>{
 const html=chatHtml({cspSource:'vscode-webview-resource:'});
 assert.match(html,/Content-Security-Policy/);
 assert.match(html,/default-src 'none'/);
 assert.match(html,/textContent=m\.draft/);
 assert.doesNotMatch(html,/innerHTML\s*=/);
 assert.match(html,/Ready · one task at a time/);
});

test('real CLI/SDK chat turn: missing required file yields WAITING_FOR_INPUT, no model/native',async()=>{
 const {RpcClient}=require('../src/rpc-client.cjs'),fs=require('node:fs/promises'),path=require('node:path'),os=require('node:os');
 const dir=await fs.mkdtemp(path.join(os.tmpdir(),'voder-chat-sdk-'));
 const rpc=new RpcClient({command:process.execPath,args:[path.join(process.env.VODER_AGENT_TEST_ROOT,'packages/voder-cli/bin/voder-cli.mjs'),'rpc','--draft','--workspace',dir,'--session-dir',path.join(dir,'journal')],cwd:dir});
 try{
  const init=await rpc.request('initialize');assert.equal(init.capabilities.vgr_commit,'native_only');
  const service=new ChatRunService(async()=>rpc),tasks=[];
  const task=await service.start({uri:{fsPath:dir}},'Read the required file before proposing any changes',u=>tasks.push(u),{requiredFiles:['missing_chat_file_that_does_not_exist.txt']});
  const result=await task.completion;
  assert.equal(result.status,'WAITING_FOR_INPUT');assert.equal(result.verified,false);
  assert.equal(tasks.some(u=>u.kind==='accepted'),true);
  assert.equal(tasks.some(u=>u.kind==='settled'),true);
  const events=await rpc.request('session.events',{sessionId:task.sessionId,after:0});
  assert.ok(Array.isArray(events));
  const diff=await rpc.request('session.diff',{sessionId:task.sessionId});assert.equal(diff.status,'NOT_AVAILABLE');
 }finally{rpc.close();await fs.rm(dir,{recursive:true,force:true});}
});

test('webview provider routes only bounded submit, exact cancellation and verified identity actions',async()=>{
 const {VoderChatViewProvider}=require('../src/chat-view.cjs');
 const sent=[],listeners={},run=new FakeRPC();let completion;
 const app={chatSessionFolder:new Map(),trusted:()=>true,chooseFolder:async()=>folder,ensureNativeChat:async()=>({path:'/private/gateway.json',target:'src/index.mjs'}),
  chatRuns:{start:async(f,goal,notify)=>{assert.equal(goal,'Fix tests');notify({kind:'accepted',sessionId:ID1});completion=new Promise(resolve=>listeners.settle=resolve);return {sessionId:ID1,completion};},cancel:async(f,id)=>{assert.equal(id,ID1);return {status:'CANCELLATION_REQUESTED'};}},
  _trackChatSession(id,f){this.chatSessionFolder.set(id,f);},inspectDiff:async ({id})=>{assert.equal(id,ID1);sent.push({type:'diff'});},showStatus:async ({id})=>{assert.equal(id,ID1);sent.push({type:'status'});}};
 const provider=new TestChatViewProvider(app,{entryPlan:()=>({kind:'native',wantsWrite:true})}),view={webview:{cspSource:'vscode-resource:',onDidReceiveMessage(fn){listeners.message=fn;},postMessage(m){sent.push(m);},options:{}},onDidDispose(fn){listeners.dispose=fn;}};
 provider.resolveWebviewView(view);
 assert.match(view.webview.html,/Voder Chat/);
 const doing=provider.handle(view,{type:'submit',goal:'Fix tests'});
 await new Promise(r=>setImmediate(r));
 assert.equal(sent.some(x=>x.type==='accepted'),true);
 await provider.handle(view,{type:'cancel',sessionId:ID1});assert.equal(sent.some(x=>x.type==='cancel_ack'),true);
 await provider.handle(view,{type:'action',command:'diff',sessionId:ID1});assert.equal(sent.some(x=>x.type==='diff'),true);
 await provider.handle(view,{type:'action',command:'anything',sessionId:ID1});assert.equal(sent.some(x=>x.type==='anything'),false);
 await assert.rejects(provider.handle(view,{type:'cancel',sessionId:ID2}),e=>e.code==='CHAT_SESSION_MISMATCH');
 listeners.settle({status:'DELIVERED'});await doing;
 listeners.dispose();assert.equal(provider.views.size,0);
});

test('follow-up chat preserves at most 3 prior user goals as nonexecuting context, within CLI cap',()=>{
 const {scopedGoal}=require('../src/chat-core.cjs');
 const h=[{goal:'Old instruction 1',status:'DELIVERED'},{goal:'Old instruction 2',status:'BLOCKED'},{goal:'Old instruction 3',status:'FAILED'},{goal:'Old instruction 4',status:'DELIVERED'}];
 const goal=scopedGoal('Now fix the new bug',h);
 assert.ok(goal.includes('CURRENT USER TASK'));
 assert.ok(goal.includes('Now fix the new bug'));
 assert.ok(!goal.includes('Old instruction 1'));
 assert.ok(goal.includes('Old instruction 4'));
 assert.ok(Buffer.byteLength(goal,'utf8')<=16384);
 assert.equal(scopedGoal('a'.repeat(16380),h),'a'.repeat(16380));
});

test('task.submit RPC timeout is classified uncertain, never as a safe rejection',async()=>{
 const rpc=new FakeRPC();rpc.request=async method=>{
  if(method==='session.create')return {sessionId:ID1};
  if(method==='session.subscribe')return {status:'SUBSCRIBED'};
  if(method==='task.submit')throw Object.assign(Error('No response'),{code:'RPC_TIMEOUT'});
 };
 const updates=[],service=new ChatRunService(async()=>rpc);
 await assert.rejects(service.start(folder,'fix',u=>updates.push(u)),e=>e.code==='RPC_TIMEOUT');
 assert.ok(updates.some(x=>x.kind==='created'&&x.sessionId===ID1));
 assert.ok(updates.some(x=>x.kind==='uncertain'&&x.sessionId===ID1));
 assert.ok(!updates.some(x=>x.kind==='settled'));
});
