'use strict';
const {TestChatViewProvider}=require('./chat-test-context.cjs');
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs/promises'),os=require('node:os'),path=require('node:path'),{execFileSync}=require('node:child_process');
const {VoderChatViewProvider}=require('../src/chat-view.cjs');
test('actual provider: investigation -> exact inline approval -> one native start; decline and wrong token never execute',async()=>{
 const root=await fs.mkdtemp(path.join(os.tmpdir(),'voder-ui-handoff-'));
 try{
  await fs.writeFile(path.join(root,'calc.py'),'def add(a,b): return a-b\n');await fs.writeFile(path.join(root,'test_calc.py'),'from calc import add\nassert add(2,3)==5\n');
  const git=(...a)=>execFileSync('git',['-C',root,...a],{stdio:'pipe'});git('init','-q');git('add','.');git('-c','user.name=fixture','-c','user.email=fixture@invalid','commit','-qm','baseline');
  for(const decision of ['approve','decline']){
   const folder={uri:{fsPath:root}},order=[],messages=[];let starts=0,request;let announce;const waiting=new Promise(r=>announce=r);
   const app={trusted:()=>true,chooseFolder:async()=>folder,_trackChatSession(){},ensureNativeChat:async(f,{goal,requestApproval})=>{order.push('authorization');assert.match(goal,/修复当前项目中的加法问题/);const allowed=await requestApproval({target:'calc.py',validator:'test_calc.py'});if(!allowed)throw Object.assign(Error('declined'),{code:'CHAT_AUTH_CANCELLED'});return {path:'/test-only/profile.json',target:'calc.py'};},chatRuns:{inspect:async()=>{order.push('inspect');return {sessionId:'inspection',completion:Promise.resolve({status:'DRAFT_UNVERIFIED',verified:false,draft:JSON.stringify({answer:'加法实现错误；未运行测试。',scope:{target:'calc.py',test:'test_calc.py'}})})};},start:async(f,goal,notify,options)=>{starts++;order.push('native');assert.equal(options.nativeProfile,'/test-only/profile.json');assert.equal(options.operatorTarget,'calc.py');return {sessionId:'native',completion:Promise.resolve({status:'NOT_STARTED',verified:false})};}}};
   const provider=new TestChatViewProvider(app);const view={webview:{cspSource:'vscode-webview:',onDidReceiveMessage(){},postMessage:m=>{messages.push(m);if(m.type==='approval_requested'){request=m;announce();}}},onDidDispose(){}};provider.resolveWebviewView(view);
   const running=provider.handle(view,{type:'submit',goal:'修复当前项目中的加法问题，禁止修改测试'});await waiting;assert.equal(starts,0);assert.equal(messages.find(m=>m.type==='inspection_complete').continuing,true);
   await provider.handle(view,{type:'authorization_decision',requestId:'wrong-token',decision:'approve'});assert.equal(starts,0);assert.ok(provider.approvals.has(view));
   await provider.handle(view,{type:'authorization_decision',requestId:request.requestId,decision});await running;assert.equal(starts,decision==='approve'?1:0);assert.deepEqual(order,decision==='approve'?['inspect','authorization','native']:['inspect','authorization']);assert.equal(provider.pending.size,0);
  }
  assert.equal(await fs.readFile(path.join(root,'calc.py'),'utf8'),'def add(a,b): return a-b\n');
 }finally{await fs.rm(root,{recursive:true,force:true});}
});
