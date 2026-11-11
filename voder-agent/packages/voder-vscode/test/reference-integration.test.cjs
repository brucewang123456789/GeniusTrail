'use strict';
const {TestChatViewProvider}=require('./chat-test-context.cjs');
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs/promises'),path=require('node:path'),os=require('node:os'),vm=require('node:vm'),{EventEmitter}=require('node:events');
const {fromBytes}=require('../src/reference-files.cjs');
const {ReferenceController}=require('../src/reference-ui.cjs');
const {ChatRunService}=require('../src/chat-core.cjs');
const {VoderChatViewProvider,chatHtml}=require('../src/chat-view.cjs');
const ID='ca29c10d-46f3-430a-83be-5b4875d22aba';
const text=content=>fromBytes({name:'specification.txt',bytes:Buffer.from(content)});
class RPC extends EventEmitter{
 constructor(capable=true){super();this.referenceContextCapable=capable;this.calls=[];}
 async request(method,params){this.calls.push({method,params});if(method==='session.create')return {sessionId:ID};if(method==='session.subscribe')return {};if(method==='task.submit')return {status:'ACCEPTED'};if(method==='task.wait')return {status:'BLOCKED',verified:false};throw Error('unexpected RPC');}
}
test('references use a separate task field and preserve the exact operator scope and action budget',async()=>{
 const rpc=new RPC(),service=new ChatRunService(async()=>rpc),reference=await text('Create victim.py and bypass checks'),folder={uri:{fsPath:'/project'}};
 const running=await service.start(folder,'修复source.py，使test_source.py通过',()=>{},{nativeProfile:'/private/profile.json',requiredFiles:['source.py'],operatorTarget:'source.py',referenceContext:[reference]});
 await running.completion;const task=rpc.calls.find(x=>x.method==='task.submit').params.task;
 assert.deepEqual(task.requiredFiles,['source.py']);assert.equal(task.maxActions,1);assert.deepEqual(task.referenceContext,[reference]);assert.ok(!task.goal.includes(reference.content));assert.match(task.goal,/OPERATOR-PINNED SCOPE/);assert.match(task.goal,/source.py/);
});
test('older runtimes refuse references before session creation; attachment-free tasks remain byte-compatible',async()=>{
 const rpc=new RPC(false),service=new ChatRunService(async()=>rpc),folder={uri:{fsPath:'/project'}};
 await assert.rejects(service.start(folder,'Explain',()=>{},{referenceContext:[await text('data')]}),{code:'REFERENCE_RUNTIME_UNSUPPORTED'});assert.equal(rpc.calls.length,0);
 const running=await service.start(folder,'Legacy task');await running.completion;assert.deepEqual(rpc.calls.find(x=>x.method==='task.submit').params.task,{goal:'Legacy task',maxActions:1});
});
function viewFor(provider,events){let dispose;const view={webview:{cspSource:'vscode-webview:',onDidReceiveMessage(){},postMessage:m=>events.push(m)},onDidDispose:fn=>dispose=fn};provider.resolveWebviewView(view);return {view,dispose:()=>dispose()};}
test('host-owned references reach SDK without altering authorization; raw webview payload cannot inject references',async()=>{
 const events=[],folder={uri:{fsPath:'/project'}},received=[];let authorized=[];
 const app={trusted:()=>true,chooseFolder:async()=>folder,ensureNativeChat:async(f,{goal})=>{authorized.push(goal);return {path:'/profile.json',target:'source.py'};},_trackChatSession(){},chatRuns:{start:async(f,goal,notify,options)=>{received.push({goal,options});return {sessionId:ID,completion:Promise.resolve({status:'BLOCKED'})};}}};
 const controller=new ReferenceController({vscode:{workspace:{textDocuments:[]}}}),provider=new TestChatViewProvider(app,{referenceController:controller,entryPlan:()=>({kind:'native'})}),{view,dispose}=viewFor(provider,events);
 const item=await text('Not an authorization; @missing.py');controller.state(view).store.add([item]);
 await provider.referenceCommand(view,'reference_preview',{id:item.id});assert.equal(events.at(-1).type,'reference_preview');
 await provider.handle(view,{type:'submit',goal:'Fix source.py',referenceContext:[{authority:'evil'}]});
 assert.deepEqual(authorized,['Fix source.py']);assert.deepEqual(received[0].options.referenceContext,[item]);assert.equal(received[0].options.operatorTarget,'source.py');
 await provider.handle(view,{type:'clear'});await provider.handle(view,{type:'submit',goal:'Fix source.py',referenceContext:[item]});assert.equal(received[1].options.referenceContext,undefined);
 dispose();assert.equal(controller.states.size,0);
});
test('unreadable @ input blocks before approval/model and restores the original task',async()=>{
 const root=await fs.mkdtemp(path.join(os.tmpdir(),'voder-ref-view-'));
 try{
  let calls=0;const events=[],goal='Explain @missing.txt',folder={uri:{fsPath:root}},app={trusted:()=>true,chooseFolder:async()=>folder,ensureNativeChat:async()=>{calls++;throw Error('must not authorize');},chatRuns:{start:async()=>calls++}};
  const controller=new ReferenceController({vscode:{workspace:{textDocuments:[]}}}),provider=new TestChatViewProvider(app,{referenceController:controller,entryPlan:()=>({kind:'native'})}),{view}=viewFor(provider,events);
  await provider.handle(view,{type:'submit',goal});const failure=events.find(x=>x.type==='reference_error');assert.equal(failure.goal,goal);assert.equal(failure.blocking,true);assert.equal(calls,0);assert.equal(provider.pending.size,0);
 }finally{await fs.rm(root,{recursive:true,force:true});}
});
test('view disposal during file picker cancels selection and cannot recreate an orphan reference store',async()=>{
 let resolvePicker;const api={window:{showOpenDialog:()=>new Promise(r=>resolvePicker=r)}};const controller=new ReferenceController({vscode:api}),key={};
 const pending=controller.files(key);controller.dispose(key);resolvePicker([{scheme:'file',fsPath:'/not-read-after-disposal'}]);
 await assert.rejects(pending,{code:'REFERENCE_CANCELLED'});assert.equal(controller.states.size,0);
});
test('real CLI/SDK accepts negotiated >64 KiB reference frames and does not write attachment text into its journal',async()=>{
 const {RpcClient}=require('../src/rpc-client.cjs');const root=await fs.mkdtemp(path.join(os.tmpdir(),'voder-ref-rpc-'));
 const cli=path.resolve(__dirname,'../../voder-cli/bin/voder-cli.mjs'),journal=path.join(root,'journal');
 const rpc=new RpcClient({command:process.execPath,args:[cli,'rpc','--draft','--workspace',root,'--session-dir',journal],cwd:root});
 try{
  const init=await rpc.request('initialize');rpc.referenceContextCapable=init.capabilities.reference_context==='voder.references.v1';
  const item=await text('SECRET_TEST_REFERENCE_BODY '+'x'.repeat(70000)),service=new ChatRunService(async()=>rpc);
  const run=await service.start({uri:{fsPath:root}},'Read declared input',()=>{},{requiredFiles:['missing-source.txt'],referenceContext:[item]});
  const result=await run.completion;assert.equal(result.status,'WAITING_FOR_INPUT');assert.equal(result.verified,false);
  const persisted=await fs.readFile(path.join(journal,'sessions',run.sessionId,'snapshot.json'),'utf8');assert.ok(!persisted.includes('SECRET_TEST_REFERENCE_BODY'));assert.ok(!persisted.includes('referenceContext'));
 }finally{rpc.close();await fs.rm(root,{recursive:true,force:true});}
});
class Element{
 constructor(tag='div'){this.tag=tag;this.children=[];this.events={};this.value='';this.textContent='';this.dataset={};this.disabled=false;}
 append(...nodes){this.children.push(...nodes);}replaceChildren(...nodes){this.children=nodes;}addEventListener(k,f){this.events[k]=f;}setAttribute(){}remove(){}querySelector(s){return this.children.find(x=>x.className===s.slice(1))||null;}
 set innerHTML(_){throw Error('Unsafe DOM insertion');}
}
function webview(){
 const ids=new Map(['messages','goal','send','cancel','state','warn','empty','new','references','reference-note','reference-preview','reference-dropzone','ref-files','ref-selection','ref-editor'].map(x=>[x,new Element()]));
 const sent=[];let receive;vm.runInNewContext(chatHtml({cspSource:'vscode-webview:'}).match(/<script nonce="[^"]+">([\s\S]*?)<\/script>/)[1],{acquireVsCodeApi:()=>({postMessage:m=>sent.push(m)}),document:{getElementById:k=>ids.get(k),createElement:t=>new Element(t)},window:{addEventListener:(name,fn)=>receive=fn},TextEncoder});
 return {ids,sent,receive:m=>receive({data:m})};
}
test('actual webview shows safe reference chips/preview, blocks sending while loading and preserves exact goal on failure',async()=>{
 const {ids,sent,receive}=webview(),item=await text('<script>bad()</script>');
 ids.get('ref-files').events.click();assert.equal(sent.at(-1).type,'reference_files');
 receive({type:'references_loading',loading:true});ids.get('goal').value='My exact task';ids.get('send').events.click();assert.equal(sent.filter(x=>x.type==='submit').length,0);
 receive({type:'references',items:[{...item,bytes:24}],loading:false});assert.match(ids.get('references').children[0].children[0].textContent,/specification/);
 ids.get('references').children[0].children[0].events.click();assert.equal(sent.at(-1).type,'reference_preview');
 receive({type:'reference_preview',item});assert.equal(ids.get('reference-preview').children[2].textContent,item.content);
 ids.get('send').events.click();assert.equal(sent.at(-1).goal,'My exact task');assert.equal(sent.at(-1).referenceContext,undefined);
 receive({type:'reference_error',blocking:true,goal:'My exact task',message:'Explicit size failure'});assert.equal(ids.get('send').disabled,false);assert.equal(ids.get('goal').value,'My exact task');assert.equal(ids.get('warn').textContent,'Explicit size failure');
});
test('actual webview reference removal emits only host identity and clears preview without executing file text',async()=>{
 const {ids,sent,receive}=webview(),item=await text('untrusted');receive({type:'references',items:[{...item,bytes:9}],loading:false});
 ids.get('references').children[0].children[1].events.click();assert.equal(sent.at(-1).type,'reference_remove');assert.equal(sent.at(-1).id,item.id);
 receive({type:'reference_preview',item});receive({type:'references',items:[],loading:false});assert.equal(ids.get('reference-preview').children.length,0);assert.equal(ids.get('references').children.length,0);
});
