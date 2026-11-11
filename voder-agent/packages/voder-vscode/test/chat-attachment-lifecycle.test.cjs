'use strict';
const {TestChatViewProvider}=require('./chat-test-context.cjs');
const test=require('node:test'),assert=require('node:assert/strict');
const {VoderChatViewProvider,chatHtml}=require('../src/chat-view.cjs');
const {ReferenceController}=require('../src/reference-ui.cjs');
const {fromBytes}=require('../src/reference-files.cjs');
const vm=require('node:vm'),fs=require('node:fs'),path=require('node:path');
const fixture=name=>fs.readFileSync(path.join(__dirname,'fixtures','documents',name));
const ID='a19f5a10-c03d-4bce-9188-7cc419a55123';
const folder={uri:{fsPath:'/tmp/voder-attachment-lifecycle-test-root'}};
const bytes=s=>Buffer.from(s,'utf8');
const item=async(name='test_calc.py')=>fromBytes({name,bytes:bytes('from calc import add\n')});
function instance({startMode='accepted',choose=true,trust=true}={}){
 const events=[],calls=[],controller=new ReferenceController({vscode:{workspace:{textDocuments:[]}}});
 const app={trusted:()=>trust,chatWorkspace:()=>folder,chooseFolder:async()=>choose?folder:null,
 ensureNativeChat:async()=>({path:'/tmp/profile.json',target:'calc.py'}),_trackChatSession(){},
 chatRuns:{start:async(_folder,goal,notify,options)=>{
  calls.push({goal,options});notify({kind:'created',sessionId:ID});
  if(startMode==='throw')throw Object.assign(Error('SDK submission refused'),{code:'CHAT_NOT_ACCEPTED'});
  if(startMode==='accepted')notify({kind:'accepted',sessionId:ID,status:'ACCEPTED'});
  return {sessionId:ID,completion:Promise.resolve({status:'BLOCKED'})};
 }}};
 const provider=new TestChatViewProvider(app,{referenceController:controller,entryPlan:()=>({kind:'native'})});
 const view={webview:{cspSource:'vscode-webview:',onDidReceiveMessage(){},postMessage:e=>{events.push(e);return true;}},onDidDispose(){}};
 provider.resolveWebviewView(view);
 return {app,controller,provider,view,events,calls};
}
const attached=(t,x)=>t.controller.state(t.view).store.add([x]);
const list=t=>t.controller.state(t.view).store.list();

test('accepted real task clears only draft host references, preserves exact dispatched source snapshot',async()=>{
 const t=instance(),x=await item();attached(t,x);
 await t.provider.handle(t.view,{type:'submit',goal:'Analyze calc.py'});
 assert.equal(list(t).length,0);
 assert.equal(t.events.filter(e=>e.type==='references'&&!e.items.length).length,1);
 assert.equal(t.calls.length,1);assert.deepEqual(t.calls[0].options.referenceContext,[x]);
 assert.equal(t.events.filter(e=>e.type==='references_included').length,1);
 const empty=t.events.findIndex(e=>e.type==='references'&&!e.items.length);
 const accepted=t.events.findIndex(e=>e.type==='accepted');assert.ok(accepted>empty);
});
test('created session without acceptance must not discard reference; retry can send exactly same snapshot',async()=>{
 const t=instance({startMode:'throw'}),x=await item();attached(t,x);
 await assert.rejects(t.provider.handle(t.view,{type:'submit',goal:'Analyze calc.py'}),/SDK submission refused/);
 assert.deepEqual(t.controller.state(t.view).store.snapshot(),[x]);
 assert.equal(t.events.some(e=>e.type==='references'&&!e.items.length),false);
 assert.equal(t.provider.pending.has(t.view),false);
 t.app.chatRuns.start=async(_folder,goal,notify,options)=>{t.calls.push({goal,options});notify({kind:'created',sessionId:ID});notify({kind:'accepted',sessionId:ID});return {sessionId:ID,completion:Promise.resolve({status:'BLOCKED'})};};
 await t.provider.handle(t.view,{type:'submit',goal:'Retry analyze calc.py'});
 assert.deepEqual(t.calls.at(-1).options.referenceContext,[x]);assert.equal(list(t).length,0);
});
test('no workspace and untrusted workspace refuse before acceptance without losing draft',async()=>{
 for(const options of [{choose:false},{trust:false}]){
  const t=instance(options),x=await item();attached(t,x);
  await t.provider.handle(t.view,{type:'submit',goal:'Analyze calc.py'});
  assert.equal(t.calls.length,0);assert.equal(list(t).length,1);
  assert.ok(t.events.some(e=>e.type==='not_started'));
 }
});
test('session creation must not clear; only matching accepted session may clear; duplicate accepted idempotent',async()=>{
 const t=instance(),x=await item();attached(t,x);
 const store=t.controller.state(t.view).store;
 t.provider.pending.set(t.view,{sessionId:ID,referenceVersion:store.version,referenceIds:[x.id],referencesConsumed:false});
 t.provider.post(t.view,{type:'created',sessionId:ID});assert.equal(list(t).length,1);
 t.provider.post(t.view,{type:'accepted',sessionId:'wrong'});assert.equal(list(t).length,1);
 t.provider.post(t.view,{type:'accepted',sessionId:ID});assert.equal(list(t).length,0);
 t.provider.post(t.view,{type:'accepted',sessionId:ID});assert.equal(t.events.filter(e=>e.type==='references'&&!e.items.length).length,1);
});
test('unexpected draft mutation is not cleared by stale acceptance',async()=>{
 const t=instance(),x=await item(),y=await item('another.txt');attached(t,x);
 const store=t.controller.state(t.view).store;
 t.provider.pending.set(t.view,{sessionId:ID,referenceVersion:store.version,referenceIds:[x.id],referencesConsumed:false});
 store.add([y]);t.provider.post(t.view,{type:'accepted',sessionId:ID});
 assert.equal(list(t).length,2);
});
test('image snapshot used for accepted request, next request attachment-free',async()=>{
 const t=instance();const png=fixture('embedded.png');
 const x=await fromBytes({name:'sample.png',bytes:png});attached(t,x);
 await t.provider.handle(t.view,{type:'submit',goal:'Describe attached screenshot'});
 assert.equal(t.calls[0].options.referenceContext[0].mimeType,'image/png');
 assert.equal(t.calls[0].options.referenceContext[0].data,x.data);
 assert.equal(list(t).length,0);
 await t.provider.handle(t.view,{type:'submit',goal:'Now answer without previous attachment'});
 assert.equal(t.calls[1].options.referenceContext,undefined);
});
test('text/PDF-derived snapshot stays in dispatched task but not next turn',async()=>{
 const t=instance(),x=await fromBytes({name:'manual.pdf',bytes:fixture('two-pages.pdf')});
 attached(t,x);await t.provider.handle(t.view,{type:'submit',goal:'Summarize pdf'});
 assert.match(t.calls[0].options.referenceContext[0].content,/PDF PHYSICAL PAGE 1\/2/);
 assert.equal(list(t).length,0);
 await t.provider.handle(t.view,{type:'submit',goal:'Followup without file'});
 assert.equal(t.calls[1].options.referenceContext,undefined);
});
test('front-end renders empty draft but preserves the prior message receipt on references reset',()=>{
 class Element{constructor(tag='div'){this.tag=tag;this.children=[];this.events={};this.value='';this.textContent='';this.dataset={};this.disabled=false;}append(...children){this.children.push(...children);}replaceChildren(...children){this.children=children;}addEventListener(k,v){this.events[k]=v;}setAttribute(){}remove(){}querySelector(selector){return this.children.find(x=>x.className===selector.slice(1))||null;}set innerHTML(_){throw Error('Unsafe HTML injection');}}
 const ids=new Map(['messages','goal','send','cancel','state','warn','empty','new','references','reference-note','reference-preview','reference-dropzone','ref-files','ref-selection','ref-editor'].map(x=>[x,new Element()]));
 const sent=[];let receive;
 const src=chatHtml({cspSource:'vscode-webview:'}).match(/<script nonce="[^"]+">([\s\S]*?)<\/script>/)[1];
 vm.runInNewContext(src,{acquireVsCodeApi:()=>({postMessage:m=>sent.push(m)}),document:{getElementById:k=>ids.get(k),createElement:k=>new Element(k)},window:{addEventListener:(_,fn)=>receive=fn},TextEncoder});
 receive({data:{type:'references',items:[{id:'ref1',name:'report.pdf',bytes:10}],loading:false}});
 assert.equal(ids.get('references').children.length,1);
 ids.get('goal').value='Summarize';ids.get('send').events.click();
 receive({data:{type:'references_included',items:[{name:'report.pdf',bytes:10,sha256:'abc'}]}});
 receive({data:{type:'references',items:[],loading:false}});
 assert.equal(ids.get('references').children.length,0);
 assert.equal(ids.get('reference-note').textContent.includes('@path'),true);
 assert.ok(ids.get('messages').children.length>=2);
 const rendered=ids.get('messages').children.map(e=>e.children.map(c=>c.className)).flat();
 assert.ok(rendered.includes('reference-receipt'));
 assert.equal(sent.filter(x=>x.type==='submit').length,1);
});
