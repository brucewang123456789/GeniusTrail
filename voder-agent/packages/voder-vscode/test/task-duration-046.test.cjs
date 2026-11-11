'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),vm=require('node:vm');
const {formatTaskDuration,createTaskClock}=require('../src/task-duration.cjs');
const {chatHtml}=require('../src/chat-view.cjs');
test('professional English duration includes hours only when needed',()=>{
 for(const [ms,text] of [[0,'0s'],[59000,'59s'],[60000,'1m'],[3599000,'59m 59s'],[3600000,'1h'],[3723000,'1h 2m 3s'],[86401000,'24h 1s']])assert.equal(formatTaskDuration(ms),text);
});
test('duration freezes exactly once and starts a fresh clock for the next task',()=>{
 let now=0;const c=createTaskClock(()=>now);c.start();now=62000;assert.equal(c.freeze(),62000);now=999999;assert.equal(c.freeze(),62000);c.start();assert.equal(c.elapsed(),62000);c.reset();c.start();now+=11000;assert.equal(c.elapsed(),11000);
});
class Element{
 constructor(tag='div'){this.tag=tag;this.children=[];this.events={};this.dataset={};this.value='';this.textContent='';this.disabled=false;this.hidden=false;}
 insertBefore(node,before){const i=this.children.indexOf(before);if(i<0)this.children.push(node);else this.children.splice(i,0,node)}
 append(...nodes){this.children.push(...nodes)}addEventListener(n,f){this.events[n]=f}setAttribute(){}remove(){}replaceChildren(...n){this.children=n}
 querySelector(s){const c=s.slice(1);return this.children.find(n=>String(n.className||'').split(' ').includes(c))||null}
}
function setup(){let now=0,receive,tick;const ids=new Map(['messages','goal','send','cancel','state','elapsed','warn','empty','new'].map(k=>[k,new Element()]));
 const code=chatHtml({cspSource:'vscode-webview:'}).match(/<script nonce="[^"]+">([\s\S]*?)<\/script>/)[1];
 vm.runInNewContext(code,{acquireVsCodeApi:()=>({postMessage(){}}),document:{getElementById:k=>ids.get(k),createElement:t=>new Element(t)},window:{addEventListener:(n,f)=>receive=f},performance:{now:()=>now},setInterval:f=>{tick=f},TextEncoder});
 return {ids,advance(ms){now+=ms;tick()},send(m){receive({data:m})},submit(goal='Research docs'){ids.get('goal').value=goal;ids.get('send').events.click()},bubble(){return ids.get('messages').children.at(-1)}};
}
test('actual webview: final-answer render freezes and retains the total while progress stays hidden',()=>{
 const x=setup();x.submit();x.send({type:'accepted',sessionId:'s'});x.advance(3723000);x.send({type:'settled',sessionId:'s',status:'DRAFT_UNVERIFIED',draft:'The final answer',presentation:{kind:'ANSWER',title:'Response complete'}});
 const b=x.bubble(),duration=b.querySelector('.task-duration');assert.equal(duration.textContent,'Completed in 1h 2m 3s');assert.equal(b.querySelector('.phase').hidden,true);
 x.advance(90000);assert.equal(duration.textContent,'Completed in 1h 2m 3s');assert.equal(x.ids.get('elapsed').textContent,'');
 x.send({type:'settled',sessionId:'s',verified:true});assert.equal(b.children.filter(n=>String(n.className).includes('task-duration')).length,1);
});
test('actual webview: early file delivery freezes once, later settlement cannot extend it',()=>{
 const x=setup();x.submit();x.advance(65000);x.send({type:'task_delivered',sessionId:'s',files:['index.html']});x.advance(10000);x.send({type:'settled',sessionId:'s',verified:true});assert.equal(x.bubble().querySelector('.task-duration').textContent,'Completed in 1m 5s');
});
test('actual webview: intermediate inspection does not freeze a continuing task',()=>{
 const x=setup();x.submit();x.advance(30000);x.send({type:'inspection_complete',continuing:true,status:'INSPECTED',answer:'Intermediate investigation'});assert.equal(x.bubble().querySelector('.task-duration').textContent,'Elapsed 30s');x.advance(20000);x.send({type:'settled',sessionId:'s',draft:'Done'});assert.equal(x.bubble().querySelector('.task-duration').textContent,'Completed in 50s');
});
test('actual webview: starting a second task preserves the first final duration',()=>{
 const x=setup();x.submit();x.advance(62000);x.send({type:'settled',sessionId:'a',draft:'Done'});const first=x.bubble().querySelector('.task-duration');x.submit('Another task');x.advance(9000);x.send({type:'settled',sessionId:'b',draft:'Done again'});assert.equal(first.textContent,'Completed in 1m 2s');assert.equal(x.bubble().querySelector('.task-duration').textContent,'Completed in 9s');
});
test('actual webview: errors retain a stopped duration rather than a completion claim',()=>{
 const x=setup();x.submit();x.advance(13000);x.send({type:'error',code:'TASK_FAILED'});assert.equal(x.bubble().querySelector('.task-duration').textContent,'Stopped after 13s');x.advance(5000);assert.equal(x.ids.get('elapsed').textContent,'');
});

test('actual webview: duration occupies the answer header and has no duplicate near the composer',()=>{
 const x=setup();x.submit();x.advance(17000);assert.equal(x.bubble().querySelector('.task-duration').textContent,'Elapsed 17s');
 x.send({type:'settled',sessionId:'s',draft:'Final answer'});const b=x.bubble(),header=b.querySelector('.task-duration');
 assert.equal(header.textContent,'Completed in 17s');assert.ok(b.children.indexOf(header)<b.children.indexOf(b.querySelector('.assistant-prose')));
 assert.ok(b.children.indexOf(header)<b.children.indexOf(b.querySelector('.response-status')));assert.equal(x.ids.get('elapsed').textContent,'');assert.equal(x.ids.get('elapsed').hidden,true);
});
test('zero-valued larger units are omitted without losing elapsed seconds',()=>{
 assert.equal(formatTaskDuration(17000),'17s');assert.equal(formatTaskDuration(60000),'1m');assert.equal(formatTaskDuration(3608000),'1h 8s');assert.equal(formatTaskDuration(3720000),'1h 2m');
});


test('duration styling follows semantic classes after header insertion, with shared typography and a decorative monochrome clock',()=>{
 const html=chatHtml({cspSource:'vscode-webview:'}),css=html.match(/<style nonce="[^"]+">([\s\S]*?)<\/style>/)[1];
 assert.ok(!/\.agent>[^{}]*:nth-child\(2\)/.test(css));
 assert.match(css,/\.agent>\.response-status,\.agent>\.task-duration\{font:500 12px\/1\.5/);
 assert.match(css,/\.agent>\.task-duration::before\{content:"";[^}]*data:image\/svg\+xml/);
 assert.match(css,/\.agent>\.task-duration\{[^}]*color:#b8b8b8/);
 const x=setup();x.submit();x.advance(15000);const row=x.bubble().querySelector('.task-duration');
 assert.equal(row.textContent,'Elapsed 15s');assert.equal(x.bubble().children.filter(n=>String(n.className).includes('task-duration')).length,1);
 x.send({type:'settled',sessionId:'s',draft:'Done'});x.advance(10000);assert.equal(row.textContent,'Completed in 15s');
});
