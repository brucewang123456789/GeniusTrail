'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),vm=require('node:vm');
const {chatHtml}=require('../src/chat-view.cjs');
class Element{
 constructor(tag='div'){this.tag=tag;this.children=[];this.events={};this.dataset={};this.value='';this.disabled=false;this.textContent='';}
 append(...nodes){this.children.push(...nodes);}
 addEventListener(name,fn){this.events[name]=fn;}
 setAttribute(){}
 remove(){}
 replaceChildren(){this.children=[];}
 querySelector(selector){const name=selector.slice(1);return this.children.find(n=>n.className===name)||null;}
 set innerHTML(_){throw Error('Unsafe DOM insertion');}
}
test('real webview script: click -> progress -> explicit result; review errors preserve verified delivery',()=>{
 const html=chatHtml({cspSource:'vscode-webview:'});const code=html.match(/<script nonce="[^"]+">([\s\S]*?)<\/script>/)[1];
 const ids=new Map(['messages','goal','send','cancel','state','warn','empty','new'].map(k=>[k,new Element()]));const sent=[];let receive;
 vm.runInNewContext(code,{acquireVsCodeApi:()=>({postMessage:m=>sent.push(m)}),document:{getElementById:k=>ids.get(k),createElement:t=>new Element(t)},window:{addEventListener:(name,fn)=>{receive=fn;}},TextEncoder});
 ids.get('goal').value='修复calc.py';ids.get('send').events.click();
 receive({data:{type:'created',sessionId:'exact'}});receive({data:{type:'settled',sessionId:'exact',verified:true,label:'DELIVERED',commitDigest:'abc'}});
 const bubble=ids.get('messages').children[1];const label=bubble.children[1].textContent;const row=bubble.querySelector('.line');const button=row.children[0];
 button.events.click();assert.equal(button.disabled,true);assert.deepEqual(JSON.parse(JSON.stringify(sent.at(-1))),{type:'action',command:'diff',sessionId:'exact'});
 receive({data:{type:'action_result',sessionId:'exact',command:'diff',status:'ERROR',message:'proof mismatch'}});
 assert.equal(button.disabled,false);assert.equal(bubble.children[1].textContent,label);assert.equal(bubble.querySelector('.trace'),null);assert.ok(bubble.children.some(n=>n.textContent==='Unable to open this view. Check task details.'));
 assert.ok(bubble.children.some(n=>n.tag==='details'));
});

test('UI discloses the one-active-task contract',()=>{assert.match(chatHtml({cspSource:'vscode-webview:'}),/Ready · one task at a time/);});
function setupAuthorizationUi(){const html=chatHtml({cspSource:'vscode-webview:'});const code=html.match(/<script nonce="[^"]+">([\s\S]*?)<\/script>/)[1];const ids=new Map(['messages','goal','send','cancel','state','warn','empty','new'].map(k=>[k,new Element()]));ids.get('goal').focus=()=>{};const sent=[];let receive;vm.runInNewContext(code,{acquireVsCodeApi:()=>({postMessage:m=>sent.push(m)}),document:{getElementById:k=>ids.get(k),createElement:t=>new Element(t)},window:{addEventListener:(name,fn)=>receive=fn},TextEncoder});return {ids,sent,receive:m=>receive({data:m})};}
test('actual UI: missing input preserves original task and ignores replacement choices; approval is inline and exact',()=>{const {ids,sent,receive}=setupAuthorizationUi();ids.get('goal').value='修复实际源码文件名.py';ids.get('send').events.click();receive({type:'input_required',goal:'修复实际源码文件名.py',message:'请补充真实文件',choices:[{target:'calc.py',test:'test_calc.py',goal:'修复calc.py，使test_calc.py通过'}]});assert.equal(ids.get('send').disabled,false);assert.equal(ids.get('goal').value,'修复实际源码文件名.py');let bubble=ids.get('messages').children.at(-1);assert.equal(bubble.children[1].textContent,'Scope needs attention · execution has not started');assert.equal(bubble.querySelector('.line'),null);assert.equal(ids.get('goal').value,'修复实际源码文件名.py');ids.get('goal').value='修复calc.py，使test_calc.py通过';ids.get('send').events.click();receive({type:'approval_requested',requestId:'exact-request',files:['calc.py'],operation:'Allow changes to the listed project files.',checksSummary:'Configured checks are enforced independently.'});bubble=ids.get('messages').children.at(-1);const panel=bubble.querySelector('.approval-panel');assert.ok(panel);assert.equal(ids.get('send').disabled,true);panel.querySelector('.line').children[0].events.click();assert.deepEqual(JSON.parse(JSON.stringify(sent.at(-1))),{type:'authorization_decision',requestId:'exact-request',decision:'approve'});assert.equal(panel.querySelector('.line').children[1].disabled,true);receive({type:'not_started',status:'NOT_STARTED',message:'你已取消授权',goal:'修复calc.py，使test_calc.py通过'});assert.equal(bubble.children[1].textContent,'Task not started');assert.equal(ids.get('send').disabled,false);});
test('actual UI: pending approval can render after view reload; execution uncertainty survives later errors',()=>{const {ids,receive}=setupAuthorizationUi();receive({type:'approval_requested',requestId:'restored',files:['calc.py'],operation:'Allow changes to the listed project files.',checksSummary:'Configured checks are enforced independently.'});const bubble=ids.get('messages').children.at(-1);assert.ok(bubble.querySelector('.approval-panel'));assert.equal(ids.get('send').disabled,true);receive({type:'uncertain',sessionId:'exact-session',message:'connection interrupted'});assert.match(bubble.children[1].textContent,/Execution outcome unknown/);receive({type:'error',message:'late RPC failure'});assert.match(bubble.children[1].textContent,/Execution outcome unknown/);});
