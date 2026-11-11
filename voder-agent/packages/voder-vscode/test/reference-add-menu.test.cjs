'use strict';
const {TestChatViewProvider}=require('./chat-test-context.cjs');
const {test}=require('node:test');
const assert=require('node:assert/strict');
const vm=require('node:vm');
const {chatHtml,VoderChatViewProvider}=require('../src/chat-view.cjs');
const {ReferenceController}=require('../src/reference-ui.cjs');
class Element{
 constructor(tag='div'){this.tag=tag;this.children=[];this.events={};this.attributes={};this.hidden=false;this.disabled=false;this.textContent='';this.value='';this.className='';this.dataset={};}
 addEventListener(event,fn){this.events[event]=fn;}
 setAttribute(k,v){this.attributes[k]=v;}
 append(...nodes){this.children.push(...nodes);}
 replaceChildren(...nodes){this.children=nodes;}
 contains(target){return this===target||this.children.includes(target);}
 focus(){this.focused=true;}
 remove(){}
 set innerHTML(_){throw Error('DOM injection');}
}
function rendered(){
 const html=chatHtml({cspSource:'vscode-webview:'});
 const keys=['messages','goal','send','cancel','state','warn','empty','new','references','reference-note','reference-preview','reference-dropzone','reference-hint','add-context','add-menu','ref-files','ref-selection','ref-editor'];
 const ids=new Map(keys.map(id=>[id,new Element()]));ids.get('add-menu').hidden=true;
 ids.get('ref-selection').disabled=true;ids.get('ref-editor').disabled=true;
 const messages=[];let receive;const documentEvents={};
 const sandbox={
  acquireVsCodeApi:()=>({postMessage:x=>messages.push(x)}),
  document:{getElementById:id=>ids.get(id),createElement:tag=>new Element(tag),addEventListener:(type,listener)=>{documentEvents[type]=listener;}},
  window:{addEventListener:(type,fn)=>{if(type==='message')receive=fn;}},
  TextEncoder
 };
 vm.runInNewContext(html.match(/<script nonce="[^"]+">([\s\S]*?)<\/script>/)[1],sandbox);
 return {html,ids,messages,receive:m=>receive({data:m}),documentEvents};
}
test('one compact Add trigger replaces three exposed buttons, with informative options',()=>{
 const {html}=rendered();assert.match(html,/id="add-context"/);assert.match(html,/id="add-menu"[^>]*hidden/);
 for(const text of ['Files','Selection','Open file','Choose documents or code','Selected text in the editor','Entire active editor buffer'])assert.ok(html.includes(text));
 assert.doesNotMatch(html,/>Add selection</);assert.doesNotMatch(html,/>Add open file</);
});
test('no editor: both editor options disabled; quiet contextual hint, no red error',()=>{
 const {ids,messages,receive}=rendered();
 receive({type:'reference_editor_state',hasEditor:false,hasSelection:false});
 ids.get('add-context').events.click();assert.equal(ids.get('add-menu').hidden,false);assert.equal(messages.at(-1).type,'reference_editor_state_request');
 receive({type:'reference_editor_state',hasEditor:false,hasSelection:false});
 assert.equal(ids.get('ref-files').disabled,false);assert.equal(ids.get('ref-selection').disabled,true);assert.equal(ids.get('ref-editor').disabled,true);
 assert.match(ids.get('reference-hint').textContent,/Open a text file/);
 assert.equal(ids.get('warn').textContent,'');
});
test('active file without selection enables Open file only; selection enables both',()=>{
 const {ids,messages,receive}=rendered();
 ids.get('add-context').events.click();receive({type:'reference_editor_state',hasEditor:true,hasSelection:false});
 assert.equal(ids.get('ref-editor').disabled,false);assert.equal(ids.get('ref-selection').disabled,true);
 assert.match(ids.get('reference-hint').textContent,/Select text/);
 ids.get('ref-editor').events.click();assert.equal(messages.at(-1).type,'reference_editor');assert.equal(ids.get('add-menu').hidden,true);
 ids.get('add-context').events.click();receive({type:'reference_editor_state',hasEditor:true,hasSelection:true});
 assert.equal(ids.get('ref-editor').disabled,false);assert.equal(ids.get('ref-selection').disabled,false);
 ids.get('ref-selection').events.click();assert.equal(messages.at(-1).type,'reference_selection');assert.equal(ids.get('add-menu').hidden,true);
});
test('Files remains selectable; Escape closes and restores focus; busy disables menu',()=>{
 const {ids,messages,receive,documentEvents}=rendered();
 ids.get('add-context').events.click();ids.get('ref-files').events.click();assert.equal(messages.at(-1).type,'reference_files');
 ids.get('add-context').events.click();documentEvents.keydown({key:'Escape'});
 assert.equal(ids.get('add-menu').hidden,true);assert.equal(ids.get('add-context').focused,true);
 receive({type:'references_loading',loading:true});assert.equal(ids.get('add-context').disabled,true);
 receive({type:'references_loading',loading:false});assert.equal(ids.get('add-context').disabled,false);
});
test('actual host bridges editor status and never paints missing editor as a red reference failure',async()=>{
 const events=[],subscriptions=[];
 const api={window:{activeTextEditor:undefined,onDidChangeActiveTextEditor:fn=>{subscriptions.push(fn);return{dispose(){}};},onDidChangeTextEditorSelection:fn=>{subscriptions.push(fn);return{dispose(){}};}},workspace:{asRelativePath:()=> 'demo.ts'}};
 const controller=new ReferenceController({vscode:api});
 const provider=new TestChatViewProvider({trusted:()=>true,chatWorkspace:()=>undefined},{referenceController:controller});
 let dispose;
 const view={webview:{cspSource:'vscode-webview:',onDidReceiveMessage(){},postMessage:m=>events.push(m)},onDidDispose:fn=>dispose=fn};
 provider.resolveWebviewView(view);
 await provider.handle(view,{type:'ready'});
 assert.deepEqual(events.filter(x=>x.type==='reference_editor_state').at(-1),{type:'reference_editor_state',hasEditor:false,hasSelection:false});
 await provider.handle(view,{type:'reference_selection'});
 assert.equal(events.some(x=>x.type==='reference_error'),false);
 assert.equal(events.some(x=>x.type==='reference_hint'),true);
 const doc={uri:{scheme:'file',fsPath:'/repo/demo.ts'},lineCount:2,getText:sel=>sel?'only chosen':'whole unsaved text\nsecond line'};
 api.window.activeTextEditor={document:doc,selection:{isEmpty:false,start:{line:0,character:0},end:{line:0,character:11}}};
 subscriptions[0]();assert.deepEqual(events.filter(x=>x.type==='reference_editor_state').at(-1),{type:'reference_editor_state',hasEditor:true,hasSelection:true});
 await provider.handle(view,{type:'reference_selection'});await provider.handle(view,{type:'reference_editor'});
 const rows=controller.state(view).store.snapshot();assert.equal(rows[0].content,'only chosen');assert.equal(rows[1].content,'whole unsaved text\nsecond line');
 dispose();assert.equal(controller.states.size,0);
});
