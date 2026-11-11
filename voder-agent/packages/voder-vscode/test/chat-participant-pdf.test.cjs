'use strict';
const test=require('node:test');
const assert=require('node:assert/strict');
const Module=require('node:module');
const originalLoad=Module._load;
Module._load=function(request,parent,isMain){
 if(request==='vscode')return {TreeItem:class{},TreeItemCollapsibleState:{None:0},ThemeIcon:class{},StatusBarAlignment:{Left:0}};
 return originalLoad.apply(this,arguments);
};
let VoderExtension;
try{({VoderExtension}=require('../src/extension.cjs'));}finally{Module._load=originalLoad;}
const folder={uri:{fsPath:'/tmp/test-voder-workspace'}};
const makeStream=()=>({messages:[],markDownCalls:0,progress(){},markdown(s){this.messages.push(String(s));this.markDownCalls++;}});
test('read-only @voder PDF request uses SDK inspection instead of native write authorization',async()=>{
 const calls=[];
 const app={
  trusted:()=>true,_folder:async()=>folder,
  chatView:{references:{requestContext:async(_folder,goal,refs)=>{calls.push(['reference',goal,refs.length]);return [{id:'6f38da0e-24ad-44dc-8f3b-6c36161a05b0',name:'x.pdf',kind:'text'}];}}},
  ensureNativeChat:async()=>{throw Error('UNEXPECTED_NATIVE_WRITE_AUTHORIZATION');},
  _trackChatSession:()=>{},
  chatRuns:{inspect:async(_folder,_goal,notify,opts)=>{calls.push(['inspect',opts.referenceContext.length]);return {sessionId:'851e3797-2e89-4d26-b806-a9a2011c2c11',completion:Promise.resolve({status:'DRAFT_UNVERIFIED',verified:false,draft:JSON.stringify({answer:'PDF图表解释已生成',scope:null}),sessionId:'851e3797-2e89-4d26-b806-a9a2011c2c11'})};}}
 };
 const stream=makeStream();
 const res=await VoderExtension.prototype._runChatParticipant.call(app,{prompt:'只分析PDF图表',references:[{value:{uri:{scheme:'file',fsPath:'/tmp/report.pdf'}}}]},{history:[]},stream,null);
 assert.deepEqual(calls.map(x=>x[0]),['reference','inspect']);
 assert.equal(res.metadata.verified,false);
 assert.equal(stream.messages.length,1);
 assert.match(stream.messages[0],/PDF图表解释已生成/);
});
test('@voder write request still requires the existing native authorization',async()=>{
 let checks=0;
 const app={trusted:()=>true,_folder:async()=>folder,
  chatView:{references:{requestContext:async()=>[]}},
  ensureNativeChat:async()=>{checks++;throw Error('SCOPE_NOT_APPROVED');}}
 ;
 await assert.rejects(VoderExtension.prototype._runChatParticipant.call(app,{prompt:'请修改这个源码并实现功能',references:[]},{history:[]},makeStream(),null),/SCOPE_NOT_APPROVED/);
 assert.equal(checks,1);
});
