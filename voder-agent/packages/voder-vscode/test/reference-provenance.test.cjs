'use strict';
const {test}=require('node:test');
const assert=require('node:assert/strict');
const {createHash}=require('node:crypto');
const {inspectionGoal}=require('../src/entry-orchestrator.cjs');
const {ChatRunService}=require('../src/chat-core.cjs');
const {validateReferences}=require('../src/reference-contract.cjs');
const content='import unittest\nfrom calc import add\nclass Tests(unittest.TestCase): pass\n';
const sha256=x=>createHash('sha256').update(x).digest('hex');
const ref=(source='attachment',name='test_calc.py')=>({id:'11111111-2222-4333-8444-555555555555',name,kind:'text',source,content,sha256:sha256(content)});
const folder={uri:{fsPath:'/tmp/voder-fixture-workspace'}};
function fakeRPC(){
 const sent=[];
 const rpc={referenceContextCapable:true,on(){},off(){},async request(method,params){sent.push({method,params});if(method==='session.create')return {sessionId:'11111111-2222-4333-8444-555555555555'};if(method==='task.submit')return {status:'ACCEPTED'};if(method==='task.wait')return {status:'DRAFT_UNVERIFIED',verified:false,draft:'{"answer":"Attachment parsed","scope":null}'};return {status:'OK'};}};
 return {rpc,sent};
}
test('no references retains normal repository inspection',()=>{
 const text=inspectionGoal('Inspect code');
 assert.match(text,/Use repository read\/list tools/);
 assert.match(text,/Do not modify any file/);
});
test('attached document does not force a repo search',()=>{
 const text=inspectionGoal('Analyze file only',{hasReferences:true});
 assert.doesNotMatch(text,/Use repository read\/list tools to inspect the actual project before answering/);
 assert.match(text,/do not list\/search\/read the repository/);
});
test('cross-source tasks may still inspect the workspace when necessary',()=>{
 const text=inspectionGoal('Compare my reference with project code',{hasReferences:true});
 assert.match(text,/Use repository tools only when the actual task also needs workspace facts/);
});
test('workspace ENOENT is not reclassified as an attachment defect',()=>{
 const text=inspectionGoal('Explain the attachment',{hasReferences:true});
 assert.match(text,/do not call it a missing attachment, a runtime failure or a high-severity defect/);
});
test('unprovided imports remain unknown not demonstrably absent',()=>{
 assert.match(inspectionGoal('Review attached tests',{hasReferences:true}),/Imported but unprovided modules are unknown/);
});
test('all inspection routes prohibit execution and edits',()=>{
 for(const flag of [false,true]){
  const text=inspectionGoal('Review',{hasReferences:flag});
  for(const rule of [/Do not modify any file/,/run scripts\/tests/,/request native actions/,/Return JSON only/])assert.match(text,rule);
 }
});
test('attachments are validated with their original source, without workspace promotion',()=>{
 assert.equal(validateReferences([ref()])[0].source,'attachment');
});
test('source=selection and source=workspace keep distinct provenance',()=>{
 for(const source of ['selection','workspace'])assert.equal(validateReferences([ref(source)])[0].source,source);
});
test('reference-only chat routes the reference and does not impose unconditional repo inspection',async()=>{
 const {rpc,sent}=fakeRPC(),service=new ChatRunService(async()=>rpc);
 const start=await service.inspect(folder,'仅根据附件内容解释测试',()=>{},{referenceContext:[ref()]});
 await start.completion;
 const submission=sent.find(x=>x.method==='task.submit').params.task;
 assert.equal(submission.referenceContext[0].source,'attachment');
 assert.match(submission.goal,/Start with the explicitly included reference snapshots/);
 assert.doesNotMatch(submission.goal,/Use repository read\/list tools to inspect the actual project before answering/);
});
test('reference-free chat still routes repository investigation',async()=>{
 const {rpc,sent}=fakeRPC(),service=new ChatRunService(async()=>rpc);
 const start=await service.inspect(folder,'Review current project');await start.completion;
 const task=sent.find(x=>x.method==='task.submit').params.task;
 assert.match(task.goal,/Use repository read\/list tools to inspect the actual project before answering/);
 assert.equal(Object.hasOwn(task,'referenceContext'),false);
});
test('reference questions may still use the project when explicitly requested',async()=>{
 const {rpc,sent}=fakeRPC(),service=new ChatRunService(async()=>rpc);
 const start=await service.inspect(folder,'Compare this reference against the checked-out project',()=>{},{referenceContext:[ref()]});await start.completion;
 assert.match(sent.find(x=>x.method==='task.submit').params.task.goal,/cross-source comparison/);
});
test('an invalid reference cannot be accepted as genuine evidence',()=>{
 assert.throws(()=>validateReferences([{...ref(),sha256:'0'.repeat(64)}]),{code:'REFERENCE_DIGEST'});
});
