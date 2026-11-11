'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs/promises'),os=require('node:os'),path=require('node:path');
const {parsePlan,parsePlanWithDiagnostics}=require('../src/task-plan.cjs');
const plan={action:'execute',answer:'Create source, then report actual limitations.',files:['art.svg','verification.md'],checks:[]};
const response=draft=>({status:'DRAFT_UNVERIFIED',verified:false,draft});
test('planner accepts exact JSON, fenced JSON and prose surrounding one complete plan',()=>{
 const json=JSON.stringify(plan);
 for(const draft of [json,'```json\n'+json+'\n```','Implementation plan follows:\n```JSON\n'+json+'\n```\nReady for scope review.','先准备文件：\n'+json+'\n以上是计划。','\uFEFF'+json])assert.deepEqual(parsePlan(response(draft)).files,plan.files);
});
test('planner decodes JSON string and known transport/report envelopes as data',()=>{
 for(const obj of [JSON.stringify(plan),{plan},{answer:JSON.stringify(plan)},{draft:JSON.stringify(plan)}])assert.equal(parsePlan(response(JSON.stringify(obj))).action,'execute');
});
test('balanced scan handles escaped quotes, braces and backslashes inside JSON strings',()=>{
 for(const answer of ['literal {inside} text','an escaped "quote" and } brace','path \\ and "{ quoted }"','{{a}} \\"suffix']){
  const obj={...plan,answer};assert.equal(parsePlan(response('Plan:\n'+JSON.stringify(obj)+'\nEnd')).answer,answer);
 }
});
test('two competing plans fail closed instead of selecting the first convenient scope',()=>{
 const other={...plan,files:['different.svg']};assert.throws(()=>parsePlan(response(JSON.stringify(plan)+'\n'+JSON.stringify(other))),/competing plans/);
});
test('truncated JSON, prose-only, JavaScript object literals and unsafe scopes are never guessed or evaluated',()=>{
 for(const draft of [JSON.stringify(plan).slice(0,-2),'Create art.svg and verification.md.',"{action:'execute',answer:'x',files:['x'],checks:[]}"] )assert.throws(()=>parsePlan(response(draft)));
 assert.throws(()=>parsePlan(response('Plan: '+JSON.stringify({...plan,files:['../escape']}))),/safe relative/);
});
test('failure capture retains exact bounded returned text privately and exposes only its diagnostic path',async()=>{
 const root=await fs.mkdtemp(path.join(os.tmpdir(),'voder-planner-contract-'));
 try{const draft='A plain-text plan without a JSON envelope.';let thrown;try{await parsePlanWithDiagnostics(response(draft),{storageRoot:root,workspace:'/task',goal:'create source'});}catch(e){thrown=e;}
  assert.ok(thrown);assert.match(thrown.message,/Diagnostic:/);assert.ok(!thrown.message.includes(draft));
  const dir=path.join(root,'task-planning-diagnostics'),files=await fs.readdir(dir);assert.equal(files.length,1);const file=path.join(dir,files[0]),record=JSON.parse(await fs.readFile(file,'utf8'));assert.equal(record.draft,draft);assert.equal(record.status,'DRAFT_UNVERIFIED');assert.equal((await fs.stat(file)).mode&0o077,0);assert.equal((await fs.stat(dir)).mode&0o077,0);
 }finally{await fs.rm(root,{recursive:true,force:true});}
});
test('successful plan extraction creates no diagnostic or workspace files',async()=>{
 const root=await fs.mkdtemp(path.join(os.tmpdir(),'voder-planner-success-'));
 try{await parsePlanWithDiagnostics(response('Plan:\n'+JSON.stringify(plan)),{storageRoot:root,workspace:'/task',goal:'create source'});assert.deepEqual(await fs.readdir(root),[]);}finally{await fs.rm(root,{recursive:true,force:true});}
});
