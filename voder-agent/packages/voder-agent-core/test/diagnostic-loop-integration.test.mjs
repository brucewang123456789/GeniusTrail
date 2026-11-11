import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {mkdtemp,mkdir,writeFile,rm} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {AgentKernel} from '../src/kernel.mjs';
import {PiModelHost} from '../src/pi-host.mjs';
import {VgrBroker} from '../src/broker.mjs';
const digest=x=>createHash('sha256').update(x).digest('hex');
const baseline='print("bad")\n',originalHash=digest(baseline);
const candidate=text=>({relative_path:'src/sample.py',old_sha256:originalHash,new_content:text});
async function workspace(fn){const dir=await mkdtemp(join(tmpdir(),'voder-fourth-'));try{await mkdir(join(dir,'src'));await writeFile(join(dir,'src/sample.py'),baseline);return await fn(dir);}finally{await rm(dir,{recursive:true,force:true});}}
function upstream(execute){return {getAgentDir:()=>'/trusted',SessionManager:{inMemory:()=>({})},DefaultResourceLoader:class{async reload(){}},defineTool:x=>x,createAgentSession:async args=>{
 const names=args.customTools.map(t=>t.name);return {session:{getActiveToolNames:()=>names,getCallableToolNames:()=>names,getAllTools:()=>names.map(name=>({name,exposure:'direct'})),async prompt(goal){await execute(args.customTools,goal,args)},dispose(){},subscribe(){return()=>{}},abort(){}}};
}};}
async function call(tools,name,payload){return JSON.parse((await tools.find(x=>x.name===name).execute('id',payload,new AbortController().signal)).content[0].text);}
function native(cwd,id,track){return new VgrBroker({workspaceRoot:cwd,sessionId:id,native:{async executeAuthorized(a){track.count++;return {authority:'VODER_NATIVE_VGR',session_id:a.session_id,action_id:a.action_id,run_id:'test-run',status:'COMMITTED',journal_digest:'a'.repeat(64),commit_digest:'b'.repeat(64)};}},commitVerifier:{async confirmCommit(r,a){return {verified:true,authority:'VODER_NATIVE_VGR_STATE',session_id:a.session_id,action_id:a.action_id,run_id:r.run_id,commit_digest:r.commit_digest};}}});}
test('two observed sandbox failures, changed candidate, native verified commit; Pi untouched',async()=>workspace(async cwd=>{
 const issued=[],nativeActions={count:0};let calls=0;
 const host=new PiModelHost({upstream:upstream(async tools=>{
  const first=await call(tools,'voder_verify_candidate',{candidate:candidate('print("wrong")\n')});issued.push(first);
  assert.equal(first.voder_tool_result.status,'FAIL');assert.equal(first.voder_tool_result.repair_advice.strategy,'DIAGNOSE');
  const second=await call(tools,'voder_verify_candidate',{candidate:candidate('print("still bad")\n')});issued.push(second);
  assert.equal(second.voder_tool_result.repair_advice.identical_failures,1);
  const third=await call(tools,'voder_verify_candidate',{candidate:candidate('print("good")\n')});issued.push(third);
  assert.equal(third.voder_tool_result.status,'PASS');assert.equal(third.voder_tool_result.repair_advice.strategy,'PATCH');
  assert.equal(third.voder_tool_result.native_commit.confirmed,true);
 })});
 const kernel=new AgentKernel({cwd,sessionId:'task-id',broker:native(cwd,'task-id',nativeActions),modelHost:host,candidateVerifier:{async verifyCandidate(c){calls++;return {status:c.new_content.includes('good')?'PASS':'FAIL',backend:'isolated-test-double',sandboxed:true,evidence_digest:'f'.repeat(64),checks:[{name:'test',passed:true}],diagnostic:c.new_content.includes('good')?'':'AssertionError: expected good actual wrong'};}}});
 const answer=await kernel.submit({goal:'Fix multi-file verification defect',maxActions:4});
 assert.equal(answer.status,'VERIFIED');assert.equal(nativeActions.count,1);assert.equal(calls,3);
 assert.equal(answer.planning.mode,'ON_DEMAND_PLAN');assert.equal(answer.diagnostics.observed_sandbox_checks,3);
 assert.equal(kernel.events.filter(e=>e.type==='diagnostics.observed').length,3);
 assert.ok(!kernel.events.some(e=>JSON.stringify(e).includes('expected good actual wrong')));
 assert.equal(issued.length,3);kernel.close();
}));
test('new candidate bytes continue after same diagnostics, never submit without PASS',async()=>workspace(async cwd=>{
 let verifierRuns=0;
 const host=new PiModelHost({upstream:upstream(async tools=>{
  for(let i=0;i<3;i++){
   const reply=await call(tools,'voder_verify_candidate',{candidate:candidate(`print("bad-${i}")\n`)});
   assert.equal(reply.voder_tool_result.repair_advice.identical_failures,1);
  }
  const stopped=await call(tools,'voder_verify_candidate',{candidate:candidate('print("next")\n')});
  assert.equal(stopped.voder_tool_result.status,'FAIL');
 })});
 const kernel=new AgentKernel({cwd,sessionId:'stop',broker:{dispatch:async()=>{throw Error('NO_NATIVE_ACTION_ALLOWED')}},modelHost:host,candidateVerifier:{async verifyCandidate(){verifierRuns++;return {status:'FAIL',backend:'isolated-test-double',diagnostic:'AssertionError: expected 2 actual 1'};}}});
 const answer=await kernel.submit({goal:'Repair repeated failing test',maxActions:5});
 assert.equal(answer.status,'BLOCKED');assert.equal(answer.diagnostics.stalled,false);
 assert.equal(verifierRuns,4);kernel.close();
}));
test('malformed sandbox receipt cannot produce a diagnostic PASS or commit',async()=>workspace(async cwd=>{
 const host=new PiModelHost({upstream:upstream(async tools=>{
  const reply=await call(tools,'voder_verify_candidate',{candidate:candidate('print("good")\n')});
  assert.equal(reply.error,'INVALID_SANDBOX_RECEIPT');
 })});
 const kernel=new AgentKernel({cwd,sessionId:'bad',broker:{},modelHost:host,candidateVerifier:{async verifyCandidate(){return {status:'PASS',backend:'fake',sandboxed:false,evidence_digest:'a'.repeat(64),checks:[]};}}});
 const answer=await kernel.submit({goal:'test failed receipts'});
 assert.equal(answer.status,'BLOCKED');assert.equal(answer.diagnostics.observed_sandbox_checks,0);kernel.close();
}));
test('missing file blocks before model and planner does not override context preflight',async()=>workspace(async cwd=>{
 let modelCalls=0;const kernel=new AgentKernel({cwd,modelHost:{async createSession(){modelCalls++;throw Error('MUST_NOT_RUN')}},broker:{}});
 const answer=await kernel.submit({goal:'fix bug',requiredFiles:['missing/required.json']});
 assert.equal(answer.status,'WAITING_FOR_INPUT');assert.equal(modelCalls,0);kernel.close();
}));
