import {test} from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,rm} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {createHash} from 'node:crypto';
import {PiModelHost} from '../src/pi-host.mjs';
import {AgentKernel} from '../src/kernel.mjs';
import {VgrBroker} from '../src/broker.mjs';
const sha=s=>createHash('sha256').update(s).digest('hex');
const model={provider:'offline-fixture',id:'fixture',input:['text'],maxTokens:4096,contextWindow:1048576};
function fixturePi(script){
 const api={providerCalls:0,model,normalizeContext:x=>x,getAgentDir:()=>'/trusted',SessionManager:{inMemory:()=>({})},DefaultResourceLoader:class{constructor(o){this.o=o}async reload(){}},defineTool:x=>x,
 async createAgentSession(options){const defs=options.customTools,names=defs.map(x=>x.name);
  const session={model,systemPrompt:options.resourceLoader.o.systemPromptOverride(),agent:{streamFunction(){api.providerCalls++;return {};}},messages:[],getActiveToolNames:()=>names,getCallableToolNames:()=>names,getAllTools:()=>names.map(name=>({name,exposure:'direct'})),subscribe(){return()=>{}},dispose(){},abort(){},
   async prompt(){await script({
    request:()=>session.agent.streamFunction(model,{messages:[{role:'system',content:session.systemPrompt}],tools:[]},{}),
    call:async(name,args)=>JSON.parse((await defs.find(t=>t.name===name).execute('fixture',args,new AbortController().signal)).content[0].text)
   });}
  };return {session};
 }};return api;
}
async function fixture(fn){const cwd=await mkdtemp(join(tmpdir(),'voder-tool-control-'));try{await mkdir(join(cwd,'src'));await writeFile(join(cwd,'src/a.py'),'print(0)\n');await fn(cwd);}finally{await rm(cwd,{recursive:true,force:true});}}
const candidate={relative_path:'src/a.py',old_sha256:sha('print(0)\n'),new_content:'print(2)\n'};
const verifier={async verifyCandidate(){return {status:'PASS',backend:'OFFLINE_FIXTURE_NOT_MODAL',sandboxed:true,evidence_digest:'a'.repeat(64),checks:[{name:'fixture-only',passed:true}]};}};
function broker(cwd,sessionId){return new VgrBroker({workspaceRoot:cwd,sessionId,native:{async executeAuthorized(a){return {authority:'VODER_NATIVE_VGR',session_id:a.session_id,action_id:a.action_id,run_id:'fixture-run',status:'COMMITTED',journal_digest:'a'.repeat(64),commit_digest:'b'.repeat(64)};}},commitVerifier:{async confirmCommit(r,a){return {verified:true,authority:'VODER_NATIVE_VGR_STATE',session_id:a.session_id,action_id:a.action_id,run_id:r.run_id,commit_digest:r.commit_digest};}}});}

test('model receives phase and budget feedback; two stalled model turns stop before another provider call',()=>fixture(async cwd=>{
 const api=fixturePi(async({request,call})=>{
  request();await call('voder_read_file',{path:'src/a.py'});await call('voder_list_files',{path:'src'});
  for(let i=0;i<3;i++){
   request();const r=await call('voder_search_code',{path:'src/a.py',query:'print'});
   assert.equal(r.error,'READ_BUDGET_RESERVED');assert.equal(r.tool_control.phase,'FINALIZING');assert.equal(r.tool_control.remaining_tool_calls,2);
  }
 });
 const kernel=new AgentKernel({cwd,sessionId:'stalled',modelHost:new PiModelHost({upstream:api}),broker:{},candidateVerifier:verifier,orchestrationOptions:{maxToolCalls:4}});
 const result=await kernel.submit({goal:'repair'});
 assert.equal(result.status,'BLOCKED');assert.equal(result.blocked_reason,'FINALIZATION_STALLED');assert.equal(api.providerCalls,3);
 assert.equal(result.orchestration.verification_calls,0);assert.equal(result.orchestration.tool_calls,2);
 assert.equal(kernel.events.find(x=>x.type==='task.settled').reason,'FINALIZATION_STALLED');kernel.close();
}));
test('finalization after read budget can still verify and explicitly commit exact bytes',()=>fixture(async cwd=>{
 const api=fixturePi(async({request,call})=>{
  request();await call('voder_read_file',{path:'src/a.py'});const read=await call('voder_list_files',{path:'src'});assert.equal(read.tool_control.phase,'FINALIZING');
  request();const checked=await call('voder_verify_candidate',{candidate});assert.equal(checked.voder_tool_result.status,'PASS');
  request();const committed=await call('voder_commit_verified_candidate',{candidate_digest:checked.voder_tool_result.candidate_digest});assert.equal(committed.voder_tool_result.status,'COMMITTED');
 });
 const kernel=new AgentKernel({cwd,sessionId:'reserved',modelHost:new PiModelHost({upstream:api}),broker:broker(cwd,'reserved'),candidateVerifier:verifier,orchestrationOptions:{maxToolCalls:4}});
 const result=await kernel.submit({goal:'repair'});
 assert.equal(result.status,'VERIFIED');assert.notEqual(result.status,'DELIVERED');assert.equal(result.orchestration.tool_calls,4);assert.equal(api.providerCalls,3);kernel.close();
}));
test('model-request reserve transitions early and leaves requests for PASS and explicit commit',()=>fixture(async cwd=>{
 const api=fixturePi(async({request,call})=>{
  for(let i=0;i<5;i++){request();const r=await call('voder_search_code',{path:'src/a.py',query:String(i)});if(i===4){assert.equal(r.error,'MODEL_FINALIZATION_RESERVE');assert.equal(r.tool_control.phase,'FINALIZING');}}
  request();const checked=await call('voder_verify_candidate',{candidate});assert.equal(checked.voder_tool_result.status,'PASS');
  request();await call('voder_commit_verified_candidate',{candidate_digest:checked.voder_tool_result.candidate_digest});
 });
 const kernel=new AgentKernel({cwd,sessionId:'model-reserve',modelHost:new PiModelHost({upstream:api,maxModelRequests:8}),broker:broker(cwd,'model-reserve'),candidateVerifier:verifier});
 const result=await kernel.submit({goal:'repair'});assert.equal(result.status,'VERIFIED');assert.equal(api.providerCalls,7);assert.equal(result.orchestration.phase_reason,'MODEL_FINALIZATION_RESERVE');kernel.close();
}));
test('hard tool exhaustion rejects later tools and prevents another provider request',()=>fixture(async cwd=>{
 let actions=0;
 const api=fixturePi(async({request,call})=>{
  request();assert.equal((await call('voder_read_file',{path:'src/a.py'})).error,'TOOL_BUDGET_EXHAUSTED');
  assert.equal((await call('voder_authorized_action',{action:{}})).error,'TOOL_BUDGET_EXHAUSTED');request();
 });
 const tools=[{name:'voder_authorized_action',description:'fixture',async execute(){actions++;return {status:'COMMITTED'};}},{name:'voder_read_file',description:'fixture',parameters:{type:'object'},async execute(){throw Object.assign(Error('spent'),{code:'TOOL_BUDGET_EXHAUSTED'});}}];
 const session=await new PiModelHost({upstream:api}).createSession({cwd,systemPrompt:'rules',tools});
 await assert.rejects(session.prompt('repair'),{code:'TOOL_BUDGET_EXHAUSTED'});assert.equal(actions,0);assert.equal(api.providerCalls,1);session.dispose();
}));
test('hard model request limit settles as an accurate blocker without unbounded replay',()=>fixture(async cwd=>{
 const api=fixturePi(async({request})=>{request();request();request();});
 const kernel=new AgentKernel({cwd,sessionId:'requests',modelHost:new PiModelHost({upstream:api,maxModelRequests:2}),broker:{}});
 const result=await kernel.submit({goal:'repair'});assert.equal(result.status,'BLOCKED');assert.equal(result.blocked_reason,'MODEL_REQUEST_BUDGET');assert.equal(api.providerCalls,2);kernel.close();
}));
test('cancellation and stale task callbacks still cannot use reserved capacity',()=>fixture(async cwd=>{
 let tools;const kernel=new AgentKernel({cwd,sessionId:'cancel',broker:{},modelHost:{async createSession(o){tools=o.tools;return {async prompt(){o.toolControl('READ_BUDGET_RESERVED');await kernel.cancel();},async abort(){},dispose(){}};}}});
 assert.equal((await kernel.submit({goal:'repair'})).status,'CANCELLED');
 await assert.rejects(tools.find(t=>t.name==='voder_authorized_action').execute({kind:'patch',...candidate}),{code:'CANCELLED'});kernel.close();
}));
test('new sandbox FAIL permits a focused read then revised PASS and explicit commit during finalization',()=>fixture(async cwd=>{
 let checks=0;const failThenPass={async verifyCandidate(){return ++checks===1?{status:'FAIL',backend:'OFFLINE_FIXTURE_NOT_MODAL',diagnostic:'fixture SyntaxError'}:verifier.verifyCandidate();}};
 const api=fixturePi(async({request,call})=>{
  request();for(let i=0;i<12;i++)await call('voder_search_code',{path:'src/a.py',query:String(i)});
  request();const failed=await call('voder_verify_candidate',{candidate:{...candidate,new_content:'bad'}});assert.equal(failed.voder_tool_result.status,'FAIL');assert.equal(failed.tool_control.repair_read_credits,1);
  request();const repair=await call('voder_read_file',{path:'src/a.py'});assert.equal(repair.tool_control.repair_reads,1);
  request();const checked=await call('voder_verify_candidate',{candidate});assert.equal(checked.voder_tool_result.status,'PASS');
  request();await call('voder_commit_verified_candidate',{candidate_digest:checked.voder_tool_result.candidate_digest});
 });
 const kernel=new AgentKernel({cwd,sessionId:'focused-repair',modelHost:new PiModelHost({upstream:api}),broker:broker(cwd,'focused-repair'),candidateVerifier:failThenPass,orchestrationOptions:{maxToolCalls:16}});
 const result=await kernel.submit({goal:'repair'});assert.equal(result.status,'VERIFIED');assert.equal(result.orchestration.tool_calls,16);assert.equal(checks,2);kernel.close();
}));
