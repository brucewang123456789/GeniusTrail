import {test} from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,rm} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {createHash} from 'node:crypto';
import {AgentKernel} from '../src/kernel.mjs';
const sha=x=>createHash('sha256').update(x).digest('hex');
const first={relative_path:'src/a.py',old_sha256:sha('before\n'),new_content:'after\n'};
const second={...first,new_content:'other\n'};
const pass={status:'PASS',backend:'modal-fixture',sandboxed:true,evidence_digest:'a'.repeat(64),checks:[{name:'official-sandbox',passed:true}]};
async function fixture(fn){const dir=await mkdtemp(join(tmpdir(),'voder-priority-'));try{await mkdir(join(dir,'src'));await writeFile(join(dir,'src/a.py'),'before\n');await fn(dir);}finally{await rm(dir,{recursive:true,force:true});}}
function makeKernel(cwd,{onPrompt,verify=async()=>pass,dispatch=async()=>({status:'COMMITTED'}),models=[{provider:'primary',id:'A'},{provider:'secondary',id:'B'}]}={}){
 let nativeCalls=0,verificationCalls=0,switches=0;
 const broker={deliveryResult:null,async dispatch(input,signal){nativeCalls++;return dispatch(input,signal);}};
 let tools;
 const host={async createSession({tools:t}){tools=t;return {async listModels(){return {current:models[0],models}},async switchModel(){switches++;return {status:'MODEL_SELECTED'}},async prompt(){await onPrompt?.({tools,k,broker})},dispose(){}}}};
 const k=new AgentKernel({cwd,broker,sessionId:'priority-fixture',modelHost:host,candidateVerifier:{async verifyCandidate(candidate){verificationCalls++;return verify(candidate);}}});
 return {k,broker,tools:()=>tools,counts:()=>({nativeCalls,verificationCalls,switches})};
}
function tool(tools,name){return tools.find(t=>t.name===name)}
test('PASS immediately invokes native VGR before model can request a second candidate or model',()=>fixture(async cwd=>{
 let x; x=makeKernel(cwd,{onPrompt:async({tools})=>{
  const checked=await tool(tools,'voder_verify_candidate').execute({candidate:first});
  assert.equal(checked.status,'PASS');assert.deepEqual(checked.native_commit,{status:'COMMITTED',confirmed:true});
  assert.equal(x.counts().nativeCalls,1);
  await assert.rejects(tool(tools,'voder_verify_candidate').execute({candidate:second}),{code:'ALREADY_COMMITTED'});
  const duplicate=await tool(tools,'voder_commit_verified_candidate').execute({candidate_digest:checked.candidate_digest});
  assert.equal(duplicate.status,'COMMITTED');assert.equal(x.counts().nativeCalls,1);
 }});
 const outcome=await x.k.submit({goal:'fix'});assert.equal(outcome.status,'VERIFIED');
 assert.equal(x.counts().nativeCalls,1);assert.equal(x.counts().switches,0);
 assert.equal(x.k.events.filter(e=>e.type==='vgr.priority_commit_requested').length,1);
 assert.equal(x.k.events.filter(e=>e.type==='vgr.action_settled'&&e.verdict==='COMMITTED').length,1);
}));
test('trusted delivery still distinguishes actual delivered from only native-verified',()=>fixture(async cwd=>{
 let x;x=makeKernel(cwd,{onPrompt:async({tools})=>{const r=await tool(tools,'voder_verify_candidate').execute({candidate:first});assert.equal(r.native_commit.confirmed,true);}});
 x.broker.delivery={status:'DELIVERED'};x.broker.deliveryResult={status:'DELIVERED'};
 const r=await x.k.submit({goal:'fix'});assert.equal(r.status,'DELIVERED');assert.equal(x.counts().nativeCalls,1);
}));
test('FAIL cannot trigger native commit or confirmed delivery',()=>fixture(async cwd=>{
 let x;x=makeKernel(cwd,{verify:async()=>({status:'FAIL',backend:'modal-fixture',diagnostic:'compiler failed'}),onPrompt:async({tools})=>{const r=await tool(tools,'voder_verify_candidate').execute({candidate:first});assert.equal(r.status,'FAIL');assert.equal(r.native_commit,undefined);}});
 const r=await x.k.submit({goal:'fix'});assert.equal(r.status,'BLOCKED');assert.equal(x.counts().nativeCalls,0);
}));
test('native denial returns repair advice and a DIFFERENT verified candidate may commit',()=>fixture(async cwd=>{
 let attempts=0,x;
 x=makeKernel(cwd,{dispatch:async()=>({status:++attempts===1?'DENIED':'COMMITTED'}),onPrompt:async({tools})=>{
  const verifyTool=tool(tools,'voder_verify_candidate');
  const firstResult=await verifyTool.execute({candidate:first});assert.equal(firstResult.native_commit.status,'DENIED');assert.equal(firstResult.next_step.status,'REPAIR_REQUIRED');
  const secondResult=await verifyTool.execute({candidate:second});assert.equal(secondResult.native_commit.confirmed,true);
 }});
 const r=await x.k.submit({goal:'fix'});assert.equal(r.status,'VERIFIED');assert.equal(x.counts().nativeCalls,2);
}));
test('source drift AFTER sandbox check blocks native submission with no writes',()=>fixture(async cwd=>{
 let x;x=makeKernel(cwd,{verify:async()=>{await writeFile(join(cwd,'src/a.py'),'changed-before-commit\n');return pass;},onPrompt:async({tools})=>{
  const r=await tool(tools,'voder_verify_candidate').execute({candidate:first});assert.equal(r.status,'PASS');assert.equal(r.native_commit.status,'REJECTED');assert.equal(r.native_commit.code,'STALE_SOURCE');
 }});
 const r=await x.k.submit({goal:'fix'});assert.equal(r.status,'BLOCKED');assert.equal(x.counts().nativeCalls,0);
 assert.equal(await readFile(join(cwd,'src/a.py'),'utf8'),'changed-before-commit\n');
}));
test('async in-flight native action blocks competing candidate before second Modal run',()=>fixture(async cwd=>{
 let unblock;const gate=new Promise(r=>{unblock=r});let x,started;
 const nativeStarted=new Promise(r=>{started=r});
 x=makeKernel(cwd,{dispatch:async()=>{started();await gate;return {status:'COMMITTED'};},onPrompt:async({tools})=>{
  const verifyTool=tool(tools,'voder_verify_candidate');const pending=verifyTool.execute({candidate:first});await nativeStarted;
  await assert.rejects(verifyTool.execute({candidate:second}),{code:'NATIVE_ACTION_IN_FLIGHT'});
  unblock();assert.equal((await pending).native_commit.confirmed,true);
 }});
 const r=await x.k.submit({goal:'fix'});assert.equal(r.status,'VERIFIED');assert.equal(x.counts().verificationCalls,1);assert.equal(x.counts().nativeCalls,1);
}));
test('uncertain native response blocks automatic model relay and all future candidates',()=>fixture(async cwd=>{
 let x;x=makeKernel(cwd,{dispatch:async()=>({status:'OBSERVED'}),onPrompt:async({tools})=>{
  const r=await tool(tools,'voder_verify_candidate').execute({candidate:first});assert.equal(r.status,'PASS');assert.equal(r.next_step.status,'RECONCILE_NATIVE');
  await assert.rejects(tool(tools,'voder_verify_candidate').execute({candidate:second}),{code:'NATIVE_OUTCOME_UNCERTAIN'});
 }});
 const r=await x.k.submit({goal:'fix'});assert.equal(r.status,'BLOCKED');assert.equal(r.blocked_reason,'NATIVE_OUTCOME_UNCERTAIN');assert.equal(x.counts().switches,0);assert.equal(x.counts().nativeCalls,1);
}));
test('unavailable native broker must never synthesize a success',()=>fixture(async cwd=>{
 let x;x=makeKernel(cwd,{dispatch:async()=>{throw Object.assign(new Error('offline'),{code:'NATIVE_DOWN'})},onPrompt:async({tools})=>{
  const r=await tool(tools,'voder_verify_candidate').execute({candidate:first});assert.equal(r.status,'PASS');assert.deepEqual(r.native_commit,{status:'REJECTED',confirmed:false,code:'NATIVE_DOWN'});
 }});
 const r=await x.k.submit({goal:'fix'});assert.equal(r.status,'BLOCKED');assert.equal(x.counts().nativeCalls,1);
}));
