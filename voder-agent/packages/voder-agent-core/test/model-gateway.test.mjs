import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile, symlink, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { AgentKernel } from '../src/kernel.mjs';
import { VgrBroker } from '../src/broker.mjs';
const HEX='f'.repeat(64);

test('model cannot grant identity, scope or budget; host constructs authoritative action',async()=>{
 let seen,called=0;
 const native={async executeAuthorized(action){seen=action;called++;return {
   authority:'VODER_NATIVE_VGR',status:'OBSERVED',action_id:action.action_id,
   session_id:action.session_id,journal_digest:HEX
 };}};
 const broker=new VgrBroker({native,sessionId:'host-session',workspaceRoot:'/tmp'});
 const modelHost={async createSession({tools}){return {
   prompt:async()=>{await tools[0].execute({kind:'patch',relative_path:'src/test.txt',old_sha256:HEX,
      new_content:'hello',session_id:'model-forged',action_id:'model-forged',
      idempotency_key:'model-forged',budget:{max_runtime_ms:999999}},new AbortController().signal)},
   dispose(){}
 };}};
 const kernel=new AgentKernel({modelHost,broker,cwd:'/tmp',sessionId:'host-session'});
 const result=await kernel.submit({goal:'change source',maxActions:1});
 assert.equal(result.status,'BLOCKED');assert.equal(called,1);
 assert.notEqual(seen.action_id,'model-forged');assert.notEqual(seen.idempotency_key,'model-forged');
 assert.equal(seen.session_id,'host-session');assert.equal(seen.budget.max_runtime_ms,120000);
 assert.deepEqual(seen.workspace_scope,['src/test.txt']);
});

test('model cannot create a second native side effect after action budget used',async()=>{
 let called=0;
 const native={executeAuthorized:async a=>{called++;return {authority:'VODER_NATIVE_VGR',status:'OBSERVED',action_id:a.action_id,session_id:a.session_id,journal_digest:HEX}}};
 const broker=new VgrBroker({native,sessionId:'host-session',workspaceRoot:'/tmp'});
 const modelHost={async createSession({tools}){return {prompt:async()=>{
   const a={kind:'patch',relative_path:'src/test.txt',old_sha256:HEX,new_content:'good'};
   await tools[0].execute(a);
   await assert.rejects(tools[0].execute(a),{code:'BUDGET_EXHAUSTED'});
 },dispose(){}};}};
 const result=await new AgentKernel({modelHost,broker,cwd:'/tmp',sessionId:'host-session'}).submit({goal:'bounded',maxActions:1});
 assert.equal(result.status,'BLOCKED');assert.equal(called,1);
});

test('trusted source preflight refuses symlink escape instead of injecting secret into model',async()=>{
 const dir=await mkdtemp(join(tmpdir(),'voder-source-'));
 try{
  await mkdir(join(dir,'src'));
  await symlink('/etc/passwd',join(dir,'src','secret.txt'));
  const broker={native:{targetPath:'src/secret.txt'}};
  let modelCalls=0;
  const modelHost={async createSession(){modelCalls++;return {prompt(){},dispose(){}}}};
  const kernel=new AgentKernel({modelHost,broker,cwd:dir,sessionId:'safe-session'});
  await assert.rejects(kernel.submit({goal:'inspect'}),{code:'UNSAFE_SOURCE'});
  assert.equal(modelCalls,0);
 }finally{await rm(dir,{recursive:true,force:true})}
});
