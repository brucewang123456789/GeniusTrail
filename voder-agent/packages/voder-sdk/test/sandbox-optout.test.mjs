import {test} from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {VoderSDK} from '../src/index.mjs';

test('operator opted out of sandbox: draft is delivered UNVERIFIED without privileged side effects',async()=>{
  const cwd=await mkdtemp(join(tmpdir(),'voder-draft-'));let nativeActions=0;let tools;
  try{
    const sdk=new VoderSDK({cwd,allowUnverifiedDraft:true,native:{async executeAuthorized(){nativeActions++;throw Error('Forbidden')}},
      modelHost:{async createSession(o){tools=o.tools;return {async prompt(){
        await assert.rejects(tools[0].execute({kind:'patch',relative_path:'a.txt',old_sha256:'0'.repeat(64),new_content:'x'}),{code:'SANDBOX_DISABLED'});
        return {draft:'Untested code sample'};
      },dispose(){}}}}});
    const id=sdk.createSession({id:'opt-out'}).sessionId;
    const result=await sdk.submitTask(id,{goal:'suggest code'});
    assert.equal(result.status,'DRAFT_UNVERIFIED');assert.equal(result.verified,false);assert.equal(result.sandbox,'DISABLED');
    assert.equal(result.draft,'Untested code sample');assert.equal(nativeActions,0);
    assert.equal(sdk.getStatus(id),'DRAFT_UNVERIFIED');sdk.dispose();
  }finally{await rm(cwd,{recursive:true,force:true});}
});
test('default remains fail closed, verified path cannot opt into draft-only mode',async()=>{
  const cwd=await mkdtemp(join(tmpdir(),'voder-draft-gate-'));
  try{
    const host={async createSession(){return {async prompt(){return {draft:'unverified'}},dispose(){}}}};
    const sdk=new VoderSDK({cwd,native:{async executeAuthorized(){}},modelHost:host});const {sessionId}=sdk.createSession();
    assert.equal((await sdk.submitTask(sessionId,{goal:'test'})).status,'BLOCKED');sdk.dispose();
    assert.throws(()=>new VoderSDK({cwd,allowUnverifiedDraft:true,candidateVerifier:{verifyCandidate(){}},native:{async executeAuthorized(){}},modelHost:host}).createSession(),{code:'DRAFT_POLICY_CONFLICT'});
  }finally{await rm(cwd,{recursive:true,force:true});}
});
test('branded upstream Pi adapter exposes final assistant text only as unverified draft',async()=>{
 const cwd=await mkdtemp(join(tmpdir(),'voder-draft-upstream-'));
 try{
   const {PiModelHost}=await import('../../voder-agent-core/src/pi-host.mjs');
   const upstream={getAgentDir:()=>'/trusted',SessionManager:{inMemory:()=>({})},DefaultResourceLoader:class{async reload(){}},defineTool:x=>x,
     async createAgentSession(o){const names=o.tools;return {session:{messages:[],getActiveToolNames:()=>names,getCallableToolNames:()=>names,
       getAllTools:()=>names.map(name=>({name,exposure:'direct'})),
       async prompt(){this.messages.push({role:'assistant',stopReason:'stop',content:[{type:'text',text:'untested hello world'}]})},
       dispose(){},subscribe(){return()=>{}}}}}};
   const sdk=new VoderSDK({cwd,allowUnverifiedDraft:true,native:{async executeAuthorized(){throw Error('native must not be called')}},modelHost:new PiModelHost({upstream})});
   const id=sdk.createSession().sessionId;
   const r=await sdk.submitTask(id,{goal:'write a code sketch'});
   assert.equal(r.status,'DRAFT_UNVERIFIED');assert.equal(r.draft,'untested hello world');assert.equal(r.nativeCommit,undefined);
   sdk.dispose();
 }finally{await rm(cwd,{recursive:true,force:true})}
});
