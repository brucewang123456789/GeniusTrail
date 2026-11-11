import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,readFile,writeFile,lstat} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {spawn} from 'node:child_process';
import {VoderSDK,VoderControlPlane,VoderJournal} from '../src/index.mjs';
import {NativeSdkContinuation} from '../src/native-continuation.mjs';
const mk=async()=>mkdtemp(join(tmpdir(),'voder-safe-resume-'));
const goal='repair the fixture safely';
const draftHost={async createSession(){return {async prompt(){return {settled:true,draft:'safe restart answer'};},async abort(){},dispose(){}};}};
const native={async executeAuthorized(){throw Error('MUST_NOT_DISPATCH');}};
const resolver=async()=>({goal});
const make=async(cwd,verdict)=>{
 const sdk=new VoderSDK({cwd,native,modelHost:draftHost,allowUnverifiedDraft:true});
 const journal=await new VoderJournal({dir:join(cwd,'private'),workspace:cwd}).initialize();
 const rec=Object.create(NativeSdkContinuation.prototype);
 rec.reconcile=async a=>verdict(a);
 return {host:new VoderControlPlane({sdk,journal,nativeContinuation:rec}),journal};
};
async function killChild(){
 const cwd=await mk(),childFile=new URL('./recovery-crash-child.mjs',import.meta.url);
 const child=spawn(process.execPath,[childFile.pathname,cwd],{stdio:['ignore','pipe','pipe']});
 let stderr='';child.stderr.on('data',c=>stderr+=c);
 let proof;
 for(let i=0;i<200;i++){
  try{proof=JSON.parse(await readFile(join(cwd,'native-started.json'),'utf8'));break;}
  catch{if(child.exitCode!==null)throw Error('CHILD_EXITED '+stderr);await new Promise(r=>setTimeout(r,20));}
 }
 assert(proof,'Native fixture never started: '+stderr);
 child.kill('SIGKILL');const [code,signal]=await new Promise(r=>child.on('close',(...args)=>r(args)));
 assert.equal(signal,'SIGKILL');
 const id=(await readFile(join(cwd,'session-id'),'utf8')).trim();
 const journal=await new VoderJournal({dir:join(cwd,'private'),workspace:cwd}).initialize();
 const rec=await journal.get(id);
 assert.equal(rec.status,'INSPECTING');
 assert.equal(rec.native_actions.length,1);
 assert.equal(rec.native_actions[0].action_id,proof.action_id);
 assert.equal(rec.native_actions[0].status,'DISPATCH_INTENT');
 assert.equal(rec.recovery.restart_count,0);
 assert(rec.owner?.ticks);
 assert(!JSON.stringify(rec).includes(goal),'Task text leaked to metadata-only journal');
 return {cwd,id,journal};
}
test('actual SIGKILL retains pre-dispatch fsynced identity, Rust rollback permits fresh model-only replan',async()=>{
 const {cwd,id,journal}=await killChild();let checks=0;
 const {host}=await make(cwd,a=>{checks++;return {status:'ROLLED_BACK',action_id:a.action_id,authority:'VODER_NATIVE_VGR_RECOVERY',delivery_allowed:false};});
 const recovered=await host.autoRecover({taskResolver:resolver});
 assert.equal(recovered.length,1);assert.equal(recovered[0].status,'AUTO_REPLANNING');assert.equal(checks,1);
 const result=await host.wait(id);
 assert.equal(result.status,'DRAFT_UNVERIFIED');assert.equal(result.draft,'safe restart answer');
 const state=await journal.get(id);assert.equal(state.recovery.restart_count,1);assert.equal(state.native_actions.length,1);
 assert(!JSON.stringify(state).includes(goal));
 assert((await host.autoRecover({taskResolver:resolver})).length===0);
 host.dispose();
});
test('native committed after crash is never automatically resubmitted or declared delivered',async()=>{
 const {cwd,id,journal}=await killChild();let prompted=0;
 const {host}=await make(cwd,a=>({status:'COMMITTED',action_id:a.action_id,authority:'VODER_NATIVE_VGR_STATE'}));
 const result=await host.autoRecover({taskResolver:async()=>{prompted++;return {goal}}});
 assert.equal(result[0].status,'RECOVERY_REVIEW_REQUIRED');assert.equal(result[0].reason,'NATIVE_ACTION_COMMITTED_NO_AUTOREPLAY');
 assert.equal(prompted,0);assert.equal((await journal.get(id)).status,'RECOVERY_REVIEW_REQUIRED');host.dispose();
});
test('unknown or counterfeit native rollback stops with no model replay',async()=>{
 for(const outcome of [a=>({status:'UNKNOWN',action_id:a.action_id}),a=>({status:'ROLLED_BACK',action_id:a.action_id,authority:'FAKE',delivery_allowed:false})]){
  const {cwd,id}=await killChild();let resolved=0;
  const {host}=await make(cwd,outcome);
  const result=await host.autoRecover({taskResolver:async()=>{resolved++;return {goal}}});
  assert.equal(result[0].status,'RECOVERY_REVIEW_REQUIRED');assert.equal(resolved,0);host.dispose();
 }
});
test('task digest mismatch cannot resume even after genuine rollback; leaves original process state',async()=>{
 const {cwd,id,journal}=await killChild();
 const {host}=await make(cwd,a=>({status:'ROLLED_BACK',action_id:a.action_id,authority:'VODER_NATIVE_VGR_RECOVERY',delivery_allowed:false}));
 const result=await host.autoRecover({taskResolver:async()=>({goal:'different task'})});
 assert.equal(result[0].status,'RECOVERY_BLOCKED');assert.equal(result[0].reason,'TASK_IDENTITY_MISMATCH');
 assert.equal((await journal.get(id)).status,'INSPECTING');host.dispose();
});
test('live-process identity prevents a second instance from recovering current owner',async()=>{
 const cwd=await mk();const sdk=new VoderSDK({cwd,native,modelHost:{async createSession(){return {async prompt(){await new Promise(()=>{})}}}},allowUnverifiedDraft:true});
 const journal=await new VoderJournal({dir:join(cwd,'private'),workspace:cwd}).initialize();
 const owner=new VoderControlPlane({sdk,journal});const {sessionId}=await owner.createSession({autoContinue:true});
 await owner.submitTask(sessionId,{goal});const {host}=await make(cwd,()=>{throw Error('SHOULD_NOT_CALL_RUST')});
 const r=await host.autoRecover({taskResolver:resolver});assert.equal(r[0].reason,'ORIGINAL_PROCESS_STILL_ALIVE');
 owner.dispose();host.dispose();
});
test('cancellation intent persisted before process crash prohibits retry',async()=>{
 const {cwd,id,journal}=await killChild();let r=await journal.get(id);r.cancel_requested=true;await journal.write(id,r);
 const {host}=await make(cwd,()=>{throw Error('SHOULD_NOT_CALL_NATIVE')});
 const result=await host.autoRecover({taskResolver:resolver});assert.equal(result[0].reason,'USER_CANCELLATION_PRECEDES_CRASH');host.dispose();
});
test('durable intent write failure never crosses into native executeAuthorized',async()=>{
 const cwd=await mk();let calls=0;
 const nativeSpy={async executeAuthorized(){calls++;return {status:'COMMITTED'};}};
 const modelHost={async createSession({tools}){const act=tools.find(t=>t.name==='voder_authorized_action');return {async prompt(){await act.execute({kind:'patch',relative_path:'src/input.txt',old_sha256:'0'.repeat(64),new_content:'foo'});}};}};
 const sdk=new VoderSDK({cwd,native:nativeSpy,modelHost});
 const journal=await new VoderJournal({dir:join(cwd,'private'),workspace:cwd}).initialize();
 const host=new VoderControlPlane({sdk,journal});const {sessionId}=await host.createSession({autoContinue:true});
 const write=journal.write.bind(journal);
 journal.write=async(id,body)=>{if(body.native_actions?.length)throw Object.assign(Error('DISK_FULL'),{code:'ENOSPC'});return write(id,body);};
 await host.submitTask(sessionId,{goal});const outcome=await host.wait(sessionId);
 assert.equal(outcome.status,'JOURNAL_FAILED');assert.equal(calls,0);host.dispose();
});
test('opt-in rejects context silently lost on recovery and invalid privileged adapters',async()=>{
 const cwd=await mk(),sdk=new VoderSDK({cwd,native,modelHost:draftHost,allowUnverifiedDraft:true});
 const journal=await new VoderJournal({dir:join(cwd,'private'),workspace:cwd}).initialize();
 assert.throws(()=>new VoderControlPlane({sdk,journal,nativeContinuation:{reconcile:async()=>({status:'ROLLED_BACK'})}}),e=>e.code==='INVALID_RECOVERY_AUTHORITY');
 const host=new VoderControlPlane({sdk,journal});
 await assert.rejects(()=>host.createSession({autoContinue:true,memoryFiles:['/tmp/memory.json']}),e=>e.code==='RECOVERY_CONTEXT_UNSUPPORTED');
 host.dispose();
});

test('two independent recovery coordinators cannot both resume the same crashed task',async()=>{
 const {cwd,id,journal}=await killChild();
 let release;const barrier=new Promise(r=>release=r);let entered;
 const onEnter=new Promise(r=>entered=r);
 const a=await make(cwd,async action=>({status:'ROLLED_BACK',action_id:action.action_id,authority:'VODER_NATIVE_VGR_RECOVERY',delivery_allowed:false}));
 const b=await make(cwd,async action=>({status:'ROLLED_BACK',action_id:action.action_id,authority:'VODER_NATIVE_VGR_RECOVERY',delivery_allowed:false}));
 const first=a.host.autoRecover({taskResolver:async()=>{entered();await barrier;return {goal};}});
 await onEnter;
 const second=await b.host.autoRecover({taskResolver:resolver});
 assert(second.every(x=>x.status!=='AUTO_REPLANNING'));
 release();const r=await first;assert.equal(r[0].status,'AUTO_REPLANNING');
 await a.host.wait(id);a.host.dispose();b.host.dispose();
});
test('repeated SIGKILL does not permit a second automatic replan',async()=>{
 const {cwd,id,journal}=await killChild();
 let rec=await journal.get(id);rec.recovery.restart_count=1;await journal.write(id,rec);
 const {host}=await make(cwd,()=>{throw Error('NO_RUST_ALLOWED')});
 const r=await host.autoRecover({taskResolver:resolver});
 assert.equal(r[0].reason,'NO_AUTHORIZED_CHECKPOINT');host.dispose();
});
test('completed native action budget cannot be regranted by recovery',async()=>{
 const {cwd,id,journal}=await killChild();
 const r=await journal.get(id); // original task had default 12 actions; re-authorization cannot increase it.
 for(let k=1;k<12;k++)r.native_actions.push({...r.native_actions[0],action_id:`action-${k}`,run_id:`run-action-${k}`,idempotency_digest:'1'.repeat(63)+String(k%10)});
 await journal.write(id,r);
 const {host}=await make(cwd,a=>({status:'ROLLED_BACK',action_id:a.action_id,authority:'VODER_NATIVE_VGR_RECOVERY',delivery_allowed:false}));
 const response=await host.autoRecover({taskResolver:resolver});assert.equal(response[0].reason,'ACTION_BUDGET_EXHAUSTED');host.dispose();
});
