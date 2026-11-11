import {test} from 'node:test';
import assert from 'node:assert/strict';
import {EvidenceScheduler} from '../src/index.mjs';
const sha='a'.repeat(64),candidate={relative_path:'src/main.py',old_sha256:sha,new_content:'print(2)'};
const sourceDigest=async()=>sha;
const pass=async()=>({status:'PASS',backend:'offline-fixture',sandboxed:true,evidence_digest:sha,checks:[{name:'fixture',passed:true}]});
const verify=e=>e.inspectCandidate(candidate,{sourceDigest,run:pass});
const commit=e=>e.inspectAction(candidate,async()=>({status:'COMMITTED'}),{verificationRequired:true,sourceDigest});

test('historical 90-read/6-error pattern cannot consume final verification and submission capacity',async()=>{
 const e=new EvidenceScheduler();let admitted=0;
 for(let i=0;i<96;i++){
  try{await e.inspectRead('voder_search_code',{path:`file-${i}`},async()=>{admitted++;if(i%16===15)throw Object.assign(Error('legacy wrong type'),{code:'INVALID_PATH'});return {i};});}
  catch(error){assert.ok(['INVALID_PATH','READ_BUDGET_RESERVED'].includes(error.code));}
 }
 assert.equal(admitted,83);assert.equal(e.snapshot().phase,'FINALIZING');
 assert.equal(e.snapshot().remaining_tool_calls,13);
 assert.equal((await verify(e)).status,'PASS');assert.equal((await commit(e)).status,'COMMITTED');
 assert.ok(e.snapshot().tool_calls<=96);assert.equal(e.snapshot().verification_calls,1);
});
test('historical no-progress trip stops exploration but not candidate validation or explicit commit',async()=>{
 const e=new EvidenceScheduler();
 const results=await Promise.allSettled(Array.from({length:10},(_,i)=>e.inspectRead('voder_search_code',{path:`invalid-${i}`},async()=>{throw Object.assign(Error('bad path'),{code:'INVALID_PATH'});}))); 
 assert.equal(results.filter(x=>x.status==='rejected').length,10);
 await assert.rejects(e.inspectRead('voder_read_file',{path:'correct'},async()=>({ok:1})),{code:'NO_EVIDENCE_PROGRESS'});
 assert.equal((await verify(e)).status,'PASS');assert.equal((await commit(e)).status,'COMMITTED');
 // A successful check never reopens exhausted exploration or resets spent calls.
 await assert.rejects(e.inspectRead('voder_read_file',{path:'again'},async()=>({ok:2})),{code:'NO_EVIDENCE_PROGRESS'});
 assert.equal(e.snapshot().tool_calls,12);
});
test('200 concurrent reads preserve one hard ledger and exactly the derived reserve',async()=>{
 const e=new EvidenceScheduler();let ran=0,release;
 const gate=new Promise(r=>release=r);
 const pending=Array.from({length:200},(_,i)=>e.inspectRead('voder_read_file',{path:`src/${i}`},async()=>{ran++;await gate;return {i};}));
 const outcomes=Promise.allSettled(pending);release();const results=await outcomes;
 assert.equal(ran,83);assert.equal(results.filter(x=>x.status==='fulfilled').length,83);
 assert.equal(e.snapshot().remaining_tool_calls,13);
 await verify(e);await commit(e);assert.equal(e.snapshot().tool_calls,85);
});
test('read byte exhaustion preserves verification without truncation or budget reset',async()=>{
 const e=new EvidenceScheduler({maxReadBytes:1024});
 await assert.rejects(e.inspectRead('voder_read_file',{path:'x'},async()=>({content:'x'.repeat(1025)})),{code:'EVIDENCE_BYTE_BUDGET'});
 assert.equal(e.snapshot().read_bytes,0);assert.equal(e.snapshot().phase,'FINALIZING');
 await verify(e);await commit(e);assert.equal(e.snapshot().tool_calls,3);
});
test('reserve supports bounded FAIL -> revised PASS -> exact commit',async()=>{
 const e=new EvidenceScheduler({maxToolCalls:16,maxVerificationCalls:2});
 for(let i=0;i<13;i++)await e.inspectRead('voder_read_file',{path:String(i)},async()=>({i}));
 await e.inspectCandidate({...candidate,new_content:'bad'},{sourceDigest,run:async()=>({status:'FAIL',backend:'offline-fixture',diagnostic:'fixture failed'})});
 await verify(e);await commit(e);assert.equal(e.snapshot().tool_calls,16);
 await assert.rejects(e.inspectAction({...candidate,new_content:'other'},async()=>({status:'COMMITTED'})),{code:'TOOL_BUDGET_EXHAUSTED'});
});
test('verification never spends the last native slot and no PASS is invented',async()=>{
 const e=new EvidenceScheduler({maxToolCalls:4});
 await e.inspectRead('voder_read_file',{path:'a'},async()=>({a:1}));
 await e.inspectRead('voder_read_file',{path:'b'},async()=>({b:1}));
 await e.inspectCandidate(candidate,{sourceDigest,run:async()=>({status:'FAIL',backend:'offline-fixture'})});
 await assert.rejects(e.inspectCandidate({...candidate,new_content:'fixed'},{sourceDigest,run:pass}),{code:'COMMIT_BUDGET_RESERVED'});
 await assert.rejects(commit(e),{code:'VERIFY_BEFORE_COMMIT'});
 assert.equal(e.snapshot().positive_sandbox_checks,0);
});
test('reserve does not admit stale bytes, unknown outcomes, duplicate commits or unauthorized candidates',async()=>{
 const e=new EvidenceScheduler({maxNoProgress:2});e.finalize('NO_EVIDENCE_PROGRESS');let calls=0;
 await assert.rejects(e.inspectCandidate({...candidate,relative_path:'../escape'},{sourceDigest,run:pass}),{code:'INVALID_CANDIDATE'});
 await assert.rejects(e.inspectCandidate(candidate,{sourceDigest:async()=> 'b'.repeat(64),run:async()=>{calls++;return pass();}}),{code:'STALE_SOURCE'});
 await assert.rejects(commit(e),{code:'VERIFY_BEFORE_COMMIT'});
 assert.equal(calls,0);assert.equal(e.snapshot().attempted_native_actions,0);
});
test('new sandbox failures fund at most two focused repair reads without resetting the ledger',async()=>{
 const e=new EvidenceScheduler();e.finalize('NO_EVIDENCE_PROGRESS');
 for(let i=0;i<3;i++){
  await e.inspectCandidate({...candidate,new_content:`failed-${i}`},{sourceDigest,run:async()=>({status:'FAIL',backend:'offline-fixture',diagnostic:`distinct error ${i}`})});
  if(i<2)await e.inspectRead('voder_read_file',{path:`repair-${i}`},async()=>({i}));
  else await assert.rejects(e.inspectRead('voder_read_file',{path:'third-repair'},async()=>({i})),{code:'NO_EVIDENCE_PROGRESS'});
 }
 assert.equal(e.snapshot().repair_reads,2);assert.equal(e.snapshot().repair_read_credits,0);
 assert.equal(e.snapshot().tool_calls,5);await verify(e);await commit(e);
 assert.equal(e.snapshot().tool_calls,7);assert.equal(e.snapshot().phase,'FINALIZING');
});
test('identical failed diagnostic cannot repeatedly fund repair reads',async()=>{
 const e=new EvidenceScheduler();e.finalize('READ_BUDGET_RESERVED');
 const run=async()=>({status:'FAIL',backend:'offline-fixture',diagnostic:'same error'});
 await e.inspectCandidate(candidate,{sourceDigest,run});
 await e.inspectRead('voder_read_file',{path:'repair'},async()=>({i:1}));
 await e.inspectCandidate({...candidate,new_content:'different bytes'},{sourceDigest,run});
 await assert.rejects(e.inspectRead('voder_read_file',{path:'another'},async()=>({i:2})),{code:'READ_BUDGET_RESERVED'});
 assert.equal(e.snapshot().repair_reads,1);
});
