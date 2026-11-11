import {test} from 'node:test';
import assert from 'node:assert/strict';
import {EvidenceScheduler,candidateDigest,normalizedCandidate,assertCandidateVerifier} from '../src/index.mjs';
const sha='a'.repeat(64);
const candidate={relative_path:'src/main.py',old_sha256:sha,new_content:'print(1)'};
const sourceDigest=async()=>sha;
test('candidate identity includes source, path and exact proposed bytes',()=>{
 assert.notEqual(candidateDigest(candidate),candidateDigest({...candidate,new_content:'print(2)'}));
 assert.notEqual(candidateDigest(candidate),candidateDigest({...candidate,relative_path:'src/other.py'}));
 assert.throws(()=>normalizedCandidate({...candidate,relative_path:'../etc/passwd'}),{code:'INVALID_CANDIDATE'});
 assert.equal(normalizedCandidate({...candidate,new_content:'x'.repeat(131073)}).new_content.length,131073);
});
test('old source must match before candidate sandbox executes',async()=>{
 const e=new EvidenceScheduler();let called=0;
 await assert.rejects(e.inspectCandidate(candidate,{sourceDigest:async()=> 'b'.repeat(64),run:async()=>{called++}}),{code:'STALE_SOURCE'});
 assert.equal(called,0);
});
test('PASS cannot be a model string or a malformed sandbox response',async()=>{
 const e=new EvidenceScheduler();
 await assert.rejects(e.inspectCandidate(candidate,{sourceDigest,run:async()=>({status:'PASS',backend:'fake',checks:[]})}),{code:'INVALID_SANDBOX_RECEIPT'});
 await assert.rejects(e.inspectAction(candidate,async()=>({status:'COMMITTED'}),{verificationRequired:true,sourceDigest}),{code:'VERIFY_BEFORE_COMMIT'});
});
test('native action is denied unless exact verified candidate bytes match',async()=>{
 const e=new EvidenceScheduler();let dispatched=0;
 await e.inspectCandidate(candidate,{sourceDigest,run:async()=>({status:'PASS',backend:'isolated-test',sandboxed:true,evidence_digest:sha,checks:[{name:'python.test',passed:true}]})});
 await assert.rejects(e.inspectAction({...candidate,new_content:'print(0)'},async()=>{dispatched++},{verificationRequired:true,sourceDigest}),{code:'VERIFY_BEFORE_COMMIT'});
 assert.equal(dispatched,0);
 assert.equal((await e.inspectAction(candidate,async()=>{dispatched++;return {status:'COMMITTED'}},{verificationRequired:true,sourceDigest})).status,'COMMITTED');
 assert.equal(dispatched,1);
 await assert.rejects(e.inspectAction(candidate,async()=>{dispatched++},{verificationRequired:true,sourceDigest}),{code:'DUPLICATE_ACTION'});
 assert.equal(dispatched,1);
});
test('failed sandbox candidate cannot be resubmitted unchanged; new root-cause patch can pass',async()=>{
 const e=new EvidenceScheduler();let probes=0;
 const reply=async(_,d)=>{probes++;return probes===1 ? {status:'FAIL',backend:'modal',diagnostic:'AssertionError: expected 2, got 1'} : {status:'PASS',backend:'modal',sandboxed:true,evidence_digest:d,checks:[{name:'unit',passed:true}]};};
 assert.equal((await e.inspectCandidate(candidate,{sourceDigest,run:reply})).status,'FAIL');
 await assert.rejects(e.inspectCandidate(candidate,{sourceDigest,run:reply}),{code:'DUPLICATE_CANDIDATE'});
 const revised={...candidate,new_content:'print(2)'};
 assert.equal((await e.inspectCandidate(revised,{sourceDigest,run:reply})).status,'PASS');
 assert.equal(probes,2);assert.equal(e.snapshot().positive_sandbox_checks,1);
});
test('unknown sandbox outcome cannot be auto-replayed',async()=>{
 const e=new EvidenceScheduler();let calls=0;
 await assert.rejects(e.inspectCandidate(candidate,{sourceDigest,run:async()=>{calls++;throw Error('timeout')}}),/timeout/);
 await assert.rejects(e.inspectCandidate(candidate,{sourceDigest,run:async()=>{calls++}}),{code:'DUPLICATE_CANDIDATE'});
 assert.equal(calls,1);
});
test('repository evidence prevents duplicate context reads without new observations',async()=>{
 const e=new EvidenceScheduler();let calls=0;
 const run=async()=>{calls++;return {path:'src/x',content:'hello'}};
 await e.inspectRead('voder_read_file',{path:'src/x'},run);
 await assert.rejects(e.inspectRead('voder_read_file',{path:'src/x'},run),{code:'DUPLICATE_NO_PROGRESS'});
 assert.equal(calls,1);
});
test('budget prevents unlimited reads and no arbitrary truncation',async()=>{
 const e=new EvidenceScheduler({maxToolCalls:4,maxReadBytes:1024});
 for(let i=0;i<2;i++)await e.inspectRead('voder_read_file',{path:String(i)},async()=>({id:i}));
 await assert.rejects(e.inspectRead('voder_read_file',{path:'third'},async()=>({})),{code:'READ_BUDGET_RESERVED'});
 assert.equal(e.snapshot().remaining_tool_calls,2);
 assert.throws(()=>new EvidenceScheduler({maxToolCalls:0}),{code:'INVALID_ORCHESTRATION_POLICY'});
 const huge=new EvidenceScheduler({maxReadBytes:1024});
 await assert.rejects(huge.inspectRead('voder_read_file',{path:'x'},async()=>({content:'X'.repeat(1025)})),{code:'EVIDENCE_BYTE_BUDGET'});
});
test('adapter must be host-installed callable and cannot be a model string',()=>{
 assert.throws(()=>assertCandidateVerifier('USE_MODAL'),{code:'INVALID_SANDBOX_ADAPTER'});
 assert.doesNotThrow(()=>assertCandidateVerifier({verifyCandidate(){}}));
});
test('concurrent duplicate candidate is reserved before asynchronous source I/O',async()=>{
 const e=new EvidenceScheduler();let readCount=0,runCount=0;
 let resolveSource;
 const gate=new Promise(resolve=>{resolveSource=resolve});
 const source=async()=>{readCount++;await gate;return sha;};
 const first=e.inspectCandidate(candidate,{sourceDigest:source,run:async()=>{runCount++;return {status:'FAIL',backend:'isolated',diagnostic:'test failed'}}});
 await assert.rejects(e.inspectCandidate(candidate,{sourceDigest:source,run:async()=>{runCount++}}),{code:'DUPLICATE_CANDIDATE'});
 resolveSource();await first;
 assert.equal(readCount,1);assert.equal(runCount,1);
});
test('concurrent duplicate repository reads cannot issue a second I/O call',async()=>{
 const e=new EvidenceScheduler();let called=0,release;
 const hold=new Promise(resolve=>{release=resolve});
 const source=async()=>{called++;await hold;return {content:'ok'}};
 const first=e.inspectRead('voder_read_file',{path:'src/x'},source);
 await assert.rejects(e.inspectRead('voder_read_file',{path:'src/x'},source),{code:'DUPLICATE_NO_PROGRESS'});
 release();await first;assert.equal(called,1);
});
test('repeated invalid no-progress calls eventually stop',async()=>{
 const e=new EvidenceScheduler({maxNoProgress:2});
 await e.inspectRead('voder_read_file',{path:'a'},async()=>({x:1}));
 for(let i=0;i<2;i++)await assert.rejects(e.inspectRead('voder_read_file',{path:'a'},async()=>({x:1})),{code:'DUPLICATE_NO_PROGRESS'});
 await assert.rejects(e.inspectRead('voder_read_file',{path:'b'},async()=>({x:2})),{code:'NO_EVIDENCE_PROGRESS'});
});
