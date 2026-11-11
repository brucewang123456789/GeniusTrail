/** OFFLINE FIXTURES only; independent Rust observer is intentionally mocked, never called live. */
import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash,randomUUID} from 'node:crypto';
import {mkdtemp,mkdir,writeFile,rename} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join,dirname} from 'node:path';
import {VoderJournal} from '../src/control-plane.mjs';
import {readNativeSessionDiff} from '../src/native-diff.mjs';
const sha=x=>createHash('sha256').update(x).digest('hex');
async function fixture({before='old\n',after='new\n',trusted=true}={}){
 const dir=await mkdtemp(join(tmpdir(),'voder-diff-test-'));
 const ws=join(dir,'workspace'),store=join(dir,'store');
 await mkdir(ws,{mode:0o700});await mkdir(store,{mode:0o700});
 const id=randomUUID(),action_id=`action-${randomUUID()}`,run_id=`run-${action_id}`,commit=sha('commit'+id);
 const journal=await new VoderJournal({dir:join(dir,'private'),workspace:ws}).initialize();await journal.create(id);
 const record=await journal.get(id);
 record.status='DELIVERED';record.result={status:'DELIVERED',verified:true,delivery_status:'DELIVERED',run_id,commit_digest:commit};
 record.native_actions=[{schema:'voder.dispatch.intent.v1',session_id:id,action_id,run_id,status:'VERIFIED_NATIVE_COMMIT'}];
 await journal.write(id,record);
 const path='src/example.txt';await mkdir(join(ws,'src'));await writeFile(join(ws,path),after);
 const policyFile=join(dir,'policy.json');await writeFile(policyFile,JSON.stringify({schema:'voder.gateway.policy.v1',repository_path:ws,target_path:path}));
 const candidate=join(store,'attempts',action_id,'owned-worktrees','candidate','src');
 await mkdir(candidate,{recursive:true});await writeFile(join(candidate,'example.txt'),after);
 const proofDir=join(store,'attempts',action_id,'verified-delivery');await mkdir(proofDir,{recursive:true,mode:0o700});
 const proof={schema:'voder.verified-delivery.v1',session_id:id,action_id,run_id,commit_digest:commit,
  journal_digest:sha('journal'),relative_path:path,before_sha256:sha(before),after_sha256:sha(after)};
 await writeFile(join(proofDir,'before.bin'),before);
 await writeFile(join(proofDir,'intent.json'),JSON.stringify({...proof,status:'INTENT'}));
 await writeFile(join(proofDir,'delivered.json'),JSON.stringify({...proof,status:'DELIVERED'}));
 let reads=0;
 const commitVerifier={async confirmCommit(receipt,input){reads++;
  if(!trusted)throw Object.assign(Error('Rust observer unavailable'),{code:'NATIVE_OBSERVATION_FAILED'});
  assert.equal(receipt.session_id,id);assert.equal(receipt.run_id,run_id);assert.equal(input.action_id,action_id);
  return {verified:true,authority:'VODER_NATIVE_VGR_STATE',session_id:id,action_id,run_id,commit_digest:commit};}};
 const run=()=>readNativeSessionDiff({journal,id,attemptStore:store,policyFile,commitVerifier});
 return {run,ws,store,policyFile,id,action_id,commit,proofDir,journal,proof,reads:()=>reads};
}
test('OFFLINE: session-bound proof returns exact before/after only after modeled Rust observer',async()=>{
 const f=await fixture();const out=await f.run();assert.equal(out.status,'VERIFIED_DELIVERED');
 assert.equal(out.authority,'VODER_NATIVE_VGR_STATE');assert.equal(out.before,'old\n');assert.equal(out.after,'new\n');assert.equal(out.sessionId,f.id);assert.equal(f.reads(),1);
});
test('OFFLINE: Rust observer unavailable causes hard rejection',async()=>{
 const f=await fixture({trusted:false});await assert.rejects(f.run(),e=>e.code==='NATIVE_OBSERVATION_FAILED');
});
test('OFFLINE: changed workspace bytes never get promoted to verified diff',async()=>{
 const f=await fixture();await writeFile(join(f.ws,'src/example.txt'),'tamper');
 await assert.rejects(f.run(),e=>e.code==='DIFF_WORKTREE_CHANGED');assert.equal(f.reads(),0);
});
test('OFFLINE: altered before image never becomes valid historical diff',async()=>{
 const f=await fixture();await writeFile(join(f.proofDir,'before.bin'),'tamper');
 await assert.rejects(f.run(),e=>e.code==='DIFF_WORKTREE_CHANGED');
});
test('OFFLINE: invalid session-scoped target path cannot access outside workspace',async()=>{
 const f=await fixture();const proof={...f.proof,relative_path:'../private/proof'};
 await writeFile(join(f.proofDir,'delivered.json'),JSON.stringify({...proof,status:'DELIVERED'}));
 await assert.rejects(f.run(),e=>e.code==='DIFF_PROOF_MISMATCH');
});
test('OFFLINE: malicious symlink in workspace rejected',async()=>{
 const f=await fixture();const {symlink,unlink}=await import('node:fs/promises');
 await unlink(join(f.ws,'src/example.txt'));await symlink('/etc/passwd',join(f.ws,'src/example.txt'));
 await assert.rejects(f.run(),e=>e.code==='DIFF_UNSAFE_PATH');
});
test('OFFLINE: other session cannot borrow committed native proof',async()=>{
 const f=await fixture();const next=randomUUID();await f.journal.create(next);
 const result=await readNativeSessionDiff({journal:f.journal,id:next,attemptStore:f.store,commitVerifier:{confirmCommit(){throw Error('never')}}});
 assert.equal(result.status,'NOT_AVAILABLE');
});
test('OFFLINE: side-effect-free read declines unverified drafts and missing native driver',async()=>{
 const f=await fixture();const out=await readNativeSessionDiff({journal:f.journal,id:f.id});assert.equal(out.status,'NOT_AVAILABLE');
});
test('OFFLINE: empty before-image is valid when native-delivered bytes match',async()=>{
 const f=await fixture({before:'',after:'hello'});assert.equal((await f.run()).before,'');
});
test('OFFLINE: workspace policy mismatch cannot launder native output',async()=>{
 const f=await fixture();await writeFile(f.policyFile,JSON.stringify({schema:'voder.gateway.policy.v1',repository_path:f.ws,target_path:'src/other.txt'}));
 await assert.rejects(f.run(),e=>e.code==='DIFF_GATEWAY_POLICY');assert.equal(f.reads(),0);
});
test('OFFLINE: candidate worktree tampering is rejected independently of delivered.json',async()=>{
 const f=await fixture();const candidate=join(f.store,'attempts',f.action_id,'owned-worktrees','candidate','src','example.txt');
 await writeFile(candidate,'postcommit tamper');
 await assert.rejects(f.run(),e=>e.code==='DIFF_WORKTREE_CHANGED');assert.equal(f.reads(),0);
});
test('OFFLINE: workspace changing during Rust observation cannot produce stale diff',async()=>{
 const f=await fixture();await assert.rejects(
  readNativeSessionDiff({journal:f.journal,id:f.id,attemptStore:f.store,policyFile:f.policyFile,
   commitVerifier:{async confirmCommit(receipt){
     await writeFile(join(f.ws,'src/example.txt'),'changed while reading');
     return {verified:true,authority:'VODER_NATIVE_VGR_STATE',session_id:f.id,action_id:f.action_id,
       run_id:receipt.run_id,commit_digest:f.commit};
   }}}),e=>e.code==='DIFF_WORKTREE_CHANGED');
});
