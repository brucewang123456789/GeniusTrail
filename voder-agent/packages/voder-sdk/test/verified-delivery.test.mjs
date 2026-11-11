import {test} from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,rm,symlink,chmod,stat} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {createHash} from 'node:crypto';
import {VerifiedWorkspaceDelivery} from '../src/verified-delivery.mjs';
import {VoderSDK} from '../src/index.mjs';
const hash=x=>createHash('sha256').update(x).digest('hex');
const before=Buffer.from('VODER_G2_HOST_BEFORE\n');
const after='VODER_G2_HOST_AFTER\n';
async function fixture(cb){
 const dir=await mkdtemp(join(tmpdir(),'voder-delivery-test-'));
 const repo=join(dir,'repo'),store=join(dir,'store'), policyFile=join(dir,'policy.json');
 const actionId='action-123';const actionRoot=join(store,'attempts',actionId);
 const candidate=join(actionRoot,'owned-worktrees','candidate');
 await mkdir(join(repo,'src'),{recursive:true});await mkdir(join(candidate,'src'),{recursive:true});
 await writeFile(join(repo,'src/input.txt'),before);await writeFile(join(candidate,'src/input.txt'),after);
 await writeFile(policyFile,JSON.stringify({schema:'voder.gateway.policy.v1',repository_path:repo,target_path:'src/input.txt'}));
 const input={schema:'voder.agent.action.v1',session_id:'session-test',action_id:actionId,kind:'patch',idempotency_key:'idem-1',workspace_scope:['src/input.txt'],
 evidence_obligations:['gateway-requirement'],budget:{max_runtime_ms:120000},relative_path:'src/input.txt',old_sha256:hash(before),new_content:after};
 const receipt={status:'COMMITTED',authority:'VODER_NATIVE_VGR',session_id:input.session_id,action_id:actionId,run_id:`run-${actionId}`,
 commit_digest:'a'.repeat(64),journal_digest:'b'.repeat(64),workspace_path:candidate};
 const delivery=new VerifiedWorkspaceDelivery({cwd:repo,attemptStore:store,policyFile});
 try{return await cb({dir,repo,store,policyFile,candidate,actionRoot,input,receipt,delivery})}finally{await rm(dir,{recursive:true,force:true})}
}
test('verified delivery: Rust-committed candidate is safely and durably delivered to original workspace',()=>fixture(async f=>{
 const result=await f.delivery.deliver({input:f.input,receipt:f.receipt});
 assert.equal(result.status,'DELIVERED');assert.equal(result.after_sha256,hash(after));
 assert.equal(await readFile(join(f.repo,'src/input.txt'),'utf8'),after);
 const evidence=JSON.parse(await readFile(join(f.actionRoot,'verified-delivery/delivered.json')));
 assert.equal(evidence.commit_digest,f.receipt.commit_digest);assert.equal(evidence.after_sha256,hash(after));
 assert.deepEqual(await readFile(join(f.actionRoot,'verified-delivery/before.bin')),before);
 await assert.rejects(f.delivery.deliver({input:f.input,receipt:f.receipt}),{code:'DELIVERY_ALREADY_ATTEMPTED'});
}));
test('delivery refuses stale user edit without overwriting or creating an intent',()=>fixture(async f=>{
 await writeFile(join(f.repo,'src/input.txt'),'user-edited\n');
 await assert.rejects(f.delivery.deliver({input:f.input,receipt:f.receipt}),{code:'DELIVERY_CONFLICT'});
 assert.equal(await readFile(join(f.repo,'src/input.txt'),'utf8'),'user-edited\n');
}));
test('delivery refuses tampered native candidate bytes',()=>fixture(async f=>{
 await writeFile(join(f.candidate,'src/input.txt'),'malicious\n');
 await assert.rejects(f.delivery.deliver({input:f.input,receipt:f.receipt}),{code:'DELIVERY_CANDIDATE'});
 assert.deepEqual(await readFile(join(f.repo,'src/input.txt')),before);
}));
test('delivery refuses unauthenticated receipt and mismatched policy',()=>fixture(async f=>{
 await assert.rejects(f.delivery.deliver({input:f.input,receipt:{...f.receipt,action_id:'wrong'}}),{code:'DELIVERY_UNVERIFIED'});
 await writeFile(f.policyFile,JSON.stringify({schema:'voder.gateway.policy.v1',repository_path:f.repo,target_path:'other/name.txt'}));
 await assert.rejects(f.delivery.deliver({input:f.input,receipt:f.receipt}),{code:'DELIVERY_POLICY'});
}));
test('delivery never follows symlinked original target',()=>fixture(async f=>{
 await rm(join(f.repo,'src/input.txt'));await writeFile(join(f.dir,'victim'),'valuable');await symlink(join(f.dir,'victim'),join(f.repo,'src/input.txt'));
 await assert.rejects(f.delivery.deliver({input:f.input,receipt:f.receipt}),{code:'DELIVERY_SYMLINK'});
 assert.equal(await readFile(join(f.dir,'victim'),'utf8'),'valuable');
}));
test('crashed intent is never overwritten or auto-replayed',()=>fixture(async f=>{
 const evidence=join(f.actionRoot,'verified-delivery');await mkdir(evidence);await writeFile(join(evidence,'intent.json'),'{}');
 await assert.rejects(f.delivery.deliver({input:f.input,receipt:f.receipt}),{code:'DELIVERY_ALREADY_ATTEMPTED'});
 assert.deepEqual(await readFile(join(f.repo,'src/input.txt')),before);
}));
test('after independently confirmed Rust COMMITTED, delivery conflict stays VERIFIED_NOT_DELIVERED without touching user file',()=>fixture(async f=>{
 let calls=0;
 const modelHost={async createSession({tools}){return {async prompt(){
  const tool=tools.find(t=>t.name==='voder_authorized_action');
  await tool.execute({kind:'patch',relative_path:'src/input.txt',old_sha256:hash(before),new_content:after});
  return {settled:true};},dispose(){}}}};
 const sdk=new VoderSDK({cwd:f.repo,nativeGateway:{binary:'/tmp/voder-gateway',daemonBinary:'/tmp/voderd',policyFile:f.policyFile,
  attemptStore:f.store,nodeBinary:'/tmp/node',modalCli:'/tmp/cli.ts',targetPath:'src/input.txt'},modelHost});
 sdk.native.executeAuthorized=async input=>{calls++;const c=join(f.store,'attempts',input.action_id,'owned-worktrees','candidate');
  await mkdir(join(c,'src'),{recursive:true});await writeFile(join(c,'src/input.txt'),after);
  return {...f.receipt,action_id:input.action_id,run_id:`run-${input.action_id}`,session_id:input.session_id,workspace_path:c};};
 sdk.commitVerifier.confirmCommit=async (r,input)=>({verified:true,authority:'VODER_NATIVE_VGR_STATE',session_id:input.session_id,
  action_id:input.action_id,run_id:r.run_id,commit_digest:r.commit_digest});
 const {sessionId}=sdk.createSession({id:'session-test'});
 await writeFile(join(f.repo,'src/input.txt'),'user-edit\n');
 const result=await sdk.submitTask(sessionId,{goal:'Replace src/input.txt'});
 assert.equal(result.status,'VERIFIED_NOT_DELIVERED');assert.equal(result.delivery.code,'DELIVERY_CONFLICT');
 assert.equal(calls,1);assert.equal(await readFile(join(f.repo,'src/input.txt'),'utf8'),'user-edit\n');
 sdk.dispose();
}));
test('original target mode is preserved even under Voder CLI restrictive umask',()=>fixture(async f=>{
 const target=join(f.repo,'src/input.txt');await chmod(target,0o640);
 const old=process.umask(0o077);
 try{assert.equal((await f.delivery.deliver({input:f.input,receipt:f.receipt})).status,'DELIVERED')}
 finally{process.umask(old)}
 assert.equal((await stat(target)).mode & 0o777,0o640);
}));
test('stale cross-process lock refuses a new delivery attempt',()=>fixture(async f=>{
 const locks=join(f.store,'verified-delivery-locks');await mkdir(locks,{mode:0o700});
 // use a real first delivery to compute the lock digest from cwd + path
 const id=hash(f.repo+'\0'+'src/input.txt');await mkdir(join(locks,id));
 await assert.rejects(f.delivery.deliver({input:f.input,receipt:f.receipt}),{code:'DELIVERY_LOCKED'});
 assert.deepEqual(await readFile(join(f.repo,'src/input.txt')),before);
}));
test('broker binds independently verified Rust commit to host-only delivery and kernel reports DELIVERED',()=>fixture(async f=>{
 let writes=0;const events=[];
 const modelHost={async createSession({tools}){return {async prompt(){
  const tool=tools.find(t=>t.name==='voder_authorized_action');
  const got=await tool.execute({kind:'patch',relative_path:'src/input.txt',old_sha256:hash(before),new_content:after});
  assert.equal(got.status,'COMMITTED');return {settled:true};
 },dispose(){}}}};
 const sdk=new VoderSDK({cwd:f.repo,nativeGateway:{binary:'/tmp/voder-gateway',daemonBinary:'/tmp/voderd',policyFile:f.policyFile,
  attemptStore:f.store,nodeBinary:'/tmp/node',modalCli:'/tmp/cli.ts',targetPath:'src/input.txt'},modelHost,onEvent:e=>events.push(e)});
 // Test double only at the native boundary: this verifies SDK delivery integration, NOT genuine Rust execution.
 sdk.native.executeAuthorized=async input=>{writes++;f.receipt.action_id=input.action_id;
  f.receipt.run_id=`run-${input.action_id}`;f.receipt.session_id=input.session_id;
  const root=join(f.store,'attempts',input.action_id,'owned-worktrees','candidate');
  await mkdir(join(root,'src'),{recursive:true});await writeFile(join(root,'src/input.txt'),after);
  f.receipt.workspace_path=root;return {...f.receipt}};
 sdk.commitVerifier.confirmCommit=async (r,input)=>({verified:true,authority:'VODER_NATIVE_VGR_STATE',session_id:input.session_id,
  action_id:input.action_id,run_id:r.run_id,commit_digest:r.commit_digest});
 const {sessionId}=sdk.createSession({id:'session-test'});
 const result=await sdk.submitTask(sessionId,{goal:'Replace src/input.txt'});
 assert.equal(result.status,'DELIVERED');assert.equal(result.delivery.status,'DELIVERED');assert.equal(sdk.getStatus(sessionId),'DELIVERED');
 assert.equal(writes,1);assert.equal(await readFile(join(f.repo,'src/input.txt'),'utf8'),after);
 sdk.dispose();
}));
