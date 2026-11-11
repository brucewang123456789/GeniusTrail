import { test } from 'node:test';
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { PassThrough } from 'node:stream';
import { mkdtempSync, mkdirSync, readFileSync, statSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { TrustedVgrGatewayTransport, GatewayCommitVerifier } from '../src/trusted-gateway.mjs';
const SHA='a'.repeat(64);
const options={binary:'/tmp/voder-trusted-gateway',daemonBinary:'/tmp/voderd',
  policyFile:'/tmp/voder-policy.json',attemptStore:'/tmp/voder-attempts',
  nodeBinary:'/tmp/node',modalCli:'/tmp/modal-driver.js',targetPath:'src/file.txt'};
const input={schema:'voder.agent.action.v1',session_id:'session-test',action_id:'action-test',
  idempotency_key:'unique-test',kind:'patch',relative_path:'src/file.txt',old_sha256:SHA,new_content:'good'};
function fakeSpawn(raw,code=0){
  const calls=[];
  const spawnImpl=(bin,args,opts)=>{
    const call={bin,args,opts};calls.push(call);
    const child=new EventEmitter();child.stdin={end(text){call.stdinPayload=text;queueMicrotask(()=>{
      if(raw!==null)child.stdout.emit('data',Buffer.from(JSON.stringify(raw)));
      child.emit('close',code,null);
    });}};
    child.stdout=new PassThrough();child.stderr=new PassThrough();child.kill=()=>{};
    return child;
  };
  return {calls,spawnImpl};
}
test('malformed model actions and traversal are rejected before spawning native',async()=>{
  let spawned=0;
  const t=new TrustedVgrGatewayTransport({...options,spawnImpl:()=>{spawned++;throw new Error('unexpected')}});
  for(const bad of [{...input,relative_path:'../escape'}, {...input,relative_path:'src/other.txt'},
    {...input,kind:'command'},{...input,old_sha256:'bad'}, {...input,session_id:'a/b'},
    {...input,new_content:''}]){
    await assert.rejects(t.executeAuthorized(bad),{name:'GatewayError'});
  }
  assert.equal(spawned,0);
});
test('valid native call uses shell:false, pinned policy and no model-selected command',async()=>{
  const receipt={authority:'VODER_NATIVE_VGR',status:'COMMITTED',session_id:input.session_id,
    action_id:input.action_id,run_id:`run-${input.action_id}`,journal_digest:SHA,commit_digest:SHA};
  const {calls,spawnImpl}=fakeSpawn(receipt);
  const t=new TrustedVgrGatewayTransport({...options,spawnImpl});
  assert.equal((await t.executeAuthorized(input)).status,'COMMITTED');
  assert.equal(calls.length,1);assert.equal(calls[0].opts.shell,false);
  assert.deepEqual(calls[0].args,['--policy',options.policyFile,'--attempt-root','/tmp/voder-attempts/attempts/action-test']);
  const sent=JSON.parse(calls[0].stdinPayload);
  assert.equal(sent.kind,'patch');
  assert.equal(Object.hasOwn(sent,'program'),false);
  assert.equal(Object.hasOwn(sent,'validator_args'),false);
  await assert.rejects(t.executeAuthorized(input),{code:'DUPLICATE'});
});
test('fabricated native commit with wrong run ID is rejected, and cannot replay',async()=>{
  const {spawnImpl}=fakeSpawn({authority:'VODER_NATIVE_VGR',status:'COMMITTED',session_id:input.session_id,
    action_id:input.action_id,run_id:'wrong',journal_digest:SHA,commit_digest:SHA});
  const t=new TrustedVgrGatewayTransport({...options,spawnImpl});
  await assert.rejects(t.executeAuthorized(input),{code:'BAD_RECEIPT'});
  await assert.rejects(t.executeAuthorized(input),{code:'DUPLICATE'});
});
test('aborted proposal cannot start an external effect',async()=>{
  let called=0; const t=new TrustedVgrGatewayTransport({...options,spawnImpl:()=>{called++;throw Error('unexpected')}});
  const ctl=new AbortController();ctl.abort();
  await assert.rejects(t.executeAuthorized(input,{signal:ctl.signal}),{code:'CANCELLED'});
  assert.equal(called,0);
});
test('native crash is always a failure, never a fabricated receipt',async()=>{
  const {spawnImpl}=fakeSpawn(null,1);
  await assert.rejects(new TrustedVgrGatewayTransport({...options,spawnImpl}).executeAuthorized(input),{code:'NATIVE_REJECTED'});
});
test('commit verifier derives trusted state directory and ignores model-supplied state',async()=>{
  const verifier=new GatewayCommitVerifier({...options});
  await assert.rejects(verifier.confirmCommit({action_id:'other',run_id:'run-other'},input),{code:'IDENTITY'});
});

test('native failure stores bounded stderr in private evidence without leaking through RPC',async()=>{
  const root=mkdtempSync(join(tmpdir(),'voder-g1-diagnostic-'));
  try{
    const dir=join(root,'attempts','action-test');
    mkdirSync(dir,{recursive:true,mode:0o700});
    const message='VODER_TRUSTED_GATEWAY=FAIL: UnsafePath(private-diagnostic)';
    const spawnImpl=()=>{
      const child=new EventEmitter();child.stdout=new PassThrough();child.stderr=new PassThrough();
      child.kill=()=>{};
      child.stdin={end(){queueMicrotask(()=>{
        child.stderr.emit('data',Buffer.from(message));
        child.emit('close',1,null);
      })}};
      return child;
    };
    const t=new TrustedVgrGatewayTransport({...options,attemptStore:root,spawnImpl});
    await assert.rejects(t.executeAuthorized(input),e=>{
      assert.equal(e.code,'NATIVE_REJECTED');
      assert.equal(e.message.includes('UnsafePath'),false);
      assert.equal(e.message.includes('private_stderr_saved=true'),true);
      return true;
    });
    const diagnostic=join(dir,'gateway.stderr');
    assert.equal(readFileSync(diagnostic,'utf8'),message);
    assert.equal(statSync(diagnostic).mode & 0o777,0o600);
    await assert.rejects(t.executeAuthorized(input),{code:'DUPLICATE'});
  }finally{rmSync(root,{recursive:true,force:true})}
});
