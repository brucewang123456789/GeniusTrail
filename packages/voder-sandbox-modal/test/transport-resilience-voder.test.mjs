import {test} from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp, writeFile, rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {executeModalTask} from '../src/execute.ts';

const timeout='/modal.task_command_router.TaskCommandRouter/TaskExecStart UNAVAILABLE: No connection established. Last error: Error: Connection attempt timed out after 20000ms. Resolution note:';
const env={VODER_MODAL_APP:'voder-test',VODER_MODAL_IMAGE:'docker.io/library/debian@sha256:'+'a'.repeat(64),MODAL_TOKEN_ID:'test-only',MODAL_TOKEN_SECRET:'test-only'};
const proc=(code=0,out='')=>({stdout:new ReadableStream({start(c){if(out)c.enqueue(new TextEncoder().encode(out));c.close()}}),stderr:new ReadableStream({start(c){c.close()}}),async wait(){return code}});
let dir;
test.beforeEach(async()=>{dir=await mkdtemp(join(tmpdir(),'voder-modal-fixture-'));await writeFile(join(dir,'src-plain.txt'),'VODER_TEST\n');});
test.afterEach(async()=>{await rm(dir,{recursive:true,force:true})});
function fixture(program='/usr/bin/grep',args=['-Fx','VODER_TEST','src-plain.txt'],fault=()=>null,cleanupFails=false){
 const state={attempts:[],terminated:0,created:0,uploads:0};
 const sandbox={sandboxId:'sb-immutable-mock', filesystem:{async writeBytes(){state.uploads++},async writeText(){state.uploads++}},
  async exec(argv){state.attempts.push(argv);const err=fault(argv,state.attempts);if(err)throw Error(err);return proc(0,argv[0]===program?'VODER_TEST\n':'');},
  async terminate(){state.terminated++;if(cleanupFails)throw Error('MOCK_TERMINATE_FAILURE')}};
 const driver={async create(_app,_img,options){assert.equal(options.blockNetwork,true);state.created++;return sandbox}};
 const task={workspace:dir,program,args,runId:'run1',actionId:'action1',requiredFiles:['src-plain.txt'],timeoutMs:120000,memoryMiB:512,cpu:1,maxOutputBytes:8192};
 return {state,driver,task};
}
const hits=(s,program)=>s.attempts.filter(a=>a[0]===program).length;

test('real fault signature: readonly grep retries once, returns verified receipt, single sandbox',async()=>{
 let failed=false;
 const {state,driver,task}=fixture('/usr/bin/grep',['-Fx','VODER_TEST','src-plain.txt'],argv=>{
  if(argv[0]==='/usr/bin/grep'&&!failed){failed=true;return timeout}return null;
 });
 const result=await executeModalTask(task,driver,env);
 assert.equal(result.status,'EXECUTED');assert.equal(result.exitCode,0);assert.equal(hits(state,'/usr/bin/grep'),2);
 assert.equal(state.created,1);assert.equal(state.terminated,1);assert.ok(result.stdoutSha256);
});
test('persistent remote timeout exhausts bounded read-only attempts and fails closed',async()=>{
 const {state,driver,task}=fixture('/usr/bin/grep',['-Fx','VODER_TEST','src-plain.txt'],argv=>argv[0]==='/usr/bin/grep'?timeout:null);
 await assert.rejects(executeModalTask(task,driver,env),e=>{
  assert.match(e.message,/MODAL_PHASE=REMOTE_TASK_EXEC/);assert.match(e.message,/MODAL_READONLY_RETRY_EXHAUSTED/);
  assert.match(e.message,/ATTEMPTS=3/);assert.match(e.message,/MODAL_OUTCOME=NOT_PROVEN_STOPPED/);return true;
 });
 assert.equal(hits(state,'/usr/bin/grep'),3);assert.equal(state.terminated,1);
});
test('arbitrary state-changing command is never replayed after unknown delivery',async()=>{
 const {state,driver,task}=fixture('/usr/bin/tee',['src-plain.txt'],argv=>argv[0]==='/usr/bin/tee'?timeout:null);
 await assert.rejects(executeModalTask(task,driver,env),/MODAL_RPC_UNCERTAIN_NO_REPLAY/);
 assert.equal(hits(state,'/usr/bin/tee'),1);assert.equal(state.terminated,1);
});
test('grep on file not present in immutable input manifest is never replayed',async()=>{
 const {state,driver,task}=fixture('/usr/bin/grep',['-Fx','VODER_TEST','/tmp/not-immutable'],argv=>argv[0]==='/usr/bin/grep'?timeout:null);
 await assert.rejects(executeModalTask(task,driver,env),/MODAL_RPC_UNCERTAIN_NO_REPLAY/);
 assert.equal(hits(state,'/usr/bin/grep'),1);
});
test('grep with unapproved flags is not certified readonly',async()=>{
 const {state,driver,task}=fixture('/usr/bin/grep',['-r','VODER_TEST','src-plain.txt'],argv=>argv[0]==='/usr/bin/grep'?timeout:null);
 await assert.rejects(executeModalTask(task,driver,env),/MODAL_RPC_UNCERTAIN_NO_REPLAY/);
 assert.equal(hits(state,'/usr/bin/grep'),1);
});
test('pre-task checksum verifier retries safe readonly command',async()=>{
 let failed=false;
 const {state,driver,task}=fixture('/usr/bin/grep',['-Fx','VODER_TEST','src-plain.txt'],argv=>{
  if(argv[0]==='/usr/bin/sha256sum'&&!failed){failed=true;return timeout}return null;
 });
 const r=await executeModalTask(task,driver,env);
 assert.equal(r.exitCode,0);assert.equal(hits(state,'/usr/bin/sha256sum'),2);
 assert.equal(hits(state,'/usr/bin/grep'),1);
});
test('sandbox-local mkdir -p is idempotent and recoverable',async()=>{
 let failed=false;
 const {state,driver,task}=fixture('/usr/bin/grep',['-Fx','VODER_TEST','src-plain.txt'],argv=>{
  if(argv[0]==='/bin/mkdir'&&!failed){failed=true;return timeout}return null;
 });
 const r=await executeModalTask(task,driver,env);
 assert.equal(r.exitCode,0);assert.equal(hits(state,'/bin/mkdir'),2);
});
test('unrelated SDK error is never replayed even for readonly command',async()=>{
 const {state,driver,task}=fixture('/usr/bin/grep',['-Fx','VODER_TEST','src-plain.txt'],argv=>argv[0]==='/usr/bin/grep'?'REMOTE_SIGNATURE_MISMATCH':null);
 await assert.rejects(executeModalTask(task,driver,env),/REMOTE_SIGNATURE_MISMATCH/);
 assert.equal(hits(state,'/usr/bin/grep'),1);
});
test('cleanup failure does not erase primary transport error',async()=>{
 const {state,driver,task}=fixture('/usr/bin/tee',['src-plain.txt'],argv=>argv[0]==='/usr/bin/tee'?timeout:null,true);
 await assert.rejects(executeModalTask(task,driver,env),e=>{
  assert.match(e.message,/MODAL_RUN_AND_CLEANUP_FAILED/);assert.match(e.message,/MODAL_RPC_UNCERTAIN_NO_REPLAY/);
  assert.match(e.message,/MOCK_TERMINATE_FAILURE/);return true;
 });
 assert.equal(hits(state,'/usr/bin/tee'),1);assert.equal(state.terminated,1);
});
test('remote process wait uncertainty is never retried',async()=>{
 const {state,driver,task}=fixture('/usr/bin/grep',['-Fx','VODER_TEST','src-plain.txt']);
 const old=driver.create;driver.create=async(...params)=>{
  const sandbox=await old(...params);const orig=sandbox.exec;
  sandbox.exec=async(argv,opts)=>argv[0]==='/usr/bin/grep'?{
   stdout:new ReadableStream({start(c){c.close()}}),stderr:new ReadableStream({start(c){c.close()}}),async wait(){throw Error(timeout)}
  }:orig(argv,opts);
  return sandbox;
 };
 await assert.rejects(executeModalTask(task,driver,env),/TaskExecStart UNAVAILABLE/);
 assert.equal(hits(state,'/usr/bin/grep'),0);assert.equal(state.terminated,1);
});
