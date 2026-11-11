import {test} from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp, writeFile, rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {executeModalTask} from '../src/execute.ts';
const env={VODER_MODAL_APP:'voder-sandbox',VODER_MODAL_IMAGE:'image@sha256:'+'b'.repeat(64),MODAL_TOKEN_ID:'test-id',MODAL_TOKEN_SECRET:'test-secret'};
const stream=()=>new ReadableStream({start(c){c.close()}});
const proc=code=>({stdout:stream(),stderr:stream(),async wait(){return code}});
async function fixture(fn){
 const workspace=await mkdtemp(join(tmpdir(),'voder-lifecycle-'));
 try{await writeFile(join(workspace,'a.txt'),'123\n');await fn({workspace,program:'/usr/bin/cat',args:['a.txt'],timeoutMs:120000,runId:'r1',actionId:'a1',memoryMiB:512,cpu:1,maxOutputBytes:4096});}
 finally{await rm(workspace,{recursive:true,force:true});}
}
const setup=(fault=()=>undefined)=>{
 const st={opts:null,terminated:0,steps:[]};
 const sb={sandboxId:'sb-fixture',filesystem:{async writeBytes(){st.steps.push('upload')},async writeText(){st.steps.push('checksums')}},
  async exec(args){st.steps.push(args[0]);const e=fault(args);if(e)throw e;return proc(args[0]==='/usr/bin/cat'?0:0)},
  async terminate(){st.terminated++}};
 return {st,driver:{async create(_app,_image,opts){st.opts=opts;return sb;}}};
};
test('120-second verifier reserves 12-minute preparation plus cleanup, not old 210-second lease',()=>fixture(async task=>{
 const {st,driver}=setup();const r=await executeModalTask(task,driver,env);
 assert.equal(st.opts.timeoutMs,120000+720000+90000);
 assert.equal(st.opts.idleTimeoutMs,st.opts.timeoutMs);
 assert.ok(st.opts.timeoutMs>210000);
 assert.equal(r.status,'EXECUTED');assert.equal(r.exitCode,0);assert.equal(st.terminated,1);
 assert.deepEqual(st.steps,['/bin/mkdir','upload','checksums','/usr/bin/sha256sum','/usr/bin/cat']);
}));
test('long command lifetime stays larger than command budget, with explicit bounded preparation override',()=>fixture(async task=>{
 const {st,driver}=setup();const r=await executeModalTask({...task,timeoutMs:3600000,setupBudgetMs:1800000},driver,env);
 assert.equal(st.opts.timeoutMs,3600000+1800000+90000);
 assert.equal(st.opts.idleTimeoutMs,st.opts.timeoutMs);
 assert.equal(r.status,'EXECUTED');assert.equal(st.terminated,1);
}));
test('unsafe setup budget is rejected before a paid sandbox exists',()=>fixture(async task=>{
 const {st,driver}=setup();await assert.rejects(executeModalTask({...task,setupBudgetMs:1},driver,env),/INVALID_SANDBOX_SETUP_BUDGET/);
 assert.equal(st.opts,null);
}));
test('sandbox shutdown during checksum is infrastructure failure, never a passing candidate receipt',()=>fixture(async task=>{
 const error=new Error('/modal.task_command_router.TaskCommandRouter/TaskExecStart FAILED_PRECONDITION: Modal Sandbox is shutting down.');
 const {st,driver}=setup(args=>args[0]==='/usr/bin/sha256sum'?error:undefined);
 await assert.rejects(executeModalTask(task,driver,env),e=>{
  assert.match(e.message,/MODAL_PHASE=REMOTE_INPUT_VERIFY/);
  assert.match(e.message,/MODAL_FAILURE_CLASS=SANDBOX_SHUTTING_DOWN/);
  assert.match(e.message,/SANDBOX_MAX_MS=930000/);
  return true;
 });
 assert.equal(st.terminated,1);assert.ok(!st.steps.includes('/usr/bin/cat'));
}));
test('legitimate failing model candidate returns executed nonzero receipt, not an infrastructure error',()=>fixture(async task=>{
 const {st,driver}=setup();driver.create=async()=>({sandboxId:'sb-test',filesystem:{async writeBytes(){},async writeText(){}},
  async exec(argv){return proc(argv[0]==='/usr/bin/cat'?1:0)},async terminate(){st.terminated++}});
 const r=await executeModalTask(task,driver,env);
 assert.equal(r.exitCode,1);assert.equal(r.status,'EXECUTED');assert.equal(st.terminated,1);
}));
