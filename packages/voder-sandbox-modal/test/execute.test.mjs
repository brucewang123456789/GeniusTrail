import {test} from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,writeFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {createHash} from 'node:crypto';
import {executeModalTask} from '../src/execute.ts';
const requireHash=x=>createHash('sha256').update(x).digest('hex');

const env={VODER_MODAL_APP:'voder-sandbox',VODER_MODAL_IMAGE:'node@sha256:'+'a'.repeat(64),MODAL_TOKEN_ID:'test-id',MODAL_TOKEN_SECRET:'test-secret'};
let work;
function pipe(content){const stream=new ReadableStream({start(controller){controller.enqueue(new TextEncoder().encode(content));controller.close();}});return stream;}
function proc(code,out='',err=''){return {stdout:pipe(out),stderr:pipe(err),async wait(){return code}};}
function fake({status=0,badHash=false,crash=false,cleanupFails=false,stdout='HELLO\n',stderr=''}={}){
  const state={created:0,commands:[],uploads:new Map(),terminated:0,opts:null};
  const sandbox={sandboxId:'sb-test-1',filesystem:{
    async writeBytes(bytes,path){state.uploads.set(path,bytes)},
    async writeText(text,path){state.uploads.set(path,text)},
  },async exec(cmd,opts){state.commands.push([cmd,opts]);if(crash)throw Error('remote failure');
    if(cmd[0]==='/bin/mkdir')return proc(0);
    if(cmd[0]==='/usr/bin/sha256sum')return proc(badHash?1:0,badHash?'wrong sha':'sha256 ok');
    return proc(status,stdout,stderr);
  },async terminate(){state.terminated++;if(cleanupFails)throw Error('terminate failed')}};
  const driver={async create(app,img,opts){state.created++;state.opts=opts;return sandbox}};
  return {driver,state};
}
const task=()=>({workspace:work,program:'/usr/bin/cat',args:['hello.txt'],runId:'r-1',actionId:'a-1',requiredFiles:['hello.txt'],timeoutMs:10000,memoryMiB:512,cpu:1,maxOutputBytes:4096});

test.beforeEach(async()=>{work=await mkdtemp(join(tmpdir(),'voder-mod-'));await writeFile(join(work,'hello.txt'),'HELLO\n');});
test.afterEach(async()=>{await rm(work,{recursive:true,force:true});});
test('verified upload => remote execution => authentic observed receipt and termination',async()=>{
  const {driver,state}=fake();const receipt=await executeModalTask(task(),driver,env);
  assert.equal(receipt.exitCode,0);assert.equal(receipt.stdout,'HELLO\n');
  assert.equal(receipt.files,1);assert.equal(receipt.status,'EXECUTED');
  assert.equal(state.terminated,1);assert.equal(state.created,1);
  assert.deepEqual(state.commands.map(c=>c[0][0]),['/bin/mkdir','/usr/bin/sha256sum','/usr/bin/cat']);
  assert.equal(state.opts.blockNetwork,true);assert.ok(state.uploads.has('/workspace/hello.txt'));
});
test('missing file detected before billable sandbox creation',async()=>{
  const {driver,state}=fake();await assert.rejects(executeModalTask({...task(),requiredFiles:['not-found']},driver,env),/MISSING_REQUIRED_FILE/);
  assert.equal(state.created,0);
});
test('integrity mismatch aborts before executing user command and terminates',async()=>{
  const {driver,state}=fake({badHash:true});await assert.rejects(executeModalTask(task(),driver,env),/REMOTE_INPUT_INTEGRITY_FAILED/);
  assert.equal(state.commands.length,2);assert.equal(state.terminated,1);
});
test('remote nonzero exit is evidence, not execution exception or success',async()=>{
  const {driver,state}=fake({status:13,stderr:'real error'});
  const receipt=await executeModalTask(task(),driver,env);
  assert.equal(receipt.exitCode,13);assert.equal(receipt.stderr,'real error');assert.equal(state.terminated,1);
});
test('remote crash releases Modal sandbox',async()=>{
  const {driver,state}=fake({crash:true});await assert.rejects(executeModalTask(task(),driver,env),/remote failure/);assert.equal(state.terminated,1);
});
test('cleanup failure blocks delivery',async()=>{
  const {driver}=fake({cleanupFails:true});await assert.rejects(executeModalTask(task(),driver,env),/MODAL_TERMINATION_FAILED/);
});
test('output limit is recorded as failure obligation',async()=>{
  const {driver}=fake({stdout:'X'.repeat(10000)});
  const receipt=await executeModalTask(task(),driver,env);assert.equal(receipt.outputLimited,true);
});
test('combined output budget bounds stdout+stderr and marks truncation',async()=>{
  const {driver}=fake({stdout:'A'.repeat(3000),stderr:'B'.repeat(3000)});
  const receipt=await executeModalTask(task(),driver,env);
  assert.equal(receipt.outputLimited,true);
  assert.ok(Buffer.byteLength(receipt.stdout)+Buffer.byteLength(receipt.stderr)<=task().maxOutputBytes);
});
test('invalid UTF-8 remote output blocks instead of fabricating text evidence',async()=>{
  const {driver,state}=fake();
  const sb=await driver.create();
  const oldExec=sb.exec;
  sb.exec=async(cmd,opts)=>{
    if(cmd[0]==='/usr/bin/cat')return {
      stdout:new ReadableStream({start(c){c.enqueue(new Uint8Array([0xff,0xfe]));c.close()}}),
      stderr:pipe(''),async wait(){return 0},
    };
    return oldExec(cmd,opts);
  };
  driver.create=async()=>sb;
  await assert.rejects(executeModalTask(task(),driver,env),/encoded data was not valid|valid for encoding|not valid for encoding/i);
  assert.equal(state.terminated,1);
});
test('UTF8 byte-order mark survives as evidence, with exact digest',async()=>{
  const {driver,state}=fake({stdout:'\uFEFFOK'});
  const receipt=await executeModalTask(task(),driver,env);
  assert.equal(receipt.stdout,'\uFEFFOK');
  assert.equal(receipt.stdoutSha256,requireHash('\uFEFFOK'));
  assert.equal(state.terminated,1);
});
test('output limit slicing inside a UTF8 codepoint produces bounded truncation evidence',async()=>{
  const {driver}=fake({stdout:'€'.repeat(2000)});
  const receipt=await executeModalTask(task(),driver,env);
  assert.equal(receipt.outputLimited,true);
  assert.ok(Buffer.byteLength(receipt.stdout)<task().maxOutputBytes);
  assert.ok(receipt.stdout.startsWith('€'));
});
test('mismatched sandbox identity and upload failure never become verified result',async()=>{
  const {driver,state}=fake();driver.create=async()=>({
    sandboxId:'sb-1',filesystem:{async writeBytes(){throw Error('UPLOAD_FAILED')},async writeText(){}},
    async exec(cmd){if(cmd[0]==='/bin/mkdir')return proc(0);throw Error('EXEC_MUST_NOT_RUN')},async terminate(){state.terminated++;}
  });
  await assert.rejects(executeModalTask(task(),driver,env),/UPLOAD_FAILED/);
  assert.equal(state.terminated,1);
});
