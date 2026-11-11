import {test} from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,rm} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {spawnSync} from 'node:child_process';
import {inputArchive} from '../src/input-archive.ts';
import {snapshotWorkspace} from '../src/manifest.ts';
import {executeModalTask} from '../src/execute.ts';
const env={VODER_MODAL_APP:'voder-test',VODER_MODAL_IMAGE:'python@sha256:'+'a'.repeat(64),MODAL_TOKEN_ID:'fixture',MODAL_TOKEN_SECRET:'fixture'};
const proc=(rc=0,out='',err='')=>({stdout:new ReadableStream({start(c){c.enqueue(Buffer.from(out));c.close()}}),stderr:new ReadableStream({start(c){c.enqueue(Buffer.from(err));c.close()}}),async wait(){return rc}});
async function fixture(options={}){
 const top=await mkdtemp(join(tmpdir(),'voder-input-archive-'));
 const workspace=join(top,'source');await mkdir(workspace);
 for(let i=0;i<40;i++)await writeFile(join(workspace,`file-${i}.txt`),`body-${i}\r\n`);
 const long='深层目录'.repeat(12)+'/'+ 'nested'.repeat(12)+'/'+ 'filename'.repeat(12)+'.txt';
 await mkdir(join(workspace,long,'..'),{recursive:true});await writeFile(join(workspace,long),'\uFEFF中文\r\n');
 const state={uploads:0,creates:0,terminated:0,tasks:0,extracts:0,archiveChecks:0};
 const driver={async create(){
  const index=++state.creates;const remote=join(top,'remote-'+index);await mkdir(join(remote,'tmp'),{recursive:true});
  const translate=x=>x.startsWith('/workspace')||x==='/tmp'||x.startsWith('/tmp/')?join(remote,x):x;
  return {sandboxId:'sb-'+index,filesystem:{async writeBytes(bytes,path){state.uploads++;const data=Buffer.from(bytes);if(options.corrupt)data[20]^=1;await writeFile(translate(path),data)},async writeText(text,path){await writeFile(translate(path),text)}},
   async exec(argv,opts){
    if(argv[0]==='/bin/tar'){state.extracts++;if(options.ambiguous)throw Error('/modal.task_command_router.TaskCommandRouter/TaskExecStart UNAVAILABLE: no connection established');if(options.badExtract)return proc(2,'','tar unavailable');}
    if(argv[0]==='/usr/bin/cat')state.tasks++;
    if(argv.includes('voder-archive.sha256'))state.archiveChecks++;
    const args=argv.slice(1).map(value=>argv[0]==='/bin/sh'&&value.includes('/tmp/voder-input.part-')?value.replaceAll('/tmp/voder-input',join(remote,'tmp','voder-input')):translate(value));const result=spawnSync(argv[0],args,{cwd:opts?.workdir?translate(opts.workdir):remote,encoding:'utf8'});
    if(result.error)throw result.error;
    return proc(result.status,result.stdout,result.stderr);
   },async terminate(){state.terminated++}};
 }};
 const task={workspace,program:'/usr/bin/cat',args:[long],runId:'r-archive',actionId:'a-archive',timeoutMs:10000,memoryMiB:512,cpu:1,maxOutputBytes:4096};
 return {top,workspace,state,driver,task,cleanup:()=>rm(top,{recursive:true,force:true})};
}
test('real tar/gzip and checksum roundtrip: long Unicode names, BOM/CRLF, one bulk upload',async()=>{
 const f=await fixture();try{
  const r=await executeModalTask(f.task,f.driver,env);
  assert.equal(r.stdout,'\uFEFF中文\r\n');assert.equal(r.files,41);assert.equal(r.exitCode,0);
  assert.equal(f.state.uploads,1);assert.equal(f.state.archiveChecks,1);assert.equal(f.state.extracts,1);assert.equal(f.state.terminated,1);
 }finally{await f.cleanup();}
});
test('corrupt transfer fails before archive extraction and task execution',async()=>{
 const f=await fixture({corrupt:true});try{
  await assert.rejects(executeModalTask(f.task,f.driver,env),/REMOTE_ARCHIVE_INTEGRITY_FAILED/);
  assert.equal(f.state.extracts,0);assert.equal(f.state.tasks,0);assert.equal(f.state.terminated,1);
 }finally{await f.cleanup();}
});
test('archive extraction failure cannot produce candidate success',async()=>{
 const f=await fixture({badExtract:true});try{
  await assert.rejects(executeModalTask(f.task,f.driver,env),/REMOTE_ARCHIVE_EXTRACTION_FAILED/);
  assert.equal(f.state.tasks,0);assert.equal(f.state.terminated,1);
 }finally{await f.cleanup();}
});
test('uncertain extraction RPC is never replayed',async()=>{
 const f=await fixture({ambiguous:true});try{
  await assert.rejects(executeModalTask(f.task,f.driver,env),/MODAL_RPC_UNCERTAIN_NO_REPLAY/);
  assert.equal(f.state.extracts,1);assert.equal(f.state.tasks,0);
 }finally{await f.cleanup();}
});
test('artifact cache revalidates bytes and returns defensive copies',async()=>{
 const f=await fixture();try{
  const snapshot=await snapshotWorkspace(f.workspace);
 const first=await inputArchive(snapshot);
 try{
  const cached=await inputArchive(snapshot);
  try{
   assert.equal(cached.cacheHit,true);assert.equal(cached.sha256,first.sha256);
   const original=await readFile(cached.path);const edited=Buffer.from(original);edited[0]^=1;
   assert.deepEqual(await readFile(first.path),original);
   await writeFile(snapshot.files[0].source,edited);
   await assert.rejects(inputArchive(snapshot),/WORKSPACE_CHANGED_DURING_SNAPSHOT/);
  }finally{await cached.dispose();}
 }finally{await first.dispose();}
 }finally{await f.cleanup();}
});
test('source changes invalidate cached transport and never reuse validator verdicts',async()=>{
 const f=await fixture();try{
  const first=await inputArchive(await snapshotWorkspace(f.workspace));
  await first.dispose();
  await writeFile(join(f.workspace,'file-0.txt'),'different');const next=await inputArchive(await snapshotWorkspace(f.workspace));
  try{assert.notEqual(first.sha256,next.sha256);}finally{await next.dispose();}
  await executeModalTask(f.task,f.driver,env);await executeModalTask({...f.task,actionId:'a-next'},f.driver,env);
  assert.equal(f.state.creates,2);assert.equal(f.state.tasks,2);assert.equal(f.state.terminated,2);
 }finally{await f.cleanup();}
});
