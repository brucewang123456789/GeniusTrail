import {test} from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,writeFile,symlink,rm,mkdir} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {snapshotWorkspace} from '../src/manifest.ts';
import {validateTask,validateConfig,assertRelative} from '../src/contract.ts';

const task={workspace:'/tmp/work',program:'/usr/bin/cat',args:['hello.txt'],runId:'r1',actionId:'a1',timeoutMs:1000,memoryMiB:256,cpu:1,maxOutputBytes:8192};
test('strong policy rejects malformed actions and filesystem traversal',()=>{
  validateTask(task);
  for(const p of ['../pass','./pass','/etc/passwd','dir//file','foo\\bar','a\n'])assert.throws(()=>assertRelative(p));
  assert.throws(()=>validateTask({...task,program:'bash'}),/ABSOLUTE_PROGRAM/);
  assert.throws(()=>validateTask({...task,memoryMiB:1}),/INVALID_MEMORY/);
  assert.throws(()=>validateTask({...task,timeoutMs:0}),/INVALID_TIMEOUT/);
});
test('missing credential or unpinned image fails closed',()=>{
  assert.throws(()=>validateConfig({VODER_MODAL_APP:'voder-modal',VODER_MODAL_IMAGE:'node:22',MODAL_TOKEN_ID:'id',MODAL_TOKEN_SECRET:'secret'}),/DIGEST/);
  assert.throws(()=>validateConfig({VODER_MODAL_APP:'voder-modal',VODER_MODAL_IMAGE:'node@sha256:'+'f'.repeat(64)}),/CREDENTIALS/);
});
test('snapshot includes required sources but never sends standard secrets',async()=>{
  const d=await mkdtemp(join(tmpdir(),'voder-src-'));
  try{
    await mkdir(join(d,'.git'));await writeFile(join(d,'.git','config'),'secret');
    await writeFile(join(d,'.env'),'token');
    await writeFile(join(d,'hello.txt'),'hello');
    const s=await snapshotWorkspace(d,['hello.txt']);
    assert.equal(s.files.length,1);assert.equal(s.files[0].relative,'hello.txt');
    assert.equal(s.totalBytes,5);assert.equal(s.digest.length,64);
    await assert.rejects(snapshotWorkspace(d,['missing.rs']),/MISSING_REQUIRED_FILE/);
  }finally{await rm(d,{recursive:true,force:true});}
});
test('symlink in workspace fails closed before any upload',async()=>{
  const d=await mkdtemp(join(tmpdir(),'voder-links-'));
  try{await symlink('/etc/passwd',join(d,'my-link'));await assert.rejects(snapshotWorkspace(d),/SYMLINK_BLOCKED/);}
  finally{await rm(d,{recursive:true,force:true});}
});
test('workspace root symlink is rejected before snapshot',async()=>{
  const real=await mkdtemp(join(tmpdir(),'voder-real-'));
  const link=real+'-link';
  try{await writeFile(join(real,'file.txt'),'content');await symlink(real,link);
    await assert.rejects(snapshotWorkspace(link),/INVALID_WORKSPACE/);
  }finally{await rm(link,{force:true});await rm(real,{recursive:true,force:true});}
});
