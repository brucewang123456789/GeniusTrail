import {test} from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,writeFile,mkdir,rm} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {executeModalTask} from '../src/execute.ts';
const env={VODER_MODAL_APP:'voder-sandbox',VODER_MODAL_IMAGE:'debian@sha256:'+'a'.repeat(64),MODAL_TOKEN_ID:'test',MODAL_TOKEN_SECRET:'fake'};
function stream(text=''){return new ReadableStream({start(c){if(text)c.enqueue(text);c.close();}})}
function processResult(code=0,out=''){return {stdout:stream(out),stderr:stream(),async wait(){return code}}}
test('nested files require remote parent directories before upload',async()=>{
  const root=await mkdtemp(join(tmpdir(),'voder-nested-'));
  try{
    await mkdir(join(root,'src','nested'),{recursive:true});
    await writeFile(join(root,'src','nested','file.txt'),'OK');
    const directories=new Set(['/']);
    const uploads=[];
    const sb={sandboxId:'test',filesystem:{
      async writeBytes(bytes,path){const parent=path.slice(0,path.lastIndexOf('/'))||'/';if(!directories.has(parent))throw Error('ENOENT_REMOTE_PARENT:'+parent);uploads.push(path)},
      async writeText(_,path){const parent=path.slice(0,path.lastIndexOf('/'))||'/';if(!directories.has(parent))throw Error('ENOENT_REMOTE_PARENT:'+parent)},
    },async exec(cmd){if(cmd[0]==='/bin/mkdir'){for(const p of cmd.slice(2))directories.add(p);return processResult()};if(cmd[0]==='/usr/bin/sha256sum')return processResult();return processResult(0,'OK')},async terminate(){}};
    const task={workspace:root,program:'/usr/bin/cat',args:['src/nested/file.txt'],runId:'r1',actionId:'a1',timeoutMs:10000,memoryMiB:512,cpu:1,maxOutputBytes:1024};
    const got=await executeModalTask(task,{async create(){return sb}},env);
    assert.equal(got.exitCode,0);assert.deepEqual(uploads,['/workspace/src/nested/file.txt']);
  }finally{await rm(root,{recursive:true,force:true})}
});
