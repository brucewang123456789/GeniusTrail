import {test} from 'node:test';
import assert from 'node:assert/strict';
import {cp,mkdtemp,mkdir,readFile,writeFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
import {spawnSync} from 'node:child_process';
const HERE=fileURLToPath(new URL('../src', import.meta.url));
const hash=bytes=>createHash('sha256').update(bytes).digest('hex');
const modalMock=`export class ModalClient {
  apps={async fromName(){return {}}};images={fromRegistry(name){return name}};
  sandboxes={async create(){
    const files=new Map();const dirs=new Set(['/']);
    const processResult=(exit,out='',err='')=>({wait:async()=>exit,
      stdout:new ReadableStream({start(c){if(out)c.enqueue(out);c.close()}}),
      stderr:new ReadableStream({start(c){if(err)c.enqueue(err);c.close()}})});
    return {sandboxId:'mock-sandbox',filesystem:{
      async writeBytes(bytes,p){const d=p.slice(0,p.lastIndexOf('/'))||'/';if(!dirs.has(d))throw Error('NO_REMOTE_PARENT');files.set(p,bytes)},
      async writeText(text,p){const d=p.slice(0,p.lastIndexOf('/'))||'/';if(!dirs.has(d))throw Error('NO_REMOTE_PARENT');files.set(p,text)},
    },async exec(command){if(command[0]==='/bin/mkdir'){for(const dir of command.slice(2))dirs.add(dir);return processResult(0)};
      if(command[0]==='/usr/bin/sha256sum')return processResult(files.has('/workspace/src/input.txt')?0:1);
      if(command[0]==='/usr/bin/cat')return processResult(Number(process.env.MOCK_EXIT||0),'REMOTE_STDOUT\\n','REMOTE_STDERR\\n');
      throw Error('UNEXPECTED_REMOTE_COMMAND:'+command[0]);},async terminate(){}}
  }};
}`;
test('real Node CLI writes isolated evidence files and metadata even with Node warning',async()=>{
  const d=await mkdtemp(join(tmpdir(),'voder-cli-ipc-'));
  try{
    await cp(HERE,join(d,'src'),{recursive:true});
    await mkdir(join(d,'node_modules','modal'),{recursive:true});
    await writeFile(join(d,'package.json'),JSON.stringify({type:'module'}));
    await writeFile(join(d,'node_modules','modal','package.json'),JSON.stringify({type:'module',exports:'./index.mjs'}));
    await writeFile(join(d,'node_modules','modal','index.mjs'),modalMock);
    await mkdir(join(d,'workspace','src'),{recursive:true});
    await writeFile(join(d,'workspace','src','input.txt'),'REMOTE_STDOUT\n');
    const task={workspace:join(d,'workspace'),program:'/usr/bin/cat',args:['src/input.txt'],
      requiredFiles:['src/input.txt'],runId:'r1',actionId:'a1',timeoutMs:10000,memoryMiB:512,cpu:1,maxOutputBytes:4096};
    const paths=['request.json','receipt.json','stdout.bin','stderr.bin'].map(p=>join(d,p));
    await writeFile(paths[0],JSON.stringify(task));
    const env={...process.env,MODAL_TOKEN_ID:'TEST_ID',MODAL_TOKEN_SECRET:'TEST_SECRET',VODER_MODAL_APP:'voder-sandbox',VODER_MODAL_IMAGE:'debian@sha256:'+'f'.repeat(64),MOCK_EXIT:'13'};
    const launched=spawnSync(process.execPath,['--experimental-strip-types',join(d,'src','cli.ts'),...paths],{env,encoding:'utf8',timeout:10_000});
    assert.equal(launched.status,0,launched.stderr);
    // Ownership diagnostics are host stderr, never part of observed remote stderr.
    // Node 24 no longer emits the strip-types warning emitted by older runtimes.
    const hostDiagnostics=launched.stderr.replace(/^VODER_OWNED_SANDBOX_ID=mock-sandbox\r?\n/gm,'');
    assert.match(hostDiagnostics, /^VODER_MODAL_LIFECYCLE=/m);
    assert.match(hostDiagnostics, /^VODER_MODAL_STAGE=END; NAME=SANDBOX_CLEANUP/m);
    assert.ok(!hostDiagnostics.includes('REMOTE_STDERR'));
    const stdout=await readFile(paths[2]);const stderr=await readFile(paths[3]);
    assert.equal(stdout.toString(),'REMOTE_STDOUT\n');assert.equal(stderr.toString(),'REMOTE_STDERR\n');
    const receipt=JSON.parse(await readFile(paths[1],'utf8'));
    assert.equal(receipt.exitCode,13);
    assert.equal(receipt.stdoutSha256,hash(stdout));assert.equal(receipt.stderrSha256,hash(stderr));
    assert.ok(!('stdout' in receipt) && !('stderr' in receipt));
    assert.equal(receipt.status,'EXECUTED');
  }finally{await rm(d,{recursive:true,force:true})}
});
