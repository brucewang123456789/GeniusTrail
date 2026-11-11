'use strict';
const test=require('node:test');const assert=require('node:assert/strict');
const {PassThrough}=require('node:stream');const {EventEmitter}=require('node:events');const {RpcClient}=require('../src/rpc-client.cjs');
const {resolveFolderForTerminal,classifyStatus,canSubmitInMode}=require('../src/terminal-mapping.cjs');
const {randomUUID}=require('node:crypto');
const node=(value)=>({uri:{fsPath:value},name:value});
test('workspace identity: never silently bind ambiguous or unrelated folders',()=>{
 const a=node('/work/alpha'),b=node('/work/beta');
 assert.equal(resolveFolderForTerminal([a,b],{creationOptions:{cwd:'/work/alpha/sub'}}),a);
 assert.equal(resolveFolderForTerminal([a,b],{creationOptions:{cwd:'/work/alphabeta'}}),null);
 assert.equal(resolveFolderForTerminal([a,b],{creationOptions:{}}),null);
 assert.equal(resolveFolderForTerminal([a],{creationOptions:{}}),a);
});
test('verify status comes from authoritative native receipt summary, never string alone',()=>{
 assert.equal(classifyStatus({status:'VERIFIED',result:{verified:false}}).verified,false);
 assert.equal(classifyStatus({status:'VERIFIED'}).verified,false);
 assert.equal(classifyStatus({status:'VERIFIED',result:{verified:true}}).verified,true);
 assert.equal(classifyStatus({status:'VERIFIED',verified:true}).verified,true);
 assert.equal(classifyStatus({status:'DRAFT_UNVERIFIED',verified:false}).verified,false);
});
test('requires trust and gateway unless explicitly unverified draft',()=>{
 assert.equal(canSubmitInMode(false,'native','/private.json'),false);
 assert.equal(canSubmitInMode(true,'native',''),false);
 assert.equal(canSubmitInMode(true,'draft',''),true);
});
function mockSpawn(){const child=new EventEmitter();child.stdin=new PassThrough();child.stdout=new PassThrough();child.stderr=new PassThrough();child.kill=()=>{child.emit('exit',null,'SIGTERM');};return child;}
test('RPC frames: multiple responses, notifications, unique identities and error codes',async()=>{
 const child=mockSpawn();const rpc=new RpcClient({args:['a'],cwd:'/tmp',spawnFn:()=>child});
 const observed=[];rpc.on('event',p=>observed.push(p));
 const p1=rpc.request('initialize');const p2=rpc.request('session.create');
 child.stdout.write(JSON.stringify({jsonrpc:'2.0',method:'session.event',params:{sessionId:randomUUID(),event:{type:'task.inspect'}}})+'\n');
 child.stdout.write(JSON.stringify({jsonrpc:'2.0',id:2,result:{sessionId:randomUUID()}})+'\n'+JSON.stringify({jsonrpc:'2.0',id:1,result:{protocol:'voder.control.v1'}})+'\n');
 assert.equal((await p1).protocol,'voder.control.v1');assert.ok((await p2).sessionId);assert.equal(observed.length,1);
 const p3=rpc.request('session.diff');child.stdout.write(JSON.stringify({jsonrpc:'2.0',id:3,error:{code:'NOT_AVAILABLE',message:'no native diff'}})+'\n');
 await assert.rejects(p3,e=>e.code==='NOT_AVAILABLE');rpc.close();
});
test('RPC request framing, failure closure and deliberate no automatic replay',async()=>{
 const child=mockSpawn();const rpc=new RpcClient({args:[],cwd:'/tmp',spawnFn:()=>child});
 await assert.rejects(rpc.request('task.submit',{goal:'x'.repeat(80000)}),e=>e.code==='RPC_FRAME_TOO_LARGE');
 const pending=rpc.request('task.wait',{sessionId:randomUUID()});child.emit('exit',1,null);
 await assert.rejects(pending,e=>e.code==='RPC_EXIT');
 await assert.rejects(rpc.request('task.submit'),e=>e.code==='RPC_DISCONNECTED');
});
test('RPC rejects invalid JSON rather than silently treating as user content',async()=>{
 const child=mockSpawn();const rpc=new RpcClient({args:[],cwd:'/tmp',spawnFn:()=>child});
 const p=rpc.request('initialize');child.stdout.write('not json\n');
 await assert.rejects(p,e=>e.code==='RPC_BAD_JSON');
});
