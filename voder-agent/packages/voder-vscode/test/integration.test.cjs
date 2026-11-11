'use strict';
const test=require('node:test'),assert=require('node:assert/strict');const {mkdtemp,mkdir,rm,stat}=require('node:fs/promises');const {tmpdir}=require('node:os');const path=require('node:path');
const {RpcClient}=require('../src/rpc-client.cjs');
const agent=process.env.VODER_AGENT_TEST_ROOT||path.resolve(__dirname,'../../../../voder-agent');
test('REAL SDK/CLI RPC draft contract: create, list, status, events, fail-closed diff, cancel historical',async t=>{
 const working=await mkdtemp(path.join(tmpdir(),'voder-vscode-rpc-'));
 await mkdir(path.join(working,'journal'),{mode:0o700});
 const cli=path.join(agent,'packages/voder-cli/bin/voder-cli.mjs');
 try{await stat(cli);}catch{t.skip('Set VODER_AGENT_TEST_ROOT to six-module voder-agent source');await rm(working,{recursive:true,force:true});return;}
 const rpc=new RpcClient({command:process.execPath,args:[cli,'rpc','--workspace',working,'--session-dir',path.join(working,'journal'),'--draft'],cwd:working});
 try{
  const init=await rpc.request('initialize');assert.equal(init.protocol,'voder.control.v1');
  assert.equal(init.capabilities.session_scoped_diff,false);
  assert.equal(init.capabilities.native_resume,false);
  assert.equal(init.capabilities.vgr_commit,'native_only');
  const s=await rpc.request('session.create');assert.match(s.sessionId,/^[a-f0-9-]{36}$/i);
  const list=await rpc.request('session.list');assert.ok(list.some(item=>item.session_id===s.sessionId));
  const state=await rpc.request('session.status',{sessionId:s.sessionId});assert.equal(state.status,'CREATED');
  const ev=await rpc.request('session.events',{sessionId:s.sessionId,after:0});assert.deepEqual(ev,[]);
  const acceptance=await rpc.request('task.submit',{sessionId:s.sessionId,task:{goal:'Examine the missing required file before any action',requiredFiles:['user_supplied_required_missing_file_863921.txt']}});
  assert.equal(acceptance.status,'ACCEPTED');
  const blocked=await rpc.request('task.wait',{sessionId:s.sessionId});
  assert.equal(blocked.status,'WAITING_FOR_INPUT');
  assert.equal(blocked.verified,false);
  const later=await rpc.request('session.status',{sessionId:s.sessionId});
  assert.equal(later.status,'WAITING_FOR_INPUT');
  const diff=await rpc.request('session.diff',{sessionId:s.sessionId});assert.equal(diff.status,'NOT_AVAILABLE');
  await assert.rejects(rpc.request('session.cancel',{sessionId:s.sessionId}),e=>e.code==='SESSION_NOT_ACTIVE');
  const resumed=await rpc.request('session.resume',{sessionId:s.sessionId});assert.equal(resumed.canContinue,false);
 }finally{rpc.close();await rm(working,{recursive:true,force:true});}
});
