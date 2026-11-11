import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp,writeFile,rm } from 'node:fs/promises';
import {tmpdir} from 'node:os';import{join}from'node:path';
import {AgentKernel,VgrBroker,preflightFiles,VoderError,assembleSystemPrompt,PiModelHost} from '../src/index.mjs';
const sha='a'.repeat(64);
async function tmp(fn){const d=await mkdtemp(join(tmpdir(),'voder-core-test-'));try{return await fn(d)}finally{await rm(d,{recursive:true,force:true})}}
function req(session_id='session1'){return {schema:'voder.agent.action.v1',session_id,action_id:'action1',kind:'command',idempotency_key:'idem1',workspace_scope:['src'],evidence_obligations:['test_exit_zero'],budget:{max_runtime_ms:5000}}}
test('true missing file blocks before model call',async()=>tmp(async cwd=>{let called=0;const k=new AgentKernel({cwd,sessionId:'session1',modelHost:{createSession(){called++;throw Error('must not call')}} ,broker:{}});const r=await k.submit({goal:'fix',requiredFiles:['absent.py']});assert.equal(r.status,'WAITING_FOR_INPUT');assert.equal(called,0);}));
test('optional known file succeeds preflight',async()=>tmp(async cwd=>{await writeFile(join(cwd,'input.py'),'print(1)');const x=await preflightFiles(cwd,['input.py']);assert.equal(x.status,'READY');}));
test('traversal and symlink omitted from read preflight',async()=>tmp(async cwd=>{await assert.rejects(()=>preflightFiles(cwd,['../.env']),{code:'INVALID_PATH'})}));
test('native transport mandatory, identity+digest pinned, duplicate refused',async()=>{let count=0;const b=new VgrBroker({sessionId:'session1',workspaceRoot:'/nowhere',commitVerifier:{async confirmCommit(r,a){return {verified:true,authority:'VODER_NATIVE_VGR_STATE',session_id:a.session_id,action_id:a.action_id,run_id:r.run_id,commit_digest:r.commit_digest}}},native:{async executeAuthorized(a){count++;return {run_id:'run1',authority:'VODER_NATIVE_VGR',session_id:a.session_id,action_id:a.action_id,status:'COMMITTED',journal_digest:sha,commit_digest:sha}}}});assert.equal((await b.dispatch(req())).status,'COMMITTED');await assert.rejects(()=>b.dispatch(req()),{code:'DUPLICATE'});assert.equal(count,1);});
test('faked native receipt cannot be accepted and never retried',async()=>{const b=new VgrBroker({sessionId:'session1',workspaceRoot:'/none',native:{async executeAuthorized(a){return {authority:'VODER_NATIVE_VGR',session_id:a.session_id,action_id:'other',status:'COMMITTED',journal_digest:sha,commit_digest:sha}}}});await assert.rejects(()=>b.dispatch(req()),{code:'UNVERIFIED'});await assert.rejects(()=>b.dispatch(req()),{code:'DUPLICATE'});});
test('model text cannot manufacture delivery',async()=>tmp(async cwd=>{const k=new AgentKernel({cwd,sessionId:'session1',broker:{dispatch(){}},modelHost:{async createSession(){return {async prompt(){return {nativeCommit:{authority:'VODER_NATIVE_VGR',status:'COMMITTED'}}}}}}});const r=await k.submit({goal:'fix'});assert.equal(r.status,'BLOCKED');}));
test('system prompt includes all 14 rules and is bounded',()=>{const prompt=assembleSystemPrompt('repair',{task:'patch files'});for(let n=1;n<=14;n++)assert.match(prompt,new RegExp('\\b'+String(n).padStart(2,'0')+' '));assert.match(prompt,/ROOT_CAUSE_REPAIR/);assert.throws(()=>assembleSystemPrompt('repair',{task:'x',maxBytes:50}),{code:'CONTEXT_OVERFLOW'});});
test('unprivileged default tools not passed to upstream',async()=>{let opts;const fakePi={getAgentDir:()=>'/unused',SessionManager:{inMemory:()=>({})},DefaultResourceLoader:class{constructor(o){opts={...o}} async reload(){}}, defineTool:x=>x,createAgentSession:async o=>{opts=o;return {session:{getActiveToolNames:()=>['voder_authorized_action'],getCallableToolNames:()=>['voder_authorized_action'],getAllTools:()=>[{name:'voder_authorized_action',exposure:'direct'}],async prompt(){},dispose(){}}}}};const host=new PiModelHost({upstream:fakePi});const sess=await host.createSession({cwd:'/tmp',systemPrompt:'p',tools:[{name:'voder_authorized_action',description:'x',execute(){}}]});assert.deepEqual(opts.tools,['voder_authorized_action']);assert.equal(opts.customTools.length,1);sess.dispose();});

test('committed receipt from a previous submission cannot verify a new task',async()=>tmp(async cwd=>{
 const session={async prompt(){return {settled:true}}};
 const k=new AgentKernel({cwd,sessionId:'session1',broker:{dispatch(){}},modelHost:{async createSession(){return session}}});
 k.nativeCommit={status:'COMMITTED'};
 const r=await k.submit({goal:'a new task'});
 assert.equal(r.status,'BLOCKED');
 assert.equal(k.nativeCommit,null);
}));
