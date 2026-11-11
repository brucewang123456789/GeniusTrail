import test from 'node:test';
import assert from 'node:assert/strict';
import {VgrBroker} from '../src/broker.mjs';
import {PiModelHost} from '../src/pi-host.mjs';
const sha='a'.repeat(64);
const action={schema:'voder.agent.action.v1',session_id:'s1',action_id:'a1',kind:'command',idempotency_key:'key1',workspace_scope:['src'],evidence_obligations:['obligation'],budget:{max_runtime_ms:100}};
function native(){return {async executeAuthorized(a){return {authority:'VODER_NATIVE_VGR',session_id:a.session_id,action_id:a.action_id,run_id:'run-1',status:'COMMITTED',commit_digest:sha,journal_digest:sha}}}};
test('COMMITTED-looking tool text is never accepted without an independent native verifier',async()=>{
 const b=new VgrBroker({native:native(),sessionId:'s1',workspaceRoot:'/tmp'});
 await assert.rejects(b.dispatch(action),{code:'NO_NATIVE_COMMIT_VERIFIER'});
 await assert.rejects(b.dispatch(action),{code:'DUPLICATE'});
});
test('verifier returning false cannot bless commit',async()=>{
 const b=new VgrBroker({native:native(),sessionId:'s1',workspaceRoot:'/tmp',commitVerifier:{async confirmCommit(){return {verified:false}}}});
 await assert.rejects(b.dispatch(action),{code:'UNVERIFIED_COMMIT'});
});
function fakePi({callable=['voder_authorized_action'],register=['voder_authorized_action']}={}){
 let fakeSession;
 return {getAgentDir:()=>'/tmp',SessionManager:{inMemory:()=>({})},DefaultResourceLoader:class{async reload(){}},defineTool:x=>x,
 createAgentSession:async ()=>{
 fakeSession={getActiveToolNames:()=>['voder_authorized_action'],getCallableToolNames:()=>callable,
 getAllTools:()=>register.map(name=>({name,exposure:'direct'})),async prompt(){},dispose(){}};
 return {session:fakeSession};
 }};
}
test('Pi host denies ambient callable tool even if only one active tool',async()=>{
 const host=new PiModelHost({upstream:fakePi({callable:['voder_authorized_action','bash']})});
 await assert.rejects(host.createSession({cwd:'/tmp',systemPrompt:'test',tools:[{name:'voder_authorized_action',description:'safe',execute(){}}]}),{code:'UNSAFE_TOOLS'});
});
test('Pi host denies unregistered extra tool even if not active or callable',async()=>{
 const host=new PiModelHost({upstream:fakePi({register:['voder_authorized_action','tool_search']})});
 await assert.rejects(host.createSession({cwd:'/tmp',systemPrompt:'test',tools:[{name:'voder_authorized_action',description:'safe',execute(){}}]}),{code:'UNSAFE_TOOLS'});
});
test('Pi host denies invented model-facing tools',async()=>{
 const host=new PiModelHost({upstream:fakePi()});
 await assert.rejects(host.createSession({cwd:'/tmp',systemPrompt:'test',tools:[{name:'bash',description:'danger',execute(){}}]}),{code:'INVALID_TOOL_SET'});
});
