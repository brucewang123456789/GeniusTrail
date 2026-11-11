import {test} from 'node:test';import assert from 'node:assert/strict';import {mkdtemp,mkdir,writeFile,rm} from 'node:fs/promises';import {tmpdir} from 'node:os';import {join} from 'node:path';import {createHash} from 'node:crypto';
import {HostCandidateVerifier} from '../src/candidate-verifier.mjs';import {AgentKernel} from '../../voder-agent-core/src/kernel.mjs';import {PiModelHost} from '../../voder-agent-core/src/pi-host.mjs';import {EvidenceScheduler} from '../../voder-orchestration/src/index.mjs';
const sha=s=>createHash('sha256').update(s).digest('hex');
async function fixture(fn){const cwd=await mkdtemp(join(tmpdir(),'voder-verifier-quota-'));try{await mkdir(join(cwd,'src'));await writeFile(join(cwd,'src/a.txt'),'old\n');await fn(cwd);}finally{await rm(cwd,{force:true,recursive:true});}}
class FixtureVerifier extends HostCandidateVerifier {async _verifyCandidate(){this.actual=(this.actual??0)+1;return {status:'FAIL',backend:'OFFLINE_FIXTURE',diagnostic:['alpha','bravo','charlie','delta','echo'][this.actual-1]??'novel'};}}
function fakePi(script){return {getAgentDir:()=>'/trusted',SessionManager:{inMemory:()=>({})},DefaultResourceLoader:class{async reload(){}},defineTool:x=>x,async createAgentSession(o){const names=o.customTools.map(t=>t.name);return {session:{getActiveToolNames:()=>names,getCallableToolNames:()=>names,getAllTools:()=>names.map(name=>({name,exposure:'direct'})),subscribe(){return()=>{}},dispose(){},abort(){},async prompt(){await script(o.customTools);}}};}};}
test('physical verifier does not impose a three-check session limit',()=>fixture(async cwd=>{
 const v=new FixtureVerifier({cwd});assert.deepEqual(v.getBudget('s'),{limit:null,used:0,remaining:null});
 for(let i=0;i<5;i++)await v.verifyCandidate({}, {sessionId:'s'});
 assert.deepEqual(v.getBudget('s'),{limit:null,used:5,remaining:null});assert.equal(v.getBudget('other').remaining,null);
 assert.equal(v.actual,5);
}));
test('fourth material candidate receives real sandbox verdict, not verifier budget denial',()=>fixture(async cwd=>{
 const v=new FixtureVerifier({cwd});let last;
 const host=new PiModelHost({upstream:fakePi(async tools=>{
  const t=tools.find(t=>t.name==='voder_verify_candidate');
  for(let i=0;i<4;i++)last=JSON.parse((await t.execute(String(i),{candidate:{relative_path:'src/a.txt',old_sha256:sha('old\n'),new_content:'new '+i}},new AbortController().signal)).content[0].text);
 })});
 const kernel=new AgentKernel({cwd,sessionId:'quota',modelHost:host,candidateVerifier:v,broker:{}});const r=await kernel.submit({goal:'repair'});
 assert.equal(r.status,'BLOCKED');assert.equal(r.blocked_reason,'NO_NATIVE_COMMIT');assert.equal(last.voder_tool_result?.status,'FAIL');assert.equal(v.actual,4);
 assert.equal(r.orchestration.verification_calls,4);assert.equal(r.orchestration.remaining_verification_calls,null);
 kernel.close();
}));
test('explicit stricter operator limit remains binding',()=>{
 const e=new EvidenceScheduler({maxVerificationCalls:1});e.bindVerificationLimit(3);assert.equal(e.snapshot().remaining_verification_calls,1);
 assert.throws(()=>e.bindVerificationLimit(0),{code:'INVALID_ORCHESTRATION_POLICY'});
});
test('old three completions do not forbid a new task session preflight',()=>fixture(async cwd=>{
 const v=new FixtureVerifier({cwd});for(let i=0;i<3;i++)await v.verifyCandidate({}, {sessionId:'spent'});
 let models=0;const k=new AgentKernel({cwd,sessionId:'spent',candidateVerifier:v,broker:{},modelHost:{async createSession(){models++;return {async prompt(){return {}},dispose(){}}}}});
 const r=await k.submit({goal:'repair'});assert.equal(r.status,'BLOCKED');assert.equal(models,1);
}));
