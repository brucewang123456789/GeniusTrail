import {test} from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {createHash} from 'node:crypto';
import {AgentKernel} from '../src/kernel.mjs';
import {HostCandidateVerifier} from '../../voder-sdk/src/candidate-verifier.mjs';
import {EvidenceScheduler} from '../../voder-orchestration/src/index.mjs';
const sha=x=>createHash('sha256').update(x).digest('hex');
const profile=id=>({provider:'test-'+id,id,contextWindow:32768,maxTokens:2048});
async function fixture(fn){const cwd=await mkdtemp(join(tmpdir(),'voder-auto-relay-'));try{await mkdir(join(cwd,'src'));await writeFile(join(cwd,'src/file.txt'),'original\n');return await fn(cwd);}finally{await rm(cwd,{force:true,recursive:true});}}
function kernel(cwd,models,{start=0,onPrompt,onSwitch,verifier,broker={}}={}){
 const traces=[];let current=start,sessionCount=0;
 const host={async createSession({tools}){sessionCount++;return {
  async listModels(){return {current:models[current],models}},
  async switchModel({provider,id}){const n=models.findIndex(x=>x.provider===provider&&x.id===id);if(onSwitch)await onSwitch(n);if(n<0)throw Error('MODEL_NOT_ALLOWED');const prev=models[current];current=n;traces.push(['switch',prev.id,models[current].id]);return {status:'MODEL_SELECTED',provider,id}},
  async prompt(text){traces.push(['prompt',models[current].id,text]);await onPrompt?.({index:current,model:models[current],tools,text})},
  dispose(){}
 }}};
 const k=new AgentKernel({cwd,sessionId:'relay-test',modelHost:host,broker,candidateVerifier:verifier});
 return {k,traces,count:()=>sessionCount};
}
test('one configured API: no second model lookup wait or switch',()=>fixture(async cwd=>{
 const x=kernel(cwd,[profile('A')]);const r=await x.k.submit({goal:'fix'});
 assert.equal(r.status,'BLOCKED');assert.deepEqual(x.traces.map(t=>t.slice(0,2)),[['prompt','A']]);assert.equal(x.count(),1);
}));
test('two models automatic relay exactly once, preserving original task and cause',()=>fixture(async cwd=>{
 const x=kernel(cwd,[profile('A'),profile('B')]);const r=await x.k.submit({goal:'fix'});
 assert.equal(r.status,'BLOCKED');assert.deepEqual(x.traces.map(t=>t.slice(0,2)),[['prompt','A'],['switch','A'],['prompt','B']]);
 assert.match(x.traces[2][2],/Continue the SAME authorized task/);
 assert.equal(r.orchestration.model_stage,1);assert.equal(x.count(),1);
 assert.equal(x.k.events.filter(e=>e.type==='model.automatic_handoff').length,1);
}));
test('manual B selection stays independent; failed B relays to A',()=>fixture(async cwd=>{
 const x=kernel(cwd,[profile('A'),profile('B')],{start:1});const r=await x.k.submit({goal:'fix'});
 assert.equal(r.status,'BLOCKED');assert.deepEqual(x.traces.map(t=>t.slice(0,2)),[['prompt','B'],['switch','B'],['prompt','A']]);
}));
test('unavailable second API is skipped, third gets prior evidence without deadlock',()=>fixture(async cwd=>{
 const x=kernel(cwd,[profile('A'),profile('B'),profile('C')],{onSwitch:async n=>{if(n===1)throw Object.assign(Error('auth'),{code:'MODEL_AUTH_UNAVAILABLE'})}});
 const r=await x.k.submit({goal:'fix'});assert.equal(r.status,'BLOCKED');
 assert.deepEqual(x.traces.map(t=>t.slice(0,2)),[['prompt','A'],['switch','A'],['prompt','C']]);
 assert.equal(x.k.events.filter(e=>e.type==='model.handoff_skipped').length,1);
}));
test('new model cannot erase prior failed verifier candidate; can verify revised bytes and commit',()=>fixture(async cwd=>{
 const checked=[];
 const verifier={getBudget(){return {limit:null,used:checked.length,remaining:null}},async verifyCandidate(c){checked.push(c);return {status:'FAIL',backend:'offline',diagnostic:'patch did not apply'}}};
 const candidate={relative_path:'src/file.txt',old_sha256:sha('original\n'),new_content:'first attempt'};
 const x=kernel(cwd,[profile('A'),profile('B')],{verifier,onPrompt:async ({index,tools})=>{
  const verify=tools.find(t=>t.name==='voder_verify_candidate');
  if(index===0){assert.equal((await verify.execute({candidate})).status,'FAIL');}
  else{
   await assert.rejects(verify.execute({candidate}),{code:'DUPLICATE_CANDIDATE'});
   assert.equal((await verify.execute({candidate:{...candidate,new_content:'fixed attempt'}})).status,'FAIL');
  }
 }});
 const r=await x.k.submit({goal:'fix'});assert.equal(r.status,'BLOCKED');assert.equal(r.orchestration.verification_calls,2);assert.equal(checked.length,2);
}));
test('no physical three-probe quota: five distinct sandbox outcomes are recorded',()=>fixture(async cwd=>{
 class Sandbox extends HostCandidateVerifier{async _verifyCandidate(){return {status:'FAIL',backend:'offline'}}}
 const checker=new Sandbox({cwd});for(let i=0;i<5;i++)assert.equal((await checker.verifyCandidate({},{sessionId:'case'})).status,'FAIL');
 assert.deepEqual(checker.getBudget('case'),{limit:null,used:5,remaining:null});
 const e=new EvidenceScheduler({maxToolCalls:32});for(let i=0;i<5;i++)await e.inspectCandidate({relative_path:'src/file.txt',old_sha256:sha('original\n'),new_content:'attempt '+i},{sourceDigest:async()=>sha('original\n'),run:async()=>({status:'FAIL',backend:'offline',diagnostic:'different'+i})});
 assert.equal(e.snapshot().verification_calls,5);assert.equal(e.snapshot().remaining_verification_calls,null);
}));
test('model-stage extension preserves all tool and candidate evidence, changes only spend allowance',async()=>{
 const e=new EvidenceScheduler({maxToolCalls:8});const c={relative_path:'src/file.txt',old_sha256:sha('old'),new_content:'new'};
 const run=async()=>({status:'FAIL',backend:'offline',diagnostic:'bad patch'});
 await e.inspectCandidate(c,{sourceDigest:async()=>sha('old'),run});const before=e.snapshot();
 e.advanceModel();assert.equal(e.snapshot().verification_calls,1);assert.equal(e.snapshot().remaining_tool_calls,8);assert.equal(e.snapshot().model_stage,1);
 await assert.rejects(e.inspectCandidate(c,{sourceDigest:async()=>sha('old'),run}),{code:'DUPLICATE_CANDIDATE'});
 assert.equal(before.positive_sandbox_checks,0);
});
test('nine configured providers chain without four-switch hard stop or repeated attempts',()=>fixture(async cwd=>{
 const models=Array.from({length:9},(_,i)=>profile('M'+i));const x=kernel(cwd,models);const r=await x.k.submit({goal:'fix'});
 assert.equal(r.status,'BLOCKED');assert.equal(x.traces.filter(e=>e[0]==='prompt').length,9);
 assert.equal(x.traces.filter(e=>e[0]==='switch').length,8);assert.equal(r.orchestration.model_stage,8);
}));
test('authenticated native commit on first model prevents calling any backup',()=>fixture(async cwd=>{
 const broker={delivery:null};let nativeCalls=0;
 const x=kernel(cwd,[profile('A'),profile('B')],{broker,onPrompt:async()=>{nativeCalls++;x.k.nativeCommit={status:'COMMITTED'};}});
 const r=await x.k.submit({goal:'fix'});
 assert.equal(r.status,'VERIFIED');assert.equal(nativeCalls,1);assert.equal(x.traces.filter(e=>e[0]==='switch').length,0);
}));
test('repeated diagnostic stagnation is released only on an authenticated model-stage transition',async()=>{
 const {DiagnosticJournal}=await import('../../voder-diagnostics/src/index.mjs');const d=new DiagnosticJournal();
 for(let i=0;i<3;i++)d.observeSandbox({status:'FAIL',diagnostic:'identical compiler failure'},{candidateDigest:sha('same-patch')});
 assert.throws(()=>d.assertMayProceed(),{code:'DIAGNOSTIC_STAGNATION'});
 assert.equal(d.snapshot().observed_sandbox_checks,3);d.nextModelStage();
 assert.doesNotThrow(()=>d.assertMayProceed());assert.equal(d.snapshot().observed_sandbox_checks,3);
});
test('primary API omitted but a configured secondary is usable without awaiting missing primary',async()=>{
 const {bindInstalledModel}=await import('../src/model-plugin-runtime.mjs');const dir=await mkdtemp(join(tmpdir(),'voder-catalog-'));const path=join(dir,'catalog.json');
 const item={provider:'voder-secondary-test',id:'second',api:'openai-completions',baseUrl:'https://api.example.com/v1',apiKey:'mock-api-key',contextWindow:32768,maxTokens:2048,reasoning:false,thinkingLevel:'off'};
 try{
  await writeFile(path,JSON.stringify({version:1,models:[item]}),{mode:0o600});
  const providers=new Map();const runtime={registerProvider(name,p){providers.set(name,p)},getModel(name,id){const x=providers.get(name)?.models.find(m=>m.id===id);return x?{...x,provider:name}:null}};
  const api={ModelRuntime:{async create(){return runtime}}};
  const bound=await bindInstalledModel(api,{},[],{catalogFile:path});
  assert.equal(bound.model.id,'second');assert.equal(bound.scopedModels.length,1);assert.equal(bound.limits.source,'CONFIGURED_MODEL_CATALOG');
 }finally{await rm(dir,{recursive:true,force:true})}
});
test('real Pi host boundary switches models without losing transcript or per-model request allowance',()=>fixture(async cwd=>{
 const {PiModelHost}=await import('../src/pi-host.mjs');const models=[profile('A'),profile('B')].map(x=>({...x,input:['text']}));
 let n=0;const transcript=[];const api={
  getAgentDir:()=>'/test',SessionManager:{inMemory:()=>({})},
  DefaultResourceLoader:class{constructor(o){this.o=o}async reload(){}},
  defineTool:x=>x,normalizeContext:x=>x,
  async createAgentSession(o){const names=o.tools;const s={
   model:models[0],messages:[],systemPrompt:o.resourceLoader.o.systemPromptOverride(),
   agent:{streamFunction(){n++;return {}}},
   setModel(m){this.model=m},
   getActiveToolNames:()=>names,getCallableToolNames:()=>names,getAllTools:()=>names.map(name=>({name,exposure:'direct'})),
   subscribe(){return()=>{}},dispose(){},
   async prompt(goal){this.agent.streamFunction(this.model,{messages:[{role:'system',content:this.systemPrompt},...this.messages]},{});
    transcript.push({model:this.model.id,goal,history:this.messages.length});
    this.messages.push({role:'user',content:goal});this.messages.push({role:'assistant',content:[{type:'text',text:'I need to fix this'}],stopReason:'stop'});
   }
  };return {session:s};}
 };
 const modelHost=new PiModelHost({upstream:api,modelProfiles:models,maxModelRequests:1});
 const k=new AgentKernel({cwd,modelHost,broker:{}});const r=await k.submit({goal:'fix'});
 assert.equal(r.status,'BLOCKED');assert.equal(n,2);
 assert.deepEqual(transcript.map(x=>x.model),['A','B']);assert.equal(transcript[1].history,2);
 assert.match(transcript[1].goal,/previous model checkpoint/);
}));
