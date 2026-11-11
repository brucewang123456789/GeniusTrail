import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,rm,writeFile,cp,mkdir,readFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {spawnSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {createVoderContext,createVoderContextProvider} from '../src/index.mjs';
import {VoderSDK} from '../../voder-sdk/src/index.mjs';
import {AgentKernel} from '../../voder-agent-core/src/kernel.mjs';
const sha=x=>createHash('sha256').update(x).digest('hex');
async function workspace(fn){const d=await mkdtemp(join(tmpdir(),'voder-context-closeout-'));try{return await fn(d)}finally{await rm(d,{recursive:true,force:true})}}

test('real default lifecycle executes host preflight and delivery exactly once without extra model turn or tool grants',async()=>workspace(async cwd=>{
 let newSessions=0,modelTurns=0,toolNames;
 const sdk=new VoderSDK({cwd,native:{async executeAuthorized(){throw Error('unexpected write')}},modelHost:{async createSession({tools}){
  newSessions++;toolNames=tools.map(x=>x.name);return {async prompt(){modelTurns++;},dispose(){}};
 }}});
 const {sessionId}=sdk.createSession();
 const result=await sdk.submitTask(sessionId,{goal:'Review the design for portability'});
 assert.equal(result.status,'BLOCKED');assert.equal(newSessions,1);assert.equal(modelTurns,1);
 assert.deepEqual(toolNames,['voder_authorized_action','voder_read_file','voder_list_files','voder_search_code']);
 const events=sdk.getEvents(sessionId);
 assert.equal(events.filter(x=>x.type==='context.preflight').length,1);
 assert.equal(events.filter(x=>x.type==='context.composed').length,1);
 assert.equal(events.filter(x=>x.type==='context.delivery').length,1);
 assert.equal(events.find(x=>x.type==='context.delivery').outcome,'NOT_VERIFIED');
 assert.equal(events.filter(x=>x.type==='model.agent_start').length,0);
 assert.ok(events.find(x=>x.type==='context.composed').estimated_token_upper_bound<=48000);
 assert.equal(events.find(x=>x.type==='context.preflight').policy_digest.length,64);
 assert.ok(!JSON.stringify(events).includes('Review the design for portability'));
 sdk.dispose();
}));

test('truly missing required source blocks before model and no speculative optional resources trigger a block',async()=>workspace(async cwd=>{
 let modelCalls=0;
 const sdk=new VoderSDK({cwd,native:{async executeAuthorized(){throw Error('unreachable')}},modelHost:{async createSession(){modelCalls++;return {async prompt(){},dispose(){}}}}});
 let id=sdk.createSession().sessionId;
 const blocked=await sdk.submitTask(id,{goal:'修订 <<File name="missing-report.docx">>'});
 assert.equal(blocked.status,'WAITING_FOR_INPUT');assert.equal(modelCalls,0);
 assert.ok(!sdk.getEvents(id).some(x=>x.type==='context.preflight'));
 const another=sdk.createSession().sessionId;
 const normal=await sdk.submitTask(another,{goal:'Suggest a new Python architecture (no files provided)'});
 assert.equal(normal.status,'BLOCKED');assert.equal(modelCalls,1);
 sdk.dispose();
}));

test('policy lifecycle cannot convert a model claim into committed evidence',async()=>{
 const p=createVoderContextProvider();
 assert.equal(typeof p.preparePhase,'function');assert.equal(typeof p.reviewDelivery,'function');
 const prepared=await p.preparePhase({stage:'delivery',task:'verify result'});
 await assert.rejects(p.reviewDelivery({task:'verify result',status:'VERIFIED',nativeCommit:{status:'FAKE'},prepared}),{code:'DELIVERY_EVIDENCE_MISSING'});
 const unverified=await p.reviewDelivery({task:'verify result',status:'BLOCKED',prepared});
 assert.equal(unverified.outcome,'NOT_VERIFIED');
 const verified=await p.reviewDelivery({task:'verify result',status:'VERIFIED',nativeCommit:{status:'COMMITTED'},prepared});
 assert.equal(verified.outcome,'NATIVE_VERIFIED');
 assert.ok(Object.isFrozen(verified));
});

test('once native commit exists, a reporting-only failure never hides it or duplicates VGR writes',async()=>workspace(async cwd=>{
 let writes=0;
 const provider=async()=>'';
 provider.preparePhase=async ({stage})=>({stage,sha256:'a'.repeat(64),bytes:10});
 provider.reviewDelivery=async()=>{throw Object.assign(new Error('report crashed'),{code:'DELIVERY_RENDER_FAILED'});};
 const kernel=new AgentKernel({cwd,contextProvider:provider,broker:{async dispatch(){writes++;return {status:'COMMITTED',action_id:'x'};}},
  modelHost:{async createSession({tools}){return {async prompt(){await tools[0].execute({kind:'patch',relative_path:'src/a.py',old_sha256:'a'.repeat(64),new_content:'print(1)'});},dispose(){}}}}});
 const r=await kernel.submit({goal:'Fix code'});
 assert.equal(r.status,'VERIFIED');assert.equal(writes,1);
 assert.equal(kernel.events.filter(x=>x.type==='context.delivery_error').length,1);
 kernel.close();
}));

test('multilingual task language does not grant tools and matches relevant pinned installed skills',async()=>{
 const c=createVoderContext();
 const jp=await c.plan({task:'Rust コードのバグを修正して テスト 実行'});
 assert.ok(jp.selectedSkills.includes('root-cause-analysis'));
 assert.ok(jp.selectedSkills.includes('sandbox-verification'));
 const ko=await c.plan({task:'Go 서비스 오류를 수정하고 테스트'});
 assert.ok(ko.selectedSkills.includes('root-cause-analysis'));
 const python=await c.plan({task:'debug Python traceback failed program'});
 assert.ok(python.selectedSkills.includes('sandbox-verification'));
 const plain=await c.plan({task:'Review style guide and summarize'});
 assert.deepEqual(plain.selectedSkills,[]);
});

test('token-tight auto selection omits complete optional skills, never truncates sources',async()=>{
 const rich=await createVoderContext().plan({task:'debug failing physics simulation Python code in sandbox'});
 const base=await createVoderContext().plan({task:'Review architecture'});
 let found;
 for(let budget=2000;budget<rich.bytes;budget+=25){
  try {const p=await createVoderContext({maxSupplementBytes:budget}).plan({task:'debug failing physics simulation Python code in sandbox'});
  if(p.omittedSkills.length>0){found=p;break;}}
  catch(e){assert.equal(e.code,'CONTEXT_OVERFLOW');}
 }
 assert.ok(found,'a lower bounded budget must omit an optional whole skill');
 assert.ok(found.bytes<=rich.bytes && found.omittedSkills.length>0);
 for(const name of found.selectedSkills)assert.match(found.supplement,new RegExp(`\\[SKILL:${name} sha256=`));
 for(const name of found.omittedSkills)assert.doesNotMatch(found.supplement,new RegExp(`\\[SKILL:${name} sha256=`));
 await assert.rejects(createVoderContext({selectedSkills:['root-cause-analysis'],maxSupplementBytes:2000}).plan({task:'Inspect code'}),{code:'CONTEXT_OVERFLOW'});
 assert.ok(base.bytes<rich.bytes);
});

test('host signals are strict and may activate repair without putting errors in task prose',async()=>{
 const c=createVoderContext();
 const a=await c.plan({task:'Inspect this',signals:{diagnostic:true,requiresExecution:true}});
 assert.deepEqual(a.selectedSkills,['root-cause-analysis','sandbox-verification']);
 await assert.rejects(c.plan({task:'inspect',signals:{grantShell:true}}),{code:'INVALID_CONTEXT_SIGNALS'});
 await assert.rejects(c.plan({task:'inspect',signals:{diagnostic:'yes'}}),{code:'INVALID_CONTEXT_SIGNALS'});
});

test('policy manifest pins each MD against out-of-band modification and fails before model work',async()=>workspace(async cwd=>{
 // Run in a disposable tree: never mutate shared real module files used by concurrent tests.
 const r=join(cwd,'voder-agent');
 await mkdir(join(r,'packages/voder-agent-core/src'),{recursive:true});
 await cp(new URL('../src/',import.meta.url),join(r,'packages/voder-context/src'),{recursive:true});
 await cp(new URL('../policies/',import.meta.url),join(r,'packages/voder-context/policies'),{recursive:true});
 await cp(new URL('../skills/',import.meta.url),join(r,'packages/voder-context/skills'),{recursive:true});
 await cp(new URL('../../voder-agent-core/src/contracts.mjs',import.meta.url),join(r,'packages/voder-agent-core/src/contracts.mjs'));
 await writeFile(join(r,'packages/voder-context/package.json'),'{"type":"module"}');
 const policy=join(r,'packages/voder-context/policies/strategy.md');
 await writeFile(policy,(await readFile(policy,'utf8'))+'\nUNPINNED_ADDITION');
 const uri=new URL(`file://${join(r,'packages/voder-context/src/index.mjs')}`).href;
 const code=`import {createVoderContext} from ${JSON.stringify(uri)}; try{await createVoderContext().plan({task:'normal'});process.exit(3)}catch(e){if(e.code==='POLICY_INTEGRITY_MISMATCH')process.exit(0);process.stderr.write(e.code);process.exit(4)}`;
 const reply=spawnSync(process.execPath,['--input-type=module','-e',code],{encoding:'utf8',timeout:4000});
 assert.equal(reply.status,0,reply.stderr);
}));

test('oversized multilingual user context fails closed, never silently truncated or submitted to model',async()=>workspace(async cwd=>{
 let modelCalls=0;
 const sdk=new VoderSDK({cwd,native:{async executeAuthorized(){throw Error('no write')}},modelHost:{async createSession(){modelCalls++;return {async prompt(){},dispose(){}}}}});
 const {sessionId}=sdk.createSession();
 const big='修复代码。'.repeat(6000);
 await assert.rejects(sdk.submitTask(sessionId,{goal:big}),{code:'CONTEXT_OVERFLOW'});
 assert.equal(modelCalls,0);
 const state=sdk.getStatus(sessionId);
 assert.equal(state,'FAILED');
 sdk.dispose();
}));

test('structured host-controlled task signals are wired from SDK to actual model-facing skill selection',async()=>workspace(async cwd=>{
 let prompt;
 const sdk=new VoderSDK({cwd,contextSignals:{diagnostic:true},native:{async executeAuthorized(){throw Error('no write')}},
  modelHost:{async createSession({systemPrompt}){prompt=systemPrompt;return {async prompt(){},dispose(){}}}}});
 const id=sdk.createSession().sessionId;
 const result=await sdk.submitTask(id,{goal:'Review the existing design'});
 assert.equal(result.status,'BLOCKED');
 assert.match(prompt,/SKILL:root-cause-analysis sha256=/);
 assert.match(prompt,/ROOT_CAUSE_REPAIR/);
 sdk.dispose();
 assert.throws(()=>new AgentKernel({cwd,contextSignals:{shell:true},broker:{},modelHost:{async createSession(){}}}),{code:'INVALID_CONTEXT_SIGNALS'});
}));
