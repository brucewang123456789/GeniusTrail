/** Offline regression probes. Any model/native verifier here is a TEST DOUBLE, never production evidence. */
import {test} from 'node:test';import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,rm,cp,symlink,readdir} from 'node:fs/promises';
import {tmpdir} from 'node:os';import {join,resolve,dirname} from 'node:path';import {fileURLToPath} from 'node:url';import {createHash,randomUUID} from 'node:crypto';import {createRequire} from 'node:module';import {spawnSync} from 'node:child_process';
import {AgentKernel} from '../../voder-agent-core/src/kernel.mjs';import {HostTaskWorkspace,taskToolImages,isHostTaskTool} from '../../voder-agent-core/src/task-workspace.mjs';
import {hostPatchPreflight} from '../../voder-sdk/src/candidate-verifier.mjs';import {renderSvg,processBytes} from '../src/svg-preview.mjs';
import {makeGeneralControl} from '../src/general-control.mjs';
const require=createRequire(import.meta.url),{launchGeneral}=require('../src/general-workspace.cjs'),{GeneralRunService}=require('../src/general-run.cjs');
const sha=b=>createHash('sha256').update(b).digest('hex');
const svg='<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1024 768"><title>Regression fixture</title><desc>Offline renderer input, not the requested artwork</desc><circle id="ball" cx="300" cy="300" r="120" fill="red"/></svg>';
let renderAvailable=true;try{await renderSvg(svg);}catch(e){if(e.code==='SVG_RENDERER_UNAVAILABLE')renderAvailable=false;else throw e;}
const pass={status:'PASS',backend:'TEST_DOUBLE_ONLY',sandboxed:true,checks:[{name:'STRUCTURE_FIXTURE',passed:true}],evidence_digest:'a'.repeat(64)};
async function fixture(fn){const dir=await mkdtemp(join(tmpdir(),'voder-ordinary-'));try{return await fn(dir);}finally{await rm(dir,{recursive:true,force:true});}}
const workspace=(overrides={})=>new HostTaskWorkspace({prepare:async()=>({status:'READY'}),render:async()=>({result:{status:'RENDERED'},images:[]}),review:async()=>({status:'MODEL_REVIEW_RECORDED'}),hash:async c=>({sha256:sha(c)}),assess:async()=>({schema:'voder.task-acceptance.v1',task_correctness:'NOT_INDEPENDENTLY_CERTIFIED'}),...overrides});
const tool=(tools,name)=>{const t=tools.find(x=>x.name===name);assert.ok(t,'tool missing '+name);return t;};
test('explicit task mode permits inspection and revised candidate after PASS, then commits exactly once',()=>fixture(async cwd=>{
 await writeFile(join(cwd,'art.svg'),'before\n');let commits=0;
 const broker={async dispatch(){commits++;return {status:'COMMITTED'};}};
 const host={async createSession({tools}){return {async prompt(){
  const first={relative_path:'art.svg',old_sha256:sha('before\n'),new_content:svg};
  const r=await tool(tools,'voder_verify_candidate').execute({candidate:first});assert.equal(commits,0);assert.equal(r.next_step.status,'CHECKS_PASSED_NOT_COMMITTED');
  const next=await tool(tools,'voder_verify_candidate').execute({candidate:{...first,new_content:svg+'\n'}});assert.equal(commits,0);
  await tool(tools,'voder_commit_verified_candidate').execute({candidate_digest:next.candidate_digest});assert.equal(commits,1);
  await tool(tools,'voder_commit_verified_candidate').execute({candidate_digest:next.candidate_digest});assert.equal(commits,1);
 },dispose(){}};}};
 const k=new AgentKernel({cwd,broker,modelHost:host,candidateVerifier:{verifyCandidate:async()=>pass},commitMode:'explicit',taskWorkspace:workspace()});
 const result=await k.submit({goal:'Create editable artwork'});assert.equal(result.status,'VERIFIED');assert.equal(result.task_acceptance.task_correctness,'NOT_INDEPENDENTLY_CERTIFIED');
}));
test('explicit native unknown outcome fences later verification and never retries',()=>fixture(async cwd=>{
 await writeFile(join(cwd,'a.txt'),'old');let calls=0;const host={async createSession({tools}){return{async prompt(){const r=await tool(tools,'voder_verify_candidate').execute({candidate:{relative_path:'a.txt',old_sha256:sha('old'),new_content:'new'}});await assert.rejects(tool(tools,'voder_commit_verified_candidate').execute({candidate_digest:r.candidate_digest}));await assert.rejects(tool(tools,'voder_verify_candidate').execute({candidate:{relative_path:'a.txt',old_sha256:sha('old'),new_content:'other'}}),{code:'NATIVE_OUTCOME_UNCERTAIN'});},dispose(){}};}};
 const k=new AgentKernel({cwd,modelHost:host,broker:{async dispatch(){calls++;throw Error('transport disconnected');}},candidateVerifier:{verifyCandidate:async()=>pass},commitMode:'explicit'});const r=await k.submit({goal:'fix'});assert.equal(calls,1);assert.equal(r.blocked_reason,'NATIVE_OUTCOME_UNCERTAIN');
}));
test('post-commit acceptance reporting failure cannot hide a confirmed native commit',()=>fixture(async cwd=>{
 await writeFile(join(cwd,'a.txt'),'old');const k=new AgentKernel({cwd,taskWorkspace:workspace({assess:async()=>{throw Error('report fault');}}),modelHost:{async createSession({tools}){return {async prompt(){await tool(tools,'voder_verify_candidate').execute({candidate:{relative_path:'a.txt',old_sha256:sha('old'),new_content:'new'}});},dispose(){}};}},broker:{dispatch:async()=>({status:'COMMITTED'})},candidateVerifier:{verifyCandidate:async()=>pass}});
 const r=await k.submit({goal:'fix'});assert.equal(r.status,'VERIFIED');assert.equal(r.task_acceptance.limitations,'POST_DELIVERY_REPORT_FAILED');
}));
test('a normal textual answer is returned without a JSON plan or false native verification',()=>fixture(async cwd=>{
 const k=new AgentKernel({cwd,taskWorkspace:workspace(),commitMode:'explicit',broker:{dispatch:async()=>assert.fail('unexpected write')},candidateVerifier:{verifyCandidate:async()=>assert.fail('unexpected sandbox')},modelHost:{async createSession(){return{prompt:async()=>({draft:'This is an ordinary explanation, not JSON.'}),dispose(){}};}}});
 const r=await k.submit({goal:'Explain the files'});assert.equal(r.status,'DRAFT_UNVERIFIED');assert.equal(r.draft,'This is an ordinary explanation, not JSON.');assert.equal(r.sandbox,'NO_NATIVE_COMMIT');
}));
test('new task capabilities cannot be forged by naming an unapproved read/write tool',()=>{
 assert.equal(isHostTaskTool({name:'voder_task_prepare'}),false);const tools=workspace().makeTools();assert.ok(tools.every(isHostTaskTool));
});
test('task scope tool rejects stale callbacks and unexpected arguments',async()=>{
 let prepared=0;const t=tool(workspace({prepare:async()=>{prepared++;}}).makeTools(),'voder_task_prepare');await assert.rejects(t.execute({paths:['a'],binary:'/bin/sh'}),{code:'TASK_TOOL_ARGUMENTS'});assert.equal(prepared,0);
 const stale=workspace().makeTools({stillCurrent(){throw Object.assign(Error('stale'),{code:'STALE_TASK'});}});await assert.rejects(tool(stale,'voder_task_prepare').execute({paths:['a']}),{code:'STALE_TASK'});
});
test('SVG parser rejects external resources, scripting, duplicate IDs and broken references',async()=>{
 for(const content of [svg.replace('<circle','<script/> <circle'),svg.replace('<circle','<image href="file:///etc/passwd"/> <circle'),svg.replace('<circle','<use href="#absent"/> <circle'),svg.replace('<circle','<g id="ball"/> <circle'),svg.replace('<circle','<style>@import "x.css";</style><circle')])await assert.rejects(renderSvg(content));
});
test('actual parser and renderer produce full/reduced PNGs; source hash is not visual review',{skip:!renderAvailable},async()=>{
 const r=await renderSvg(svg);assert.equal(r.check.width,1024);assert.equal(r.check.height,768);assert.equal(r.images.length,2);assert.equal(r.images[0].readUInt32BE(16),1024);assert.equal(r.images[1].readUInt32BE(16),512);assert.equal(r.images[0].readUInt32BE(20),768);assert.equal(r.images[1].readUInt32BE(20),384);
});
test('image exposure is tracked separately from renderer success and requires vision support',async()=>{
 let exposed=false;const host=workspace({render:async()=>({result:{status:'RENDERED'},images:[{type:'image',mimeType:'image/png',data:'fixture'}],onExposed:()=>{exposed=true;}})}),r=await tool(host.makeTools(),'voder_task_render_svg').execute({path:'art.svg',content:svg});
 assert.equal(exposed,false);assert.deepEqual(taskToolImages(r,{supported:false}),[]);assert.equal(exposed,false);assert.equal(taskToolImages(r,{supported:true}).length,1);assert.equal(exposed,true);assert.deepEqual(taskToolImages({...r},{supported:true}),[]);
});
test('renderer subprocess cancellation stops work and does not report success',async()=>{
 const abort=new AbortController();abort.abort();await assert.rejects(processBytes(process.execPath,['-e','setInterval(()=>{},1000)'],'',{signal:abort.signal}),{code:'CANCELLED'});
});
test('real Git checks root patches at root even when an unrelated project directory exists',()=>fixture(async cwd=>{
 await mkdir(join(cwd,'project'));await mkdir(join(cwd,'evidence'));await writeFile(join(cwd,'a.txt'),'old\n');await writeFile(join(cwd,'project/keep.txt'),'unrelated\n');assert.equal(spawnSync('git',['init','-q',cwd]).status,0);
 const patch='diff --git a/a.txt b/a.txt\n--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-old\n+new\n',args={repo:cwd,dir:join(cwd,'evidence'),members:[{relative_path:'change.patch',new_content:patch}],candidate_digest:sha(patch)};
 assert.equal(await hostPatchPreflight(args),null);assert.equal(await readFile(join(cwd,'a.txt'),'utf8'),'old\n');
}));
test('explicit patch root works for nested source without inference and rejects escaping roots',()=>fixture(async cwd=>{
 await mkdir(join(cwd,'nested'));await mkdir(join(cwd,'evidence'));await writeFile(join(cwd,'nested/a.txt'),'old\n');spawnSync('git',['init','-q',cwd]);
 const p='diff --git a/a.txt b/a.txt\n--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-old\n+new\n',a={repo:cwd,dir:join(cwd,'evidence'),members:[{relative_path:'change.patch',new_content:p}],candidate_digest:sha(p)};
 await assert.rejects(hostPatchPreflight({...a,patchBaseDir:'../escape'}),{code:'VERIFIER_PATCH_ROOT'});assert.equal(await hostPatchPreflight({...a,patchBaseDir:'nested'}),null);
}));
test('actual executing kernel can prepare new SVG/OBJ/game files without a planning session (native infrastructure is a preflight-only TEST DOUBLE)',()=>fixture(async host=>{
 const root=join(host,'task'),storageRoot=join(host,'storage'),agentRoot=join(host,'voder-agent');
 for(const d of [root,storageRoot,agentRoot,join(host,'.voder-build/release'),join(host,'packages/voder-sandbox-modal/src'),join(host,'.voder-tools/node/bin')])await mkdir(d,{recursive:true,mode:0o700});
 await cp(resolve(dirname(fileURLToPath(import.meta.url)),'../..'),join(agentRoot,'packages/voder-vscode'),{recursive:true});
 // Other core modules are copied from the current repair source, never loaded from installed cloud state.
 for(const n of ['voder-sdk','voder-agent-core','voder-native-bridge','voder-orchestration','voder-context','voder-tools','voder-skills','voder-memory','voder-task-planner','voder-diagnostics','voder-search'])await cp(resolve(dirname(fileURLToPath(import.meta.url)),'../../',n),join(agentRoot,'packages',n),{recursive:true});
 await writeFile(join(host,'.voder-build/release/voder-gateway'),'#!/usr/bin/python3\nimport sys,json\np=json.load(open(sys.argv[2]))\nprint(json.dumps({"schema":"voder.gateway.preflight.v1","path_contract":"voder.source-path.v1","targets":p.get("target_paths",[p.get("target_path")]),"status":"PASS"}))\n',{mode:0o700});
 await writeFile(join(host,'.voder-build/release/voderd'),'TEST_DOUBLE_NEVER_EXECUTED\n',{mode:0o600});await writeFile(join(host,'packages/voder-sandbox-modal/src/cli.ts'),'TEST_DOUBLE_NEVER_EXECUTED\n',{mode:0o600});
 await writeFile(join(host,'.voder-tools/node/bin/node'),'#!/bin/sh\nexec '+JSON.stringify(process.execPath)+' "$@"\n',{mode:0o700});
 const launch=await launchGeneral({root,agentRoot,storageRoot});let modelSessions=0,received;
 const hostModel={async createSession({tools,systemPrompt}){modelSessions++;assert.match(systemPrompt,/no separate JSON planning reply/);return {async prompt(goal){received=goal;
  const prep=await tool(tools,'voder_task_prepare').execute({paths:['pelican-bicycle.svg','verification.md','scene.obj','game.html']});assert.equal(prep.status,'READY');assert.equal(Object.keys(prep.files).length,4);assert.equal(prep.files['pelican-bicycle.svg'],sha(await readFile(join(root,'pelican-bicycle.svg'))));
  await assert.rejects(tool(tools,'voder_task_prepare').execute({paths:['../escape.py']}));
  // Stop before any remote sandbox/native execution. This is not the live pelican acceptance.
  return {draft:'Offline entry and scope probe only; no artwork delivery claimed.'};
 },dispose(){}};}};
 const control=await makeGeneralControl({cwd:root,sessionDir:join(storageRoot,'journal'),gate:launch.profile,agentRoot,modelHost:hostModel});
 const {sessionId}=await control.createSession({autoContinue:false});await control.submitTask(sessionId,{goal:await readFile(new URL('./fixtures/original-pelican.txt',import.meta.url),'utf8'),maxActions:1});const result=await control.wait(sessionId);
 assert.equal(modelSessions,1);assert.ok(received.startsWith('Create a polished, original SVG illustration'));assert.doesNotMatch(received,/READ-ONLY INVESTIGATION|TASK PLANNING/);assert.equal(result.status,'DRAFT_UNVERIFIED');assert.equal(result.verified,false);await assert.rejects(control.createSession(),{code:'GENERAL_SESSION_CONSUMED'});control.dispose();
 const restored=await makeGeneralControl({cwd:root,sessionDir:join(storageRoot,'journal'),gate:launch.profile,agentRoot,modelHost:{createSession:async()=>assert.fail('historical review started a model')}});
 assert.equal((await restored.status(sessionId)).status,'DRAFT_UNVERIFIED');await assert.rejects(restored.createSession(),{code:'GENERAL_SESSION_CONSUMED'});assert.equal(modelSessions,1);restored.dispose();
}));
test('Pi adapter returns two actual image blocks to vision models and tracks them on the next request',{skip:!renderAvailable},async()=>{
 const {PiModelHost}=await import('../../voder-agent-core/src/pi-host.mjs'),{EvidenceScheduler}=await import('../../voder-orchestration/src/index.mjs');
 const r=await renderSvg(svg),model={provider:'fixture',id:'vision',contextWindow:1048576,maxTokens:393216,input:['text','image']};let exposed=false,requested=false,requests=0;
 const w=workspace({render:async()=>({result:{status:'RENDERED'},images:r.images.map(b=>({type:'image',mimeType:'image/png',data:b.toString('base64')})),onExposed:()=>{exposed=true;},onRequested:()=>{requested=true;}})});
 const api={getAgentDir:()=>'/fixture',SessionManager:{inMemory:()=>({})},DefaultResourceLoader:class{constructor(o){this.o=o;}async reload(){}},normalizeContext:x=>x,defineTool:x=>x,async createAgentSession(o){const names=o.tools;const s={model,systemPrompt:o.resourceLoader.o.systemPromptOverride(),messages:[],agent:{streamFunction(){requests++;return{};}},getActiveToolNames:()=>names,getCallableToolNames:()=>names,getAllTools:()=>names.map(name=>({name,exposure:'direct'})),subscribe:()=>()=>{},dispose(){},async prompt(){
  const t=o.customTools.find(x=>x.name==='voder_task_render_svg'),reply=await t.execute('render',{path:'art.svg',content:svg},new AbortController().signal);assert.equal(reply.content.filter(x=>x.type==='image').length,2);assert.equal(exposed,true);assert.equal(requested,false);
  s.messages=[{role:'toolResult',toolCallId:'render',toolName:t.name,content:reply.content,isError:false,timestamp:1}];
  s.agent.streamFunction(model,{messages:[{role:'system',content:s.systemPrompt},...s.messages]},{});assert.equal(requested,true);
  s.messages.push({role:'assistant',stopReason:'stop',content:[{type:'text',text:'Fixture images inspected; not a live model.'}]});
 }};return{session:s};}};
 const h=new PiModelHost({upstream:api}),sched=new EvidenceScheduler(),s=await h.createSession({cwd:process.cwd(),systemPrompt:'FIXTURE RULES',tools:[{name:'voder_authorized_action',description:'TEST DOUBLE never called',execute:async()=>assert.fail('native call')},...w.makeTools()],toolControl:()=>sched.control()});await s.prompt('offline image transport');assert.equal(requests,1);s.dispose();
});
test('actual chat entry submits the original prompt once without asking a model for JSON first',async(t)=>{
 const Module=require('node:module'),{EventEmitter}=require('node:events'),originalLoad=Module._load;
 const selected=await mkdtemp(join(tmpdir(),'voder-ordinary-entry-'));t.after(()=>rm(selected,{recursive:true,force:true}));
 const folder={uri:{fsPath:selected},name:'task'},prompt=await readFile(new URL('./fixtures/original-pelican.txt',import.meta.url),'utf8');let submitted=0,submittedGoal,launches=0;
 const fakeLaunch={root:folder.uri.fsPath,profile:'/private/session.json',nodeBinary:process.execPath};
 const entryPath=require.resolve('../src/general-entry.cjs');delete require.cache[entryPath];
 Module._load=function(name,parent,...args){if(name==='vscode')return{Uri:{file:p=>({fsPath:p})}};if(parent?.filename===entryPath&&name==='./native-env.cjs')return{nativeEnv:()=>({})};if(parent?.filename===entryPath&&name==='./task-workspace.cjs')return{planTask:async()=>({})};if(parent?.filename===entryPath&&name==='./ordinary-workspace.cjs')return{ordinaryWorkspace:async({root})=>({root,isolated:false})};if(parent?.filename===entryPath&&name==='./general-workspace.cjs')return{launchGeneral:async()=>{launches++;return fakeLaunch;}};return originalLoad.call(this,name,parent,...args);};
 let Entry;try{Entry=require(entryPath).TaskEntry;}finally{Module._load=originalLoad;delete require.cache[entryPath];}
 const state=new Map(),events=[],rpc=new EventEmitter(),id=randomUUID();rpc.request=async(method,p)=>{if(method==='session.create')return{sessionId:id};if(method==='task.submit'){submitted++;submittedGoal=p.task.goal;return{status:'ACCEPTED'};}if(method==='task.wait')return{status:'BLOCKED',verified:false};return{};};
 const app={context:{globalStorageUri:{fsPath:'/private'},globalState:{get:(k,d)=>state.get(k)??d,update:async(k,v)=>state.set(k,v)}},chatRuns:{activeSession:()=>null,inspect:async()=>assert.fail('unexpected read-only planner')},_agentRootForChat:async()=>'/agent',_trackChatSession(){},_open:async()=>rpc};
 const entry=new Entry(app),view={},provider={views:new Set([view]),pending:new Map([[view,{}]]),post:(_,x)=>events.push(x),requestNativeApproval:async()=>false};
 let approvals=0;provider.requestNativeApproval=async()=>{approvals++;return false;};
 await entry.start(provider,view,{goal:prompt},folder);assert.equal(approvals,0);assert.equal(submitted,1);assert.equal(submittedGoal,prompt);assert.equal(launches,1);assert.equal(events.some(x=>x.phase==='plan'||x.type==='task_planned'),false);assert.equal(events.some(x=>x.type==='task_delivered'),false);assert.equal(entry.reserved.size,0);
});
test('novel task observations advance task evidence without resetting verifier or native budgets',()=>fixture(async cwd=>{
 let revisions=[];const k=new AgentKernel({cwd,taskWorkspace:workspace(),commitMode:'explicit',broker:{dispatch:async()=>assert.fail('unexpected write')},candidateVerifier:{verifyCandidate:async()=>assert.fail('unexpected sandbox')},modelHost:{async createSession({tools,toolControl}){return{async prompt(){const h=tool(tools,'voder_task_content_hash');revisions.push(toolControl().task_evidence_revisions);await h.execute({content:'first'});revisions.push(toolControl().task_evidence_revisions);await h.execute({content:'first'});revisions.push(toolControl().task_evidence_revisions);await h.execute({content:'second'});revisions.push(toolControl().task_evidence_revisions);assert.equal(toolControl().verification_calls,0);assert.equal(toolControl().attempted_native_actions,0);return{draft:'Offline evidence probe only'};},dispose(){}};}}});await k.submit({goal:'explain'});assert.deepEqual(revisions,[0,1,1,2]);
}));
test('protecting unrelated files or tests does not downgrade creation to read-only investigation',()=>{
 const {writeRequest,readOnlyRequest}=require('../src/entry-orchestrator.cjs');
 for(const goal of ['Build a game; do not modify unrelated files.','Create an SVG. Do not modify tests.','请绘制一个三维模型，不要修改测试文件']){assert.equal(writeRequest(goal),true);assert.equal(readOnlyRequest(goal),false);}
 for(const goal of ['Read-only: inspect this project.','请只读检查，不要修改任何文件','Do not modify any files.'])assert.equal(readOnlyRequest(goal),true);
});
