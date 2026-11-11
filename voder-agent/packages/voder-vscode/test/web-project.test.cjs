'use strict';
const {TestChatViewProvider}=require('./chat-test-context.cjs');
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs/promises'),os=require('node:os'),path=require('node:path'),Module=require('node:module');
const {execFileSync,spawnSync}=require('node:child_process'),{EventEmitter}=require('node:events');
const {isProjectGoal,planProject,createProject,validateProject,TARGETS,digest}=require('../src/project-workspace.cjs');
const {ProjectRunService}=require('../src/project-run.cjs');
const {VoderChatViewProvider}=require('../src/chat-view.cjs');
const ID='12345678-1234-4234-8234-123456789abc';
const VALID={
 'index.html':'<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>Animation fixture</title><link rel="stylesheet" href="style.css"></head><body><canvas id="scene"></canvas><script src="app.js"></script></body></html>',
 'app.js':'const draw=()=>{requestAnimationFrame(draw);}; requestAnimationFrame(draw);\n',
 'style.css':'html,body{margin:0;background:#080808;color:white}canvas{width:100%;height:100vh}\n'
};
async function fixture(fn){
 const host=await fs.mkdtemp(path.join(os.tmpdir(),'voder-web-contract-')),agentRoot=path.join(host,'voder-agent'),storageRoot=path.join(host,'storage');
 try{
  for(const d of [agentRoot,storageRoot,path.join(host,'.voder-build/release'),path.join(host,'packages/voder-sandbox-modal/src'),path.join(host,'.voder-tools/node/bin')])await fs.mkdir(d,{recursive:true,mode:0o700});
  for(const f of ['.voder-build/release/voder-gateway','.voder-build/release/voderd','packages/voder-sandbox-modal/src/cli.ts'])await fs.writeFile(path.join(host,f),'test fixture only\n',{mode:0o600});
  await fs.writeFile(path.join(host,'.voder-tools/node/bin/node'),'#!/bin/sh\nexec '+JSON.stringify(process.execPath)+' "$@"\n',{mode:0o700});
  const marker=path.join(agentRoot,'unchanged.txt');await fs.writeFile(marker,'existing project must stay unchanged');
  await fn({host,agentRoot,storageRoot,marker,plan:await planProject({agentRoot,storageRoot})});
  assert.equal(await fs.readFile(marker,'utf8'),'existing project must stay unchanged');
 }finally{await fs.rm(host,{recursive:true,force:true});}
}
function fakeRpc({submitError=false,result={status:'DELIVERED',verified:true,delivery_status:'DELIVERED',commit_digest:'a'.repeat(64)}}={}){
 const rpc=new EventEmitter();rpc.calls=[];rpc.request=async(name,args)=>{rpc.calls.push([name,args]);if(name==='session.create')return{sessionId:ID};if(name==='session.subscribe')return{};if(name==='task.submit'){if(submitError)throw Object.assign(Error('timeout'),{code:'RPC_TIMEOUT'});return{status:'ACCEPTED'};}if(name==='task.wait')return result;if(name==='session.cancel')return{status:'CANCEL_REQUESTED'};throw Error(name);};return rpc;
}
test('new web intent is explicit; read-only and ordinary repair keep the existing entry',()=>{
 for(const g of ['创建一个独立网页项目：鹦鹉骑自行车','Create a new web project showing a cycling parrot.','创建一个独立网页项目。不要修改已有源码。'])assert.equal(isProjectGoal(g),true,g);
 for(const g of ['只读调查如何创建一个独立网页项目','修复 app.js','给我鹦鹉骑自行车的建议','不要创建一个独立网页项目','Read-only: create a new web project'])assert.equal(isProjectGoal(g),false,g);
});
test('managed creation uses a new Git root, exact three-file batch and pinned validator; cannot overwrite a project',()=>fixture(async({plan,storageRoot,host})=>{
 assert.equal(await fs.stat(plan.root).then(()=>true,()=>false),false);
 const project=await createProject(plan),cfg=JSON.parse(await fs.readFile(project.profile,'utf8')),policy=JSON.parse(await fs.readFile(cfg.policyFile,'utf8'));
 assert.deepEqual(cfg.targetPaths,TARGETS);assert.equal(policy.schema,'voder.gateway.policy.v2');assert.equal(policy.repository_path,plan.root);assert.deepEqual(policy.target_paths,TARGETS);assert.equal(cfg.targetPath,'app.js');
 assert.deepEqual(execFileSync('/usr/bin/git',['-C',project.root,'ls-files'],{encoding:'utf8'}).trim().split('\n'),['app.js','index.html','style.css','verify_web.py']);
 await assert.rejects(createProject(plan),e=>e.code==='EEXIST');assert.equal(await fs.readFile(path.join(plan.root,'app.js'),'utf8'),'/* Voder managed blank preimage; awaiting generation. */\n');
 await fs.writeFile(path.join(project.root,'verify_web.py'),'# tampered\n');await assert.rejects(validateProject(project.recordPath,storageRoot),/validator changed/);
}));
test('structural validator rejects blank scaffold, unsafe HTML and missing animation; never claims visual validation',()=>fixture(async({plan})=>{
 const p=await createProject(plan),run=()=>spawnSync('python3',['-B','verify_web.py'],{cwd:p.root,encoding:'utf8'});
 assert.notEqual(run().status,0);
 for(const [n,c] of Object.entries(VALID))await fs.writeFile(path.join(p.root,n),c);
 const ok=run();assert.equal(ok.status,0,ok.stderr);assert.match(ok.stdout,/visuals NOT tested/);
 await fs.writeFile(path.join(p.root,'index.html'),VALID['index.html'].replace('src="app.js"','src="https://example.invalid/app.js"'));assert.notEqual(run().status,0);
 await fs.writeFile(path.join(p.root,'index.html'),VALID['index.html']);await fs.writeFile(path.join(p.root,'app.js'),'console.log("static content is not an animation");\n');assert.notEqual(run().status,0);
}));
test('project scope refuses profile widening and linked output paths before opening SDK',()=>fixture(async({plan,storageRoot,host})=>{
 const p=await createProject(plan),cfg=JSON.parse(await fs.readFile(p.profile,'utf8'));cfg.targetPaths.push('../outside.py');await fs.writeFile(p.profile,JSON.stringify(cfg));let opens=0;
 const svc=new ProjectRunService(async()=>{opens++;return fakeRpc();},storageRoot);await assert.rejects(svc.start({uri:{fsPath:p.root}},'create an animation',()=>{},{project:p}),/profile changed/);assert.equal(opens,0);
 const second=await planProject({agentRoot:path.join(host,'voder-agent'),storageRoot});await fs.symlink(path.join(host,'voder-agent'),second.root);await assert.rejects(createProject(second),e=>e.code==='EEXIST');
}));
test('project RPC preserves exact goal, uses one batch action, records identity before dispatch and never replays uncertain submit',()=>fixture(async({plan,storageRoot})=>{
 const p=await createProject(plan),folder={uri:{fsPath:p.root}},goal='创建一个独立网页项目：鹦鹉骑自行车',rpc=fakeRpc(),updates=[];let recorded=false;
 const svc=new ProjectRunService(async()=>rpc,storageRoot);const run=await svc.start(folder,goal,m=>updates.push(m),{project:p,onCreated:async id=>{assert.equal(id,ID);recorded=true;}});assert.equal(recorded,true);assert.equal((await run.completion).verified,true);
 const task=rpc.calls.find(x=>x[0]==='task.submit')[1].task;assert.ok(task.goal.endsWith(goal));assert.match(task.goal,/ONE batch candidate/);assert.deepEqual(task.requiredFiles,TARGETS);assert.equal(task.maxActions,1);assert.equal(rpc.calls.filter(x=>x[0]==='task.submit').length,1);
 const broken=fakeRpc({submitError:true}),uncertain=[],svc2=new ProjectRunService(async()=>broken,storageRoot);await assert.rejects(svc2.start(folder,goal,m=>uncertain.push(m),{project:p}),/timeout/);assert.equal(uncertain.at(-1).kind,'uncertain');assert.equal(broken.calls.filter(x=>x[0]==='task.submit').length,1);
}));
test('actual provider routes new project before existing folder/scope discovery; duplicate submit and forged approval cannot start it',async()=>{
 let starts=0,announced;const waiting=new Promise(r=>announced=r),messages=[];
 const app={trusted:()=>true,chooseFolder:()=>{throw Error('must not select an existing project');},webProjects:{start:async(provider,view)=>{starts++;const ok=await provider.requestNativeApproval(view,{target:'/isolated/web-exact',validator:'structural only'});messages.push({decision:ok});}},chatRuns:{start(){throw Error('old write entry must not run');}}};
 const provider=new TestChatViewProvider(app),view={webview:{cspSource:'vscode-webview:',postMessage:m=>{messages.push(m);if(m.type==='approval_requested')announced(m);},onDidReceiveMessage(){}},onDidDispose(){}};provider.resolveWebviewView(view);
 const running=provider.handle(view,{type:'submit',goal:'Create a new web project for the parrot animation.'}),approval=await waiting;
 await assert.rejects(provider.handle(view,{type:'submit',goal:'Create a new web project again'}),e=>e.code==='CHAT_BUSY');
 await provider.handle(view,{type:'authorization_decision',requestId:'forged',decision:'approve'});assert.equal(provider.approvals.size,1);
 await provider.handle(view,{type:'authorization_decision',requestId:approval.requestId,decision:'decline'});await running;assert.equal(starts,1);assert.equal(messages.at(-1).decision,false);assert.equal(provider.pending.size,0);
});
// Preview runs generated JS only in an unprivileged Webview, using digest-checked diff bytes.
function loadEntry(api){const orig=Module._load;delete require.cache[require.resolve('../src/project-entry.cjs')];try{Module._load=function(id,...args){return id==='vscode'?api:orig.call(this,id,...args);};return require('../src/project-entry.cjs').ProjectEntry;}finally{Module._load=orig;}}
test('preview requires exact native diff, checks JS syntax without execution, and has no privileged message bridge',()=>fixture(async({plan,storageRoot})=>{
 const p=await createProject(plan);for(const[n,c]of Object.entries(VALID))await fs.writeFile(path.join(p.root,n),c);
 let panel,disposed;const uri=f=>({fsPath:f,toString:()=>f});const api={Uri:{file:uri},ViewColumn:{Beside:2},window:{createWebviewPanel:(kind,title,col,opts)=>{panel={opts,webview:{cspSource:'vscode-webview:',asWebviewUri:u=>({toString:()=>`vscode-webview://local/${path.basename(u.fsPath)}`})},onDidDispose:f=>disposed=f};return panel;}}};
 const ProjectEntry=loadEntry(api),records={},app={context:{globalStorageUri:{fsPath:storageRoot},globalState:{get:(k,d)=>records[k]??d}},_open:async()=>({diffCapable:true,request:async(name)=>{assert.equal(name,'session.diff');return{schema:'voder.session.diff.v2',status:'VERIFIED_DELIVERED',authority:'VODER_NATIVE_VGR_STATE',sessionId:ID,commit_digest:'a'.repeat(64),action_id:'action-'+ID,run_id:'run-action-'+ID,files:TARGETS.map(n=>({relative_path:n,before:'',before_sha256:digest(''),after:VALID[n],after_sha256:digest(VALID[n])}))};}})};
 const entry=new ProjectEntry(app);entry.records.set(ID,p);await entry.preview(ID);assert.match(panel.webview.html,/connect-src 'none'/);assert.match(panel.webview.html,/script-src 'nonce-/);assert.match(panel.webview.html,/vscode-webview:\/\/local\/app.js/);assert.equal(panel.webview.onDidReceiveMessage,undefined);assert.equal(panel.opts.localResourceRoots.length,1);assert.notEqual(panel.opts.localResourceRoots[0].fsPath,p.root);
 await fs.writeFile(path.join(p.root,'app.js'),VALID['app.js']+'//changed');await assert.rejects(entry.preview(ID),/changed after delivery/);disposed();
}));
test('controller decline creates nothing; approved unverified result never opens preview or touches existing source',()=>fixture(async({host,agentRoot,storageRoot})=>{
 const prior={};for(const k of ['VODER_MODEL_ID','VODER_MODEL_API_KEY','MODAL_TOKEN_ID','MODAL_TOKEN_SECRET','VODER_MODAL_APP']){prior[k]=process.env[k];process.env[k]='offline-test-only';}
 try{
  const state={},uri=f=>({fsPath:f}),api={Uri:{file:uri}},ProjectEntry=loadEntry(api);let opens=0,previewed=0;
  const app={context:{globalStorageUri:uri(storageRoot),globalState:{get:(k,d)=>state[k]??d,update:async(k,v)=>{state[k]=v;}}},_agentRootForChat:async()=>agentRoot,_trackChatSession(){},_open:async()=>{opens++;return fakeRpc({result:{status:'DRAFT_UNVERIFIED',verified:false,draft:'Model did not provide a verified candidate'}});}};
  const entry=new ProjectEntry(app),view={},messages=[],provider={views:new Set([view]),pending:new Map([[view,{}]]),post:(v,m)=>messages.push(m),requestNativeApproval:async()=>false};entry.preview=async()=>previewed++;
  const msg={goal:'Create a new web project with a cycling parrot.'};await entry.start(provider,view,msg);assert.equal(opens,0);assert.equal(await fs.stat(path.join(host,'voder-projects')).then(()=>true,()=>false),false);assert.equal(messages.at(-1).type,'not_started');
  provider.requestNativeApproval=async()=>true;await entry.start(provider,view,msg);assert.equal(opens,1);assert.equal(previewed,0);assert.equal(messages.some(m=>m.type==='project_delivered'),false);assert.equal(messages.find(m=>m.type==='settled').verified,false);assert.ok(state['voder.webProjects'][ID]);
 }finally{for(const[k,v]of Object.entries(prior)){if(v===undefined)delete process.env[k];else process.env[k]=v;}}
}));
test('actual SDK verifier accepts the managed three-file policy through OFFLINE modeled transport; rejects incomplete batch',()=>fixture(async({plan,storageRoot,host})=>{
 const p=await createProject(plan),cfg=JSON.parse(await fs.readFile(p.profile,'utf8'));
 const transport=path.join(host,'offline-modeled-transport.cjs');
 await fs.writeFile(transport,`// OFFLINE TEST DOUBLE: local Python execution, NOT a real Modal receipt.\nconst fs=require('node:fs'),{spawnSync}=require('node:child_process'),{createHash}=require('node:crypto');const [request,receipt,out,err]=process.argv.slice(2);const r=JSON.parse(fs.readFileSync(request));const s=spawnSync('python3',['-B','verify_web.py'],{cwd:r.workspace});fs.writeFileSync(out,s.stdout);fs.writeFileSync(err,s.stderr);const hash=b=>createHash('sha256').update(b).digest('hex');fs.writeFileSync(receipt,JSON.stringify({protocol:'voder-sandbox-receipt-v1',provider:'modal',status:'EXECUTED',runId:r.runId,actionId:r.actionId,image:process.env.VODER_MODAL_IMAGE,stdoutSha256:hash(s.stdout),stderrSha256:hash(s.stderr),exitCode:s.status,timeout:false,outputLimited:false}));\n`);
 const agent=process.env.VODER_AGENT_TEST_ROOT;assert.ok(agent,'VODER_AGENT_TEST_ROOT required');const {pathToFileURL}=require('node:url');const {HostCandidateVerifier}=await import(pathToFileURL(path.join(agent,'packages/voder-sdk/src/candidate-verifier.mjs')).href);
 const verifier=new HostCandidateVerifier({cwd:p.root,nativeGateway:{...cfg,nodeBinary:process.execPath,modalCli:transport}}),candidate={patches:await Promise.all(TARGETS.map(async name=>({relative_path:name,old_sha256:digest(await fs.readFile(path.join(p.root,name))),new_content:VALID[name]})))};
 // The normalizer requires canonical key ordering semantics, not an external receipt authority.
 const result=await verifier.verifyCandidate(candidate,{sessionId:ID});assert.equal(result.status,'PASS');assert.match(result.diagnostic,/visuals NOT tested/);
 await assert.rejects(verifier.verifyCandidate({patches:candidate.patches.slice(0,2)},{sessionId:ID}),e=>e.code==='VERIFIER_SCOPE');
 assert.equal(await fs.readFile(path.join(p.root,'app.js'),'utf8'),'/* Voder managed blank preimage; awaiting generation. */\n');
}));
test('REAL Rust read-only preflight accepts the exact managed HTML/JS/CSS policy; no action or model dispatch',()=>fixture(async({plan})=>{
 const binary=process.env.VODER_NATIVE_TEST_GATEWAY;assert.ok(binary,'VODER_NATIVE_TEST_GATEWAY required');const p=await createProject(plan),cfg=JSON.parse(await fs.readFile(p.profile,'utf8'));
 const raw=execFileSync(binary,['--check-policy',cfg.policyFile],{timeout:12000,encoding:'utf8'}),reply=JSON.parse(raw);assert.equal(reply.status,'PASS');assert.deepEqual(reply.targets,TARGETS);assert.equal(reply.path_contract,'voder.source-path.v1');
 assert.equal(await fs.stat(path.join(cfg.attemptStore,'attempts')).then(()=>true),true);assert.deepEqual(await fs.readdir(path.join(cfg.attemptStore,'attempts')),[]);
}));
