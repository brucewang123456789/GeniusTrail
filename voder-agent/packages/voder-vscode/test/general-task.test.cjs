'use strict';
const {TestChatViewProvider}=require('./chat-test-context.cjs');
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs/promises'),path=require('node:path'),os=require('node:os'),{execFileSync,spawnSync}=require('node:child_process'),{EventEmitter}=require('node:events');
const {parsePlan,planningGoal}=require('../src/task-plan.cjs');
const {planTask,createTask,validateTask,validatorFor}=require('../src/task-workspace.cjs');
const {ChatRunService}=require('../src/chat-core.cjs');
const {TaskRunService}=require('../src/task-run.cjs');
const ID='12345678-1234-4234-8234-123456789abc';
const PROMPT='Create a polished, original SVG illustration of a pelican riding a bicycle.\nTreat this as a complete visual engineering task: build the artwork, inspect the rendered result, repair defects, and deliver verified files.\nSave pelican-bicycle.svg, pelican-bicycle.png when rendering is available, and verification.md. Do not mark an unperformed check as passed.';
const response=p=>({status:'DRAFT_UNVERIFIED',verified:false,draft:JSON.stringify(p)});
const scope=(files=['pelican-bicycle.svg','verification.md'],checks=[])=>({action:'execute',answer:'Create editable source; binary rendering and visual inspection unavailable.',files,checks});
async function fixture(fn){const host=await fs.mkdtemp(path.join(os.tmpdir(),'voder-general-')),agentRoot=path.join(host,'voder-agent'),root=path.join(host,'task'),storageRoot=path.join(host,'storage');try{
 for(const d of [agentRoot,root,storageRoot,path.join(host,'.voder-build/release'),path.join(host,'packages/voder-sandbox-modal/src'),path.join(host,'.voder-tools/node/bin')])await fs.mkdir(d,{recursive:true,mode:0o700});
 // Inert infrastructure placeholders: these tests do not claim native/Rust/Modal execution.
 for(const f of ['.voder-build/release/voder-gateway','.voder-build/release/voderd','packages/voder-sandbox-modal/src/cli.ts'])await fs.writeFile(path.join(host,f),'TEST INFRASTRUCTURE PLACEHOLDER\n',{mode:0o600});
 await fs.writeFile(path.join(host,'.voder-tools/node/bin/node'),'#!/bin/sh\nexec '+JSON.stringify(process.execPath)+' "$@"\n',{mode:0o700});
 const git=(...a)=>execFileSync('git',['-C',root,...a],{stdio:'pipe'}).toString();
 await fn({host,agentRoot,root,storageRoot,git});
 }finally{await fs.rm(host,{recursive:true,force:true});}}
test('natural SVG/game/3D requests preserve original prompt and accept text scopes without fixed web files',()=>{
 for(const goal of [PROMPT,'Build a playable game with keyboard controls.','请创建一个三维模型并保存为 scene.obj']){assert.ok(planningGoal(goal).startsWith(goal+'\n'));assert.doesNotMatch(planningGoal(goal),/\[VODER READ-ONLY INVESTIGATION\]/);}
 for(const files of [['art.svg','verification.md'],['game.html'],['scene.obj','materials.mtl'],['src/main.rs','Cargo.toml']])assert.deepEqual(parsePlan(response(scope(files))).files,[...files].sort());
 assert.equal(parsePlan(response({action:'answer',answer:'Read-only answer'})).action,'answer');
});
test('planner output never grants unsafe scope or pretends binary support',()=>{
 for(const file of ['../escape.py','/root/file','a\\b','.env','node_modules/a.js','a/b/../x','CON.txt','output.png','render.blend'])assert.throws(()=>parsePlan(response(scope([file]))));
 assert.throws(()=>parsePlan(response(scope(['a','a/b']))));
 assert.throws(()=>parsePlan(response(scope(['a'],[['curl','https://example.com']]))));
 assert.throws(()=>parsePlan({status:'DELIVERED',verified:true,draft:'{}'}));
 assert.throws(()=>parsePlan(response({action:'execute',answer:'done',files:[],checks:[]})));
});
test('actual ChatRunService planning uses read-only transport but the execution planning contract, not investigation completion',async()=>{
 let captured;const rpc=new EventEmitter();rpc.request=async(method,args)=>{if(method==='session.create')return{sessionId:ID};if(method==='task.submit'){captured=args.task;return{status:'ACCEPTED'};}if(method==='task.wait')return response(scope());return{};};
 const runs=new ChatRunService(async(_,profile,options)=>{assert.equal(profile,null);assert.equal(options.readOnly,true);return rpc;});
 const task=await runs.start({uri:{fsPath:'/selected-task'}},PROMPT,()=>{},{planning:true,readOnly:true});assert.equal(parsePlan(await task.completion).action,'execute');assert.ok(captured.goal.startsWith(PROMPT+'\n'));assert.match(captured.goal,/TASK PLANNING/);assert.doesNotMatch(captured.goal,/\[VODER READ-ONLY INVESTIGATION\]/);
 await assert.rejects(runs.start({uri:{fsPath:'/selected-task'}},PROMPT,()=>{},{planning:true}),e=>e.code==='TASK_PLAN_POLICY');
});
test('real filesystem/Git preparation preserves unrelated staged bytes and snapshots selected dirty/new paths only',()=>fixture(async args=>{
 const {root,git}=args;await fs.writeFile(path.join(root,'scene.obj'),'v 0 0 0\n');await fs.writeFile(path.join(root,'unrelated.txt'),'original\n');git('init','-q');git('add','.');git('-c','user.name=test','-c','user.email=test@invalid','commit','-qm','baseline');
 await fs.writeFile(path.join(root,'scene.obj'),'v 1 2 3\n');await fs.writeFile(path.join(root,'unrelated.txt'),'preserve staged\n');git('add','unrelated.txt');
 const plan=await planTask({...args,scope:scope(['scene.obj','verification.md'])});assert.equal((await fs.readdir(root)).includes('verification.md'),false);
 const project=await createTask(plan);assert.equal(project.root,root);assert.equal(await fs.readFile(path.join(root,'scene.obj'),'utf8'),'v 1 2 3\n');assert.equal(git('diff','--cached','--name-only').trim(),'unrelated.txt');assert.equal(git('show','HEAD:unrelated.txt'),'original\n');assert.equal(await fs.readFile(path.join(root,'unrelated.txt'),'utf8'),'preserve staged\n');
 assert.equal((await validateTask(project.recordPath,args.storageRoot)).targets.length,2);
 await fs.appendFile(path.join(root,project.validatorName),'# tamper\n');await assert.rejects(validateTask(project.recordPath,args.storageRoot),/validator changed/);
}));
test('stale approval and symlink parents fail before creating candidate preimages',()=>fixture(async args=>{
 const plan=await planTask({...args,scope:scope(['art.svg'])});await fs.writeFile(path.join(args.root,'art.svg'),'user-created-after-plan');await assert.rejects(createTask(plan),/changed after scope approval/);assert.equal(await fs.readFile(path.join(args.root,'art.svg'),'utf8'),'user-created-after-plan');
 await fs.symlink(args.storageRoot,path.join(args.root,'linked'));await assert.rejects(planTask({...args,scope:scope(['linked/output.svg'])}),/Noncanonical|Linked/);
}));
test('real pinned validator parses SVG references and refuses duplicates/external assets; it never certifies visuals',()=>fixture(async args=>{
 const file=path.join(args.root,'art.svg'),v=path.join(args.root,'validator.py');await fs.writeFile(v,await validatorFor(scope(['art.svg'])));
 const good='<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1024 768"><title>Bird</title><desc>Test only</desc><defs><path id="x" d="M0 0L5 5"/></defs><use href="#x"/></svg>';
 await fs.writeFile(file,good);let r=spawnSync('python3',['-I','-B',v],{cwd:args.root,encoding:'utf8'});assert.equal(r.status,0,r.stderr);assert.match(r.stdout,/VISUAL_QUALITY_NOT_CERTIFIED/);
 for(const content of [good.replace('href="#x"','href="#missing"'),good.replace('</svg>','<g id="x"/></svg>'),good.replace('</svg>','<image href="https://example.com/x"/></svg>')]){await fs.writeFile(file,content);r=spawnSync('python3',['-I','-B',v],{cwd:args.root});assert.notEqual(r.status,0);}
}));
test('reviewed sandbox commands run against candidate files; tool absence and source mutation are explicit failures',()=>fixture(async args=>{
 const v=path.join(args.root,'check.py');await fs.writeFile(path.join(args.root,'hello.py'),'print("ok")\n');await fs.writeFile(v,await validatorFor(scope(['hello.py'],[['python3','-B','hello.py']])));let r=spawnSync('python3',['-I','-B',v],{cwd:args.root,encoding:'utf8'});assert.equal(r.status,0,r.stderr);assert.match(r.stdout,/exit_code/);
 await fs.writeFile(path.join(args.root,'hello.py'),'from pathlib import Path\nPath("hello.py").write_text("changed")\n');r=spawnSync('python3',['-I','-B',v],{cwd:args.root,encoding:'utf8'});assert.notEqual(r.status,0);assert.match(r.stderr,/modified candidate/);
}));
test('actual TaskRunService carries selected source scope into native SDK transport without web animation requirements (mock RPC, no native claims)',()=>fixture(async args=>{
 const project=await createTask(await planTask({...args,scope:scope(['art.svg'])}));let task;const rpc=new EventEmitter();rpc.request=async(method,params)=>{if(method==='session.create')return{sessionId:ID};if(method==='task.submit'){task=params.task;return{status:'ACCEPTED'};}if(method==='task.wait')return{status:'BLOCKED',verified:false};return{};};
 const run=new TaskRunService(async(_,profile)=>{assert.equal(profile,project.profile);return rpc;},args.storageRoot);const session=await run.start({uri:{fsPath:args.root}},PROMPT,()=>{},{project});const result=await session.completion;assert.equal(result.verified,false);assert.deepEqual(task.requiredFiles,['art.svg']);assert.ok(task.goal.endsWith(PROMPT));assert.doesNotMatch(task.goal,/requires a full HTML5|requestAnimationFrame|ALL THREE/);assert.equal(run.activeSession({uri:{fsPath:args.root}}),null);
}));
function withHostModules(api,fn){const Module=require('node:module'),old=Module._load;delete require.cache[require.resolve('../src/task-entry.cjs')];try{Module._load=function(id,...args){if(id==='vscode')return api;if(id==='./native-env.cjs')return{nativeEnv:()=>({})};return old.call(this,id,...args);};return fn(require('../src/task-entry.cjs'));}finally{Module._load=old;}}
test('actual general TaskEntry progresses from read-only plan to approved native service; decline creates nothing (modeled RPC only)',()=>fixture(async args=>{
 const api={Uri:{file:p=>({fsPath:p})}},folder={uri:{fsPath:args.root},name:'selected'},events=[],state=new Map();
 const rpcFor=(planning)=>{const rpc=new EventEmitter();rpc.request=async(method,p)=>{if(method==='session.create')return{sessionId:ID};if(method==='task.submit'){if(planning)assert.match(p.task.goal,/TASK PLANNING/);else assert.ok(p.task.goal.endsWith(PROMPT));return{status:'ACCEPTED'};}if(method==='task.wait')return planning?response(scope()):{status:'BLOCKED',verified:false};return{};};return rpc;};
 const app={context:{globalStorageUri:{fsPath:args.storageRoot},globalState:{get:(k,d)=>state.get(k)||d,update:async(k,v)=>state.set(k,v)}},_agentRootForChat:async()=>args.agentRoot,_trackChatSession(){},_open:async(_,o)=>{assert.equal(o.generalTask,true);return rpcFor(false);}};
 app.chatRuns=new ChatRunService(async()=>rpcFor(true));
 const entry=withHostModules(api,({TaskEntry})=>new TaskEntry(app)),view={},provider={views:new Set([view]),pending:new Map([[view,{}]]),post:(_,m)=>events.push(m),requestNativeApproval:async()=>false};
 await entry.start(provider,view,{goal:PROMPT},folder);assert.equal((await fs.readdir(args.root)).length,0);assert.equal(events.some(e=>e.type==='task_planned'),true);
 provider.requestNativeApproval=async(_,scope)=>{assert.match(scope.target,/pelican-bicycle.svg/);assert.match(scope.validator,/Rust VGR/);return true;};
 await entry.start(provider,view,{goal:PROMPT},folder);assert.ok((await fs.readdir(args.root)).includes('pelican-bicycle.svg'));assert.ok(events.some(e=>e.type==='phase_started'&&e.phase==='native'));assert.equal(events.some(e=>e.type==='task_delivered'),false);assert.equal(entry.reserved.size,0);
}));
test('planning acceptance preserves attachment draft until execution is actually accepted',()=>{
 const {VoderChatViewProvider}=require('../src/chat-view.cjs'),provider=new TestChatViewProvider({}),view={webview:{postMessage(){}}};provider.views.add(view);let consumed=0;provider.consumeAcceptedReferences=()=>consumed++;
 provider.post(view,{type:'accepted',phase:'plan',sessionId:ID});assert.equal(consumed,0);provider.post(view,{type:'accepted',sessionId:ID});assert.equal(consumed,1);
});
test('host-declared content hashing accepts a full bounded text candidate, not just 4KiB',async()=>{
 const {HostToolRegistry}=await import('../../voder-tools/src/index.mjs'),{createHash}=require('node:crypto');const r=new HostToolRegistry();r.register({name:'voder_ext_hash_test',description:'test pure content hashing',parameters:{type:'object',properties:{content:{type:'string',maxLength:65536}},required:['content'],additionalProperties:false},execute:async({content})=>({sha256:createHash('sha256').update(content).digest('hex')})});
 const content='a'.repeat(8192),receipt=await r.makeTools()[0].execute({content});assert.equal(receipt.data.sha256,createHash('sha256').update(content).digest('hex'));await assert.rejects(r.makeTools()[0].execute({content:'a'.repeat(65537)}));
});
test('candidate admission parses JavaScript before sandbox and returns the exact delegated receipt; single-file scope is supported',async()=>{
 const {TaskCandidateVerifier}=await import('../src/task-control.mjs');let calls=0;const receipt={status:'PASS',backend:'TEST_DOUBLE_ONLY'},base={getBudget:()=>({used:0}),verifyCandidate:async()=>{calls++;return receipt;}};
 const verifier=new TaskCandidateVerifier(base,{targets:['game.js'],node:process.execPath});
 const bad=await verifier.verifyCandidate({relative_path:'game.js',new_content:'const broken = ;'});assert.equal(bad.status,'FAIL');assert.equal(calls,0);
 assert.equal(await verifier.verifyCandidate({relative_path:'game.js',new_content:'const valid = 1;'}),receipt);assert.equal(calls,1);
 await assert.rejects(verifier.verifyCandidate({relative_path:'other.js',new_content:'x'}),e=>e.code==='TASK_CANDIDATE_SCOPE');
});
