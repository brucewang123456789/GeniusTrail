'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs/promises'),os=require('node:os'),path=require('node:path'),Module=require('node:module'),{execFileSync}=require('node:child_process'),{EventEmitter}=require('node:events');
const {parseMultiGoal,operations,validateExtended}=require('../src/multi-workspace.cjs');
const {planProject,createProject,validateProject,digest}=require('../src/project-workspace.cjs');
const {readScope,inspectLanguages,inspectDependencies}=require('../src/workspace-tools.cjs');
const {validateObservation,createObserver}=require('../src/browser-observer.cjs');
const {MultiRunService}=require('../src/multi-run.cjs');
const ID='12345678-1234-4234-8234-123456789abc',GOAL='Create a new multi-file web project\nFiles: index.html, app.js, style.css, src/helper.js, data/settings.json, utils.py\nCreate a code-drawn animation with local modules.';
const GOOD={'app.js':"import {value} from './src/helper.js';\nconst loop=()=>requestAnimationFrame(loop);loop();\n",'index.html':'<!DOCTYPE html><html><head><title>Modular scene</title><link rel="stylesheet" href="style.css"></head><body><canvas></canvas><script type="module" src="app.js"></script></body></html>','style.css':'html,body{background:#080808;color:white}canvas{width:100%;height:100vh}','src/helper.js':'export const value = 1;\n','data/settings.json':'{"speed":1}','utils.py':'def speed(value):\n    return value\n'};
async function fixture(fn){const host=await fs.mkdtemp(path.join(os.tmpdir(),'voder-toolkit-')),agentRoot=path.join(host,'voder-agent'),storageRoot=path.join(host,'storage');try{
 for(const dir of [agentRoot,storageRoot,path.join(host,'.voder-build/release'),path.join(host,'packages/voder-sandbox-modal/src'),path.join(host,'.voder-tools/node/bin')])await fs.mkdir(dir,{recursive:true,mode:0o700});
 for(const f of ['.voder-build/release/voder-gateway','.voder-build/release/voderd','packages/voder-sandbox-modal/src/cli.ts'])await fs.writeFile(path.join(host,f),'offline fixture\n',{mode:0o600});await fs.writeFile(path.join(host,'.voder-tools/node/bin/node'),'#!/bin/sh\nexec '+JSON.stringify(process.execPath)+' "$@"\n',{mode:0o700});
 const marker=path.join(agentRoot,'original.txt');await fs.writeFile(marker,'preserve original');await fn({host,agentRoot,storageRoot});assert.equal(await fs.readFile(marker,'utf8'),'preserve original');
 }finally{await fs.rm(host,{recursive:true,force:true});}}
async function multi(args){const targets=parseMultiGoal(GOAL),ops=operations(targets),p=await ops.createProject(await ops.planProject(args));return p;}
function loadWithVscode(name,api){const old=Module._load;delete require.cache[require.resolve('../src/'+name)];try{Module._load=function(id,...a){return id==='vscode'?api:old.call(this,id,...a);};return require('../src/'+name);}finally{Module._load=old;}}
function observation(token,extra={}){return {type:'runtime_observation',token,frames:100,clicks:1,samples:2,changes:1,elapsed:1000,width:1200,height:700,errors:[],...extra};}
test('multi-file entry requires explicit intent and exact bounded safe paths; normal web/Python entry is untouched',()=>{
 assert.deepEqual(parseMultiGoal(GOAL),Object.keys(GOOD).sort());assert.equal(parseMultiGoal('Create a new web project with a parrot'),null);assert.equal(parseMultiGoal('修复 calc.py，使 test_calc.py 通过'),null);
 for(const files of ['app.js,index.html,style.css,../outside.py','app.js,index.html,style.css,verify_web.py','app.js,index.html,style.css,app.js','app.js,index.html,style.css,src/helpers.js,a.js/x.json,a.js'])assert.throws(()=>parseMultiGoal('Create a new multi-file web project\nFiles: '+files));
});
test('new scope and validator do not invalidate original three-file project records',()=>fixture(async args=>{
 const old=await createProject(await planProject(args)),before=await fs.readFile(path.join(old.root,'verify_web.py')),created=await multi(args),cfg=JSON.parse(await fs.readFile(created.profile,'utf8'));
 assert.deepEqual(cfg.targetPaths,Object.keys(GOOD).sort());assert.equal((await validateExtended(created.recordPath,args.storageRoot)).root,created.root);assert.equal((await validateProject(old.recordPath,args.storageRoot)).root,old.root);assert.deepEqual(await fs.readFile(path.join(old.root,'verify_web.py')),before);
 assert.notEqual(old.root,created.root);await assert.rejects(validateProject(created.recordPath,args.storageRoot));
}));
test('REAL Rust preflight accepts nested multi-file scope and leaves action store empty',()=>fixture(async args=>{
 const p=await multi(args),cfg=JSON.parse(await fs.readFile(p.profile,'utf8'));assert.ok(process.env.VODER_NATIVE_TEST_GATEWAY);const r=JSON.parse(execFileSync(process.env.VODER_NATIVE_TEST_GATEWAY,['--check-policy',cfg.policyFile],{encoding:'utf8'}));assert.equal(r.status,'PASS');assert.deepEqual(r.targets,Object.keys(GOOD).sort());assert.deepEqual(await fs.readdir(path.join(cfg.attemptStore,'attempts')),[]);
}));
test('independent extended validator parses auxiliary Python/JSON and enforces total review budget',()=>fixture(async args=>{
 const p=await multi(args);for(const[n,c]of Object.entries(GOOD))await fs.writeFile(path.join(p.root,n),c);
 const check=()=>{try{return execFileSync('/usr/bin/python3',['-I','-B','verify_web.py'],{cwd:p.root,stdio:'pipe'}).toString();}catch(e){return e;}};assert.match(check(),/PASS:/);
 await fs.writeFile(path.join(p.root,'data/settings.json'),'{broken');assert.ok(check()instanceof Error);await fs.writeFile(path.join(p.root,'data/settings.json'),'{}');await fs.writeFile(path.join(p.root,'utils.py'),'def broken(:\n');assert.ok(check()instanceof Error);
}));
test('multi-file model submission carries exact approved list once and keeps unknown outcomes uncertain',()=>fixture(async args=>{
 const p=await multi(args),rpc=new EventEmitter(),calls=[],updates=[];rpc.request=async(name,value)=>{calls.push([name,value]);if(name==='session.create')return{sessionId:ID};if(name==='session.subscribe')return{};if(name==='task.submit')throw Object.assign(Error('uncertain'),{code:'RPC_TIMEOUT'});throw Error(name);};
 const svc=new MultiRunService(async()=>rpc,args.storageRoot);await assert.rejects(svc.start({uri:{fsPath:p.root}},GOAL,u=>updates.push(u),{project:p}),/uncertain/);const submits=calls.filter(x=>x[0]==='task.submit');assert.equal(submits.length,1);assert.deepEqual(submits[0][1].task.requiredFiles,p.targets);assert.equal(submits[0][1].task.maxActions,1);assert.equal(updates.at(-1).kind,'uncertain');
}));
test('JavaScript, Python and JSON parsing never executes source; unsupported build languages are explicitly NOT_RUN',()=>fixture(async({host})=>{
 const files={'danger.js':"require('node:fs').writeFileSync('/tmp/voder-never-execute','bad');\n",'danger.py':"raise RuntimeError('MUST_NOT_EXECUTE')\n",'bad.json':'{invalid','component.ts':'const x: number = 1;'};for(const[n,c]of Object.entries(files))await fs.writeFile(path.join(host,n),c);
 const report=await inspectLanguages(host,Object.keys(files));assert.deepEqual(report.checks.map(c=>c.status),['PASS','PASS','FAIL','NOT_RUN']);assert.equal(report.executed_user_code,false);
 await assert.rejects(readScope(host,['../bad.py']));await fs.symlink(path.join(host,'danger.py'),path.join(host,'link.py'));await assert.rejects(readScope(host,['link.py']),/SYMLINK/);
}));
test('parsers recheck source identity after observation, not just before invoking the checker',()=>fixture(async({host})=>{
 await fs.writeFile(path.join(host,'x.js'),'const x=1;');await assert.rejects(inspectLanguages(host,['x.js'],{run:async()=>{await fs.writeFile(path.join(host,'x.js'),'const x=2;');return{status:'PASS'};}}),/SOURCE_CHANGED/);
}));
test('dependency inventory does not install dependencies or run malicious lifecycle scripts',()=>fixture(async({host})=>{
 await fs.writeFile(path.join(host,'package.json'),JSON.stringify({dependencies:{fake:'1.0.0'},scripts:{postinstall:'rm -rf /THIS_MUST_NEVER_RUN'}}));await fs.writeFile(path.join(host,'package-lock.json'),'{}');const r=inspectDependencies(await readScope(host,['package.json','package-lock.json']));assert.equal(r.ecosystems[0].installation,'NOT_RUN');assert.equal(r.ecosystems[0].lockfile,'package-lock.json');assert.equal(r.services_started,0);await fs.writeFile(path.join(host,'credentials.json'),'{}');await assert.rejects(readScope(host,['credentials.json']),/UNSAFE_FILE/);
}));
test('browser messages are bounded non-authoritative diagnostics, never a command or native proof',()=>{
 let receive,rows=[],disposed=false;const panel={webview:{onDidReceiveMessage:f=>{receive=f;return{dispose:()=>disposed=true};}}};const o=createObserver(panel,r=>rows.push(r)),token=JSON.parse(o.source.match(/token=("[^"]+")/)[1]);
 receive(observation('wrong'));receive({...observation(token),command:'task.submit'});receive(observation(token,{errors:['x'.repeat(181)]}));assert.equal(rows.length,0);
 for(let i=0;i<20;i++)receive(observation(token,{elapsed:i*1000}));assert.equal(rows.length,12);assert.equal(rows[0].visual_acceptance,'NOT_ESTABLISHED');assert.equal(rows[0].control_correctness,'NOT_ESTABLISHED');assert.match(rows[0].authority,/UNTRUSTED/);o.dispose();assert.equal(disposed,true);
});
test('report viewer escapes model-controlled content and executes no scripts',()=>{
 const {reportHtml}=loadWithVscode('toolkit-controller.cjs',{}),html=reportHtml({text:'<script>alert(1)</script>',result:'PASS'});assert.doesNotMatch(html,/<script>/);assert.match(html,/&lt;script&gt;/);assert.match(html,/default-src 'none'/);
});
test('restoring saved sessions uses only draft status/events with no native profile and no task replay',()=>fixture(async args=>{
 const p=await createProject(await planProject(args)),saved={'voder.webProjects':{[ID]:p.recordPath}},calls=[];const api={Uri:{file:f=>({fsPath:f})},window:{showQuickPick:async rows=>rows[0]}};const {ToolkitController}=loadWithVscode('toolkit-controller.cjs',api);
 const app={trusted:()=>true,chatSessionFolder:new Map(),context:{globalStorageUri:{fsPath:args.storageRoot},globalState:{get:(k,d)=>saved[k]??d}},_open:async(folder,opts)=>{assert.equal(folder.uri.fsPath,p.root);assert.equal(opts.mode,'draft');assert.equal(opts.gateway,'');return{request:async(name)=>{calls.push(name);return name==='session.status'?{status:'DELIVERED',result:{status:'DELIVERED'}}:[];}};}};const controller=new ToolkitController(app);controller.show=r=>r;await controller.restore();assert.deepEqual(calls,['session.status','session.events']);assert.equal(app.chatSessionFolder.get(ID).uri.fsPath,p.root);
}));
test('isolated Python runner pins command/image, verifies modeled receipt and never writes original files',()=>fixture(async args=>{
 const {runPythonChecks}=require('../src/sandbox-checks.cjs');await fs.writeFile(path.join(args.host,'calc.py'),'def add(a,b): return a+b\n');await fs.writeFile(path.join(args.host,'test_calc.py'),'import unittest\nfrom calc import add\nclass Check(unittest.TestCase):\n def test_add(self): self.assertEqual(add(1,2),3)\n');
 const keys=['VODER_MODEL_ID','VODER_MODEL_API_KEY','MODAL_TOKEN_ID','MODAL_TOKEN_SECRET','VODER_MODAL_APP'],prior={};for(const k of keys){prior[k]=process.env[k];process.env[k]='offline-test-only';}
 try{const {IMAGE}=require('../src/native-profile.cjs');let calls=0;const r=await runPythonChecks({root:args.host,files:['calc.py','test_calc.py'],...args,execute:async(node,argv,cwd,env)=>{
  calls++;const [request,receipt,out,err]=argv.slice(2),q=JSON.parse(await fs.readFile(request,'utf8'));assert.equal(q.program,'/usr/bin/env');assert.deepEqual(q.args,['/usr/local/bin/python3.10','-I','-B','-m','unittest','discover','-s','.','-p','test*.py']);assert.equal(env.VODER_MODAL_IMAGE,IMAGE);assert.notEqual(q.workspace,args.host);
  const stdout=Buffer.from(''),stderr=Buffer.from('Ran 1 test in 0.001s\nOK\n');await fs.writeFile(out,stdout);await fs.writeFile(err,stderr);await fs.writeFile(receipt,JSON.stringify({protocol:'voder-sandbox-receipt-v1',provider:'modal',status:'EXECUTED',runId:q.runId,actionId:q.actionId,image:IMAGE,stdoutSha256:digest(stdout),stderrSha256:digest(stderr),exitCode:0,timeout:false,outputLimited:false}));
 }});assert.equal(calls,1);assert.equal(r.status,'PASS');assert.equal(r.source_writes,0);assert.equal(r.native_actions,0);assert.equal(await fs.readFile(path.join(args.host,'calc.py'),'utf8'),'def add(a,b): return a+b\n');}
 finally{for(const[k,v]of Object.entries(prior)){if(v===undefined)delete process.env[k];else process.env[k]=v;}}
}));
test('multi-file preview stages exact nested snapshot members and permits only local module resources',()=>fixture(async args=>{
 const p=await multi(args);for(const[n,c]of Object.entries(GOOD))await fs.writeFile(path.join(p.root,n),c);
 let panel,disposed;const uri=f=>({fsPath:f,toString:()=>f}),api={Uri:{file:uri},ViewColumn:{Beside:2},window:{createWebviewPanel:(kind,title,col,opts)=>{panel={opts,webview:{cspSource:'vscode-webview:',asWebviewUri:u=>({toString:()=>`vscode-webview://local/${path.relative(args.storageRoot,u.fsPath)}`})},onDidDispose:fn=>disposed=fn};return panel;}}};
 const {MultiEntry}=loadWithVscode('multi-entry.cjs',api),app={context:{globalStorageUri:uri(args.storageRoot),globalState:{get:(k,d)=>d}}},entry=new MultiEntry(app);
 entry.verifiedFiles=async()=>({project:p,files:Object.keys(GOOD).sort().map(n=>({relative_path:n,after:GOOD[n]}))});await entry.preview(ID);const dir=panel.opts.localResourceRoots[0].fsPath;assert.equal(await fs.readFile(path.join(dir,'src/helper.js'),'utf8'),GOOD['src/helper.js']);assert.match(panel.webview.html,/type="module"/);assert.match(panel.webview.html,/connect-src vscode-webview:/);assert.doesNotMatch(panel.webview.html,/unsafe-eval/);assert.equal(panel.webview.onDidReceiveMessage,undefined);disposed();
}));
test('runtime observation bridge stages verified bytes only and cannot dispatch host actions',()=>fixture(async args=>{
 const p=await multi(args);for(const[n,c]of Object.entries(GOOD))await fs.writeFile(path.join(p.root,n),c);
 const panels=[],uri=f=>({fsPath:f,toString:()=>f});let receive;
 const api={Uri:{file:uri},ViewColumn:{Beside:2},window:{createWebviewPanel:(kind,title,col,opts)=>{const panel={opts,webview:{cspSource:'vscode-webview:',asWebviewUri:u=>({toString:()=>`vscode-webview://local/${path.basename(u.fsPath)}`}),onDidReceiveMessage:f=>{receive=f;return{dispose(){}};}},onDidDispose:f=>panel.close=f};panels.push(panel);return panel;}}};
 const {ToolkitController}=loadWithVscode('toolkit-controller.cjs',api),logs=[],app={trusted:()=>true,context:{globalStorageUri:{fsPath:args.storageRoot},globalState:{get:()=>({})}},output:{appendLine:x=>logs.push(x),show(){}},webProjects:{verifiedFiles:async()=>({project:p,files:Object.keys(GOOD).sort().map(n=>({relative_path:n,after:GOOD[n]}))})}};
 const controller=new ToolkitController(app);await controller.runtime(ID);const panel=panels[0];assert.notEqual(panel.opts.localResourceRoots[0].fsPath,p.root);receive({type:'task.submit',goal:'malicious'});assert.equal(logs.length,0);assert.match(panel.webview.html,/UNTRUSTED|runtime_observation/);panel.close();assert.equal(controller.panels.size,0);
}));
test('actual SDK verifies the extended candidate in a clone through explicitly MODELED offline transport',()=>fixture(async args=>{
 const p=await multi(args),cfg=JSON.parse(await fs.readFile(p.profile,'utf8')),transport=path.join(args.host,'offline-transport.cjs');
 await fs.writeFile(transport,`// TEST DOUBLE only, not a real Modal receipt.\nconst fs=require('node:fs'),{spawnSync}=require('node:child_process'),{createHash}=require('node:crypto');const [request,receipt,out,err]=process.argv.slice(2);const q=JSON.parse(fs.readFileSync(request)),r=spawnSync('/usr/bin/python3',['-I','-B','verify_web.py'],{cwd:q.workspace});fs.writeFileSync(out,r.stdout);fs.writeFileSync(err,r.stderr);const h=b=>createHash('sha256').update(b).digest('hex');fs.writeFileSync(receipt,JSON.stringify({protocol:'voder-sandbox-receipt-v1',provider:'modal',status:'EXECUTED',runId:q.runId,actionId:q.actionId,image:process.env.VODER_MODAL_IMAGE,stdoutSha256:h(r.stdout),stderrSha256:h(r.stderr),exitCode:r.status,timeout:false,outputLimited:false}));\n`);
 const {pathToFileURL}=require('node:url'),{HostCandidateVerifier}=await import(pathToFileURL(path.join(process.env.VODER_AGENT_TEST_ROOT,'packages/voder-sdk/src/candidate-verifier.mjs')).href),verifier=new HostCandidateVerifier({cwd:p.root,nativeGateway:{...cfg,nodeBinary:process.execPath,modalCli:transport}});
 const candidate={patches:await Promise.all(p.targets.map(async relative_path=>({relative_path,old_sha256:digest(await fs.readFile(path.join(p.root,relative_path))),new_content:GOOD[relative_path]})))};
 const receipt=await verifier.verifyCandidate(candidate,{sessionId:ID});assert.equal(receipt.status,'PASS');assert.match(receipt.diagnostic,/visuals NOT tested/);assert.notEqual(await fs.readFile(path.join(p.root,'app.js'),'utf8'),GOOD['app.js']);
}));
test('new candidate syntax gate blocks invalid JavaScript before sandbox and cannot synthesize a passing receipt',async()=>{
 const {pathToFileURL}=require('node:url'),{SyntaxThenSandboxVerifier}=await import(pathToFileURL(path.resolve(__dirname,'../src/tool-control.mjs')).href);let calls=0;const native={status:'PASS',backend:'real_delegate_marker'};
 const gate=new SyntaxThenSandboxVerifier({verifyCandidate:async()=>{calls++;return native;}},{targets:['app.js'],node:process.execPath});
 const invalid=await gate.verifyCandidate({patches:[{relative_path:'app.js',new_content:'const broken = ;'}]},{sessionId:ID});assert.equal(invalid.status,'FAIL');assert.equal(calls,0);
 assert.equal(await gate.verifyCandidate({patches:[{relative_path:'app.js',new_content:'const valid = 1;'}]},{sessionId:ID}),native);assert.equal(calls,1);
 await assert.rejects(gate.verifyCandidate({patches:[{relative_path:'outside.js',new_content:'const valid = 1;'}]},{sessionId:ID}),{code:'TOOLKIT_CANDIDATE_SCOPE'});
 // Local rejections never consume the separate three-attempt sandbox allowance.
 for(let i=0;i<2;i++)assert.equal(await gate.verifyCandidate({patches:[{relative_path:'app.js',new_content:'const valid = '+(i+2)+';'}]},{sessionId:ID}),native);
 await assert.rejects(gate.verifyCandidate({patches:[{relative_path:'app.js',new_content:'const valid = 4;'}]},{sessionId:ID}),{code:'TOOLKIT_CANDIDATE_BUDGET'});assert.equal(calls,3);
});
test('opt-in RPC factory installs genuine SDK read-only registry and leaves original CLI entry untouched',()=>fixture(async args=>{
 const binary=process.env.VODER_NATIVE_TEST_GATEWAY;await fs.copyFile(binary,path.join(args.host,'.voder-build/release/voder-gateway'));await fs.chmod(path.join(args.host,'.voder-build/release/voder-gateway'),0o700);const p=await multi(args);
 const {pathToFileURL}=require('node:url'),{makeToolkitControl}=await import(pathToFileURL(path.resolve(__dirname,'../src/tool-control.mjs')).href);
 const control=await makeToolkitControl({cwd:p.root,sessionDir:path.join(args.storageRoot,'rpc-journal'),gate:p.profile,agentRoot:process.env.VODER_AGENT_TEST_ROOT});
 try{const names=control.sdk.toolRegistry.list().map(t=>t.name);assert.deepEqual(names,['voder_ext_project_syntax','voder_ext_project_dependencies','voder_ext_project_capabilities','voder_ext_memory_dialogue']);const tools=control.sdk.toolRegistry.makeTools(),r=await tools[2].execute({});assert.equal(r.data.arbitrary_shell,false);assert.equal(r.data.package_installation,false);assert.deepEqual(r.data.scope,p.targets);assert.equal((await control.journal.list()).length,0);}
 finally{await control.shutdown();}
}));
