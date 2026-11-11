'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs/promises'),os=require('node:os'),path=require('node:path'),{execFileSync,spawnSync}=require('node:child_process');
const {planTask,createTask,resolveTaskScope}=require('../src/task-workspace.cjs');
const scope=(files,checks=[])=>({action:'execute',answer:'create requested artifacts',files,checks});
async function fixture(fn){const host=await fs.mkdtemp(path.join(os.tmpdir(),'voder-git-boundary-')),root=path.join(host,'voder-projects'),inner=path.join(root,'web-id'),agentRoot=path.join(host,'voder-agent'),storageRoot=path.join(host,'storage');try{
 for(const d of [inner,agentRoot,storageRoot,path.join(host,'.voder-build/release'),path.join(host,'packages/voder-sandbox-modal/src'),path.join(host,'.voder-tools/node/bin')])await fs.mkdir(d,{recursive:true,mode:0o700});
 for(const f of ['.voder-build/release/voder-gateway','.voder-build/release/voderd','packages/voder-sandbox-modal/src/cli.ts'])await fs.writeFile(path.join(host,f),'TEST INFRASTRUCTURE ONLY\n',{mode:0o600});
 await fs.writeFile(path.join(host,'.voder-tools/node/bin/node'),'#!/bin/sh\nexec '+JSON.stringify(process.execPath)+' "$@"\n',{mode:0o700});
 const git=(dir,...args)=>execFileSync('git',['-C',dir,...args],{stdio:'pipe'}).toString();git(inner,'init','-q');await fs.writeFile(path.join(inner,'app.js'),'const oldScene = 1;\n');git(inner,'add','.');git(inner,'-c','user.name=test','-c','user.email=test@invalid','commit','-qm','old project');await fn({host,root,inner,agentRoot,storageRoot,git});
 }finally{await fs.rm(host,{recursive:true,force:true});}}
test('reproduces reported pathspec failure with real Git: outer add silently omits nested-repository files',()=>fixture(async({root,inner,git})=>{
 git(root,'init','-q');await fs.writeFile(path.join(inner,'pelican-bicycle.svg'),'placeholder');await fs.writeFile(path.join(root,'.validator.py'),'pass');git(root,'add','--','web-id/pelican-bicycle.svg','.validator.py');assert.ok(!git(root,'ls-files').includes('pelican-bicycle.svg'));
 const result=spawnSync('git',['-C',root,'-c','user.name=test','-c','user.email=test@invalid','commit','--only','-qm','baseline','--','web-id/pelican-bicycle.svg','.validator.py'],{encoding:'utf8'});assert.notEqual(result.status,0);assert.match(result.stderr,/pathspec.*did not match/);
}));
test('actual plan/create normalizes to nested owner before approval and commits exact selected preimages there',()=>fixture(async args=>{
 args.git(args.root,'init','-q');await fs.writeFile(path.join(args.root,'keep.txt'),'unrelated staged');args.git(args.root,'add','keep.txt');
 const before=args.git(args.inner,'show','HEAD:app.js');
 const plan=await planTask({...args,scope:scope(['web-id/pelican-bicycle.svg','web-id/verification.md'])});assert.equal(plan.root,args.inner);assert.deepEqual(plan.scope.files,['pelican-bicycle.svg','verification.md']);assert.equal((await fs.readdir(args.inner)).includes('pelican-bicycle.svg'),false);
 const project=await createTask(plan);assert.equal(project.root,args.inner);assert.equal(args.git(args.inner,'show','HEAD:pelican-bicycle.svg'),'Voder task preimage; awaiting verified candidate.\n');assert.equal(args.git(args.inner,'show','HEAD:app.js'),before);assert.equal(args.git(args.root,'diff','--cached','--name-only').trim(),'keep.txt');
}));
test('literal check paths are rebased; opaque evaluated commands and cross-repository writes fail before preimages',()=>fixture(async args=>{
 const r=await resolveTaskScope(args.root,scope(['web-id/source.py'],[['python3','-B','web-id/source.py']]));assert.equal(r.root,args.inner);assert.deepEqual(r.scope.checks,[['python3','-B','source.py']]);
 await assert.rejects(planTask({...args,scope:scope(['web-id/new.py'],[['python3','-c','print(1)']])}),/opaque evaluated/);
 await assert.rejects(planTask({...args,scope:scope(['web-id/a.svg','outer.md'])}),/cross independent Git/);assert.equal((await fs.readdir(args.inner)).includes('new.py'),false);
}));
test('actual TaskEntry approval, pending folder and native service use normalized repository consistently (mock transport only)',()=>fixture(async args=>{
 const Module=require('node:module'),{EventEmitter}=require('node:events'),{ChatRunService}=require('../src/chat-core.cjs');const old=Module._load;delete require.cache[require.resolve('../src/task-entry.cjs')];let TaskEntry;
 try{Module._load=function(id,...rest){if(id==='vscode')return{Uri:{file:p=>({fsPath:p})}};if(id==='./native-env.cjs')return{nativeEnv:()=>({})};return old.call(this,id,...rest);};({TaskEntry}=require('../src/task-entry.cjs'));}finally{Module._load=old;}
 const id='12345678-1234-4234-8234-123456789abc',states=new Map(),events=[],view={},folder={uri:{fsPath:args.root},name:'parent'};let nativeRoot,approval;
 const rpc=planning=>{const r=new EventEmitter();r.request=async(method)=>method==='session.create'?{sessionId:id}:method==='task.submit'?{status:'ACCEPTED'}:method==='task.wait'?planning?{status:'DRAFT_UNVERIFIED',verified:false,draft:JSON.stringify(scope(['web-id/art.svg']))}:{status:'BLOCKED',verified:false}:{};return r;};
 const app={context:{globalStorageUri:{fsPath:args.storageRoot},globalState:{get:(k,d)=>states.get(k)||d,update:async(k,v)=>states.set(k,v)}},_agentRootForChat:async()=>args.agentRoot,_trackChatSession(){},_open:async f=>{nativeRoot=f.uri.fsPath;return rpc(false);}};app.chatRuns=new ChatRunService(async()=>rpc(true));
 const entry=new TaskEntry(app),provider={views:new Set([view]),pending:new Map([[view,{}]]),post:(_,m)=>events.push(m),requestNativeApproval:async(_,s)=>{approval=s;return true;}};
 await entry.start(provider,view,{goal:'Create editable art.svg inside the existing project.'},folder);
 assert.ok(approval.target.startsWith(args.inner+'\nart.svg'));assert.equal(nativeRoot,args.inner);assert.equal(provider.pending.get(view).folder.uri.fsPath,args.inner);assert.ok(events.some(e=>e.type==='workspace_selected'&&e.path===args.inner));assert.equal(entry.reserved.size,0);
}));
