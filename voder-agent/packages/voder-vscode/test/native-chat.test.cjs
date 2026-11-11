'use strict';
const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs/promises');
const os=require('node:os');
const path=require('node:path');
const crypto=require('node:crypto');
const {execFileSync}=require('node:child_process');
const {createProfile,validateProfile,argumentPin,IMAGE}=require('../src/native-profile.cjs');
const {nativeEnv}=require('../src/native-env.cjs');
const {ChatRunService}=require('../src/chat-core.cjs');
function git(...args){execFileSync('/usr/bin/git',args,{stdio:'pipe',timeout:5000});}
async function fixture(){
 const tmp=await fs.mkdtemp(path.join(os.tmpdir(),'voder-native-ui-')),repo=path.join(tmp,'repo');
 await fs.mkdir(repo);await fs.writeFile(path.join(repo,'target.py'),'def add(a,b):\n    return a+b\n');await fs.writeFile(path.join(repo,'test_target.py'),'from target import add\nassert add(2,2)==4\n');
 git('-C',repo,'init','-q');git('-C',repo,'add','target.py','test_target.py');git('-C',repo,'-c','user.name=Voder Test','-c','user.email=bot@example.test','commit','-qm','initial');
 const agent=path.join(tmp,'voder-agent'),root=tmp;
 for(const p of ['.voder-build/release/voder-gateway','.voder-build/release/voderd','.voder-tools/node/bin/node','packages/voder-sandbox-modal/src/cli.ts']){
  const full=path.join(root,p);await fs.mkdir(path.dirname(full),{recursive:true});await fs.writeFile(full,'fixture');if(p.endsWith('/node'))await fs.chmod(full,0o755);
 }
 const folder={uri:{fsPath:repo}};
 return {tmp,repo,agent,folder,finish:()=>fs.rm(tmp,{recursive:true,force:true})};
}
test('native policy is pinned to committed existing workspace file and independent validator',async()=>{
 const f=await fixture();try{
  const profile=await createProfile({folder:f.folder,agentRoot:f.agent,storageRoot:f.tmp,targetFile:path.join(f.repo,'target.py'),testFile:path.join(f.repo,'test_target.py')});
  assert.equal(profile.target,'target.py');const cfg=JSON.parse(await fs.readFile(profile.path));const policy=JSON.parse(await fs.readFile(cfg.policyFile));
  assert.equal(policy.validator_sha256,argumentPin(policy.validator_program,policy.validator_args));
  assert.equal((await fs.stat(profile.path)).mode&0o077,0);
  assert.equal((await fs.stat(cfg.policyFile)).mode&0o077,0);
  assert.equal((await validateProfile(profile.path,f.folder)).target,'target.py');
  await fs.chmod(profile.path,0o644);await assert.rejects(validateProfile(profile.path,f.folder),e=>e.code==='CHAT_PROFILE_NOT_PRIVATE');
 }finally{await f.finish();}
});
test('new file and dirty source are rejected before any native execution',async()=>{
 const f=await fixture();try{
  await fs.writeFile(path.join(f.repo,'brand_new.py'),'pass\n');
  await assert.rejects(createProfile({folder:f.folder,agentRoot:f.agent,storageRoot:f.tmp,targetFile:path.join(f.repo,'brand_new.py'),testFile:path.join(f.repo,'test_target.py')}),e=>e.code==='CHAT_FILE_NOT_TRACKED');
  await fs.writeFile(path.join(f.repo,'target.py'),'modified');
  await assert.rejects(createProfile({folder:f.folder,agentRoot:f.agent,storageRoot:f.tmp,targetFile:path.join(f.repo,'target.py'),testFile:path.join(f.repo,'test_target.py')}),e=>e.code==='CHAT_FILE_NOT_TRACKED');
 }finally{await f.finish();}
});
test('native profile does not authorize a different workspace or target',async()=>{
 const f=await fixture();try{
  const profile=await createProfile({folder:f.folder,agentRoot:f.agent,storageRoot:f.tmp,targetFile:path.join(f.repo,'target.py'),testFile:path.join(f.repo,'test_target.py')});
  const other={uri:{fsPath:f.tmp}};
  await assert.rejects(validateProfile(profile.path,other),e=>e.code==='CHAT_PROFILE_POLICY_MISMATCH');
  const cfg=JSON.parse(await fs.readFile(profile.path));cfg.targetPath='test_target.py';await fs.writeFile(profile.path,JSON.stringify(cfg),{mode:0o600});
  await assert.rejects(validateProfile(profile.path,f.folder),e=>e.code==='CHAT_PROFILE_POLICY_MISMATCH');
 }finally{await f.finish();}
});
test('native RPC receives bounded scope and cannot silently downgrade into draft',async()=>{
 const {EventEmitter}=require('node:events');
 const f=await fixture();try{
  const rows=[];class Mock extends EventEmitter{async request(method,p){rows.push({method,p});if(method==='session.create')return{sessionId:'ae4d4a1c-2248-4b4a-99d2-3fa5784fc005'};if(method==='task.submit')return{status:'ACCEPTED'};if(method==='task.wait')return{status:'BLOCKED',verified:false};return {};}}
  let seen;const svc=new ChatRunService(async(folder,profile)=>{seen=profile;return new Mock();});
  const job=await svc.start(f.folder,'Repair addition',()=>{},{nativeProfile:'/private/gate.json',operatorTarget:'target.py',requiredFiles:['target.py']});await job.completion;
  assert.equal(seen,'/private/gate.json');const submit=rows.find(x=>x.method==='task.submit');
  assert.deepEqual(submit.p.task.requiredFiles,['target.py']);assert.match(submit.p.task.goal,/OPERATOR-PINNED SCOPE/);assert.match(submit.p.task.goal,/No new files/);
  assert.equal(submit.p.task.maxActions,1);
  await assert.rejects(svc.start(f.folder,'Bad',()=>{},{nativeProfile:'/private/gate.json',operatorTarget:'other.py',requiredFiles:['target.py']}),e=>e.code==='CHAT_NATIVE_SCOPE_REQUIRED');
 }finally{await f.finish();}
});
test('native env checks missing keys in isolation and honors the user private model config',()=>{
 const env=nativeEnv({VODER_MODEL_ID:'model',VODER_MODEL_API_KEY:'fake-secret',MODAL_TOKEN_ID:'token',MODAL_TOKEN_SECRET:'fake-value',VODER_MODAL_APP:'app'});
 assert.equal(env.VODER_MODAL_IMAGE,IMAGE);assert.equal(env.VODER_SANDBOX_BACKEND,'modal');assert.equal(env.MODAL_TOKEN_SECRET,'fake-value');
 assert.equal(env.VODER_ALLOW_MODEL_API,'YES');assert.equal(env.VODER_ALLOW_MODAL_CHARGES,'YES');
 // The user's REAL ~/.config/voder/model-provider.json may legitimately supply the missing
 // key. Isolate this test so its assertion does not depend on any host credentials.
 const os=require('node:os');const fsSync=require('node:fs');
 const isolated=fsSync.mkdtempSync(path.join(os.tmpdir(),'voder-chat-credential-test-'));
 const savedHome=os.homedir;
 try{
  os.homedir=()=>isolated;
  assert.throws(()=>nativeEnv({VODER_MODEL_ID:'x'}),e=>e.code==='CHAT_MODEL_NOT_CONFIGURED');
  const cfgDir=path.join(isolated,'.config','voder');fsSync.mkdirSync(cfgDir,{recursive:true,mode:0o700});
  fsSync.writeFileSync(path.join(cfgDir,'model-provider.json'),JSON.stringify({VODER_MODEL_ID:'private-model',VODER_MODEL_API_KEY:'fixture-only-key'}),{mode:0o600});
  const configured=nativeEnv({VODER_MODEL_ID:'x',MODAL_TOKEN_ID:'t',MODAL_TOKEN_SECRET:'s',VODER_MODAL_APP:'a'});
  assert.equal(configured.VODER_MODEL_ID,'private-model');
  assert.equal(configured.VODER_MODEL_API_KEY,'fixture-only-key');
 }finally{os.homedir=savedHome;fsSync.rmSync(isolated,{recursive:true,force:true});}
});
test('explicit new-file intent is rejected before RPC, with no paid model call',async()=>{
 let calls=0;const f={uri:{fsPath:'/tmp/voder-project'}};
 const svc=new ChatRunService(async()=>{calls++;throw Error('must not start');});
 await assert.rejects(svc.start(f,'请在当前工作区根目录新增一个文件voder_chat_smoke_test.py',()=>{},{nativeProfile:'/private/gate.json',operatorTarget:'source.py',requiredFiles:['source.py']}),e=>e.code==='CHAT_NEW_FILE_UNSUPPORTED');
 assert.equal(calls,0);
});


test('A12-compatible Node version-manager symlink is canonicalized and persisted, not weakened',async()=>{
 const f=await fixture();try{
  const requested=path.join(f.tmp,'.voder-tools/node/bin/node');
  const canonical=path.join(f.tmp,'.voder-tools/node-versions/node-v24.13.0-linux-x64/bin/node');
  await fs.mkdir(path.dirname(canonical),{recursive:true});
  await fs.writeFile(canonical,'fixture real node',{mode:0o755});
  await fs.unlink(requested);await fs.symlink(canonical,requested);
  const profile=await createProfile({folder:f.folder,agentRoot:f.agent,storageRoot:f.tmp,targetFile:path.join(f.repo,'target.py'),testFile:path.join(f.repo,'test_target.py')});
  const cfg=JSON.parse(await fs.readFile(profile.path));
  assert.equal(cfg.nodeBinary,canonical);
  assert.equal((await validateProfile(profile.path,f.folder)).target,'target.py');
  assert.equal((await fs.lstat(cfg.nodeBinary)).isSymbolicLink(),false);
 }finally{await f.finish();}
});

test('managed Node multihop symlink inside version tree resolves to canonical executable',async()=>{
 const f=await fixture();try{
  const requested=path.join(f.tmp,'.voder-tools/node/bin/node');
  const dist=path.join(f.tmp,'.voder-tools/node-versions/node-v24/bin/node');
  const middle=path.join(f.tmp,'.voder-tools/node-versions/current-node');
  await fs.mkdir(path.dirname(dist),{recursive:true});await fs.writeFile(dist,'fixture',{mode:0o755});
  await fs.symlink(dist,middle);await fs.unlink(requested);await fs.symlink(middle,requested);
  const {pinnedManagedNode}=require('../src/native-profile.cjs');
  assert.equal(await pinnedManagedNode(f.tmp),dist);
 }finally{await f.finish();}
});

test('managed Node symlink escaping trusted Voder tool tree is rejected before any profile write',async()=>{
 const f=await fixture();try{
  const requested=path.join(f.tmp,'.voder-tools/node/bin/node');const outside=path.join(f.tmp,'outside-node');
  await fs.writeFile(outside,'fixture',{mode:0o755});await fs.unlink(requested);await fs.symlink(outside,requested);
  await assert.rejects(createProfile({folder:f.folder,agentRoot:f.agent,storageRoot:f.tmp,targetFile:path.join(f.repo,'target.py'),testFile:path.join(f.repo,'test_target.py')}),e=>e.code==='CHAT_NODE_OUTSIDE_TRUSTED_ROOT');
  await assert.rejects(fs.stat(path.join(f.tmp,'native-profiles')),e=>e.code==='ENOENT');
 }finally{await f.finish();}
});

test('broken Node symlink is blocked clearly and before profile creation',async()=>{
 const f=await fixture();try{
  const requested=path.join(f.tmp,'.voder-tools/node/bin/node');await fs.unlink(requested);await fs.symlink(path.join(f.tmp,'.voder-tools/node-versions/absent/bin/node'),requested);
  const {pinnedManagedNode}=require('../src/native-profile.cjs');
  await assert.rejects(pinnedManagedNode(f.tmp),e=>e.code==='CHAT_NODE_RESOLVE_FAILED');
 }finally{await f.finish();}
});

test('writable or nonexecutable Node target is refused',async()=>{
 const f=await fixture();try{
  const requested=path.join(f.tmp,'.voder-tools/node/bin/node');
  await fs.chmod(requested,0o666);
  const {pinnedManagedNode}=require('../src/native-profile.cjs');
  await assert.rejects(pinnedManagedNode(f.tmp),e=>e.code==='CHAT_NODE_UNSAFE_EXECUTABLE');
  await fs.chmod(requested,0o777);
  await assert.rejects(pinnedManagedNode(f.tmp),e=>e.code==='CHAT_NODE_UNSAFE_EXECUTABLE');
 }finally{await f.finish();}
});

test('insecure managed Node directory is rejected even when target is valid',async()=>{
 const f=await fixture();try{
  const dir=path.join(f.tmp,'.voder-tools/node/bin');await fs.chmod(dir,0o777);
  const {pinnedManagedNode}=require('../src/native-profile.cjs');
  await assert.rejects(pinnedManagedNode(f.tmp),e=>e.code==='CHAT_NODE_UNSAFE_DIRECTORY');
 }finally{await f.finish();}
});

test('gateway and daemon symlink refusal remains unchanged',async()=>{
 const f=await fixture();try{
  const gateway=path.join(f.tmp,'.voder-build/release/voder-gateway');
  const gatewayReal=path.join(f.tmp,'.voder-build/release/gateway-real');
  await fs.rename(gateway,gatewayReal);await fs.symlink(gatewayReal,gateway);
  await assert.rejects(createProfile({folder:f.folder,agentRoot:f.agent,storageRoot:f.tmp,targetFile:path.join(f.repo,'target.py'),testFile:path.join(f.repo,'test_target.py')}),e=>e.code==='CHAT_UNSAFE_FILE');
 }finally{await f.finish();}
});

test('a swapped Node symlink never changes a previously pinned profile executable',async()=>{
 const f=await fixture();try{
  const requested=path.join(f.tmp,'.voder-tools/node/bin/node');
  const original=path.join(f.tmp,'.voder-tools/node-versions/v24/bin/node');
  const replacement=path.join(f.tmp,'.voder-tools/node-versions/v25/bin/node');
  for(const file of [original,replacement]){await fs.mkdir(path.dirname(file),{recursive:true});await fs.writeFile(file,'fixture',{mode:0o755});}
  await fs.unlink(requested);await fs.symlink(original,requested);
  const profile=await createProfile({folder:f.folder,agentRoot:f.agent,storageRoot:f.tmp,targetFile:path.join(f.repo,'target.py'),testFile:path.join(f.repo,'test_target.py')});
  await fs.unlink(requested);await fs.symlink(replacement,requested);
  assert.equal(JSON.parse(await fs.readFile(profile.path)).nodeBinary,original);
  assert.equal((await validateProfile(profile.path,f.folder)).target,'target.py');
 }finally{await f.finish();}
});

test('validator runs in pinned sandbox image, never assumes Python 3.10 exists on the host',async()=>{
 const f=await fixture();try{
  const profile=await createProfile({folder:f.folder,agentRoot:f.agent,storageRoot:f.tmp,targetFile:path.join(f.repo,'target.py'),testFile:path.join(f.repo,'test_target.py')});
  const cfg=JSON.parse(await fs.readFile(profile.path));const policy=JSON.parse(await fs.readFile(cfg.policyFile));
  assert.deepEqual(policy.validator_args,['/usr/local/bin/python3.10','-B','test_target.py']);
  assert.equal(policy.validator_program,'/usr/bin/env');
 }finally{await f.finish();}
});

// Real AutoDL incident: installed Node distribution owned by UID 1001, process UID 0.
test('foreign-owned 0755 Node tree is snapshotted into current-user-only pinned executable',async(t)=>{
 if(process.getuid?.()!==0){t.skip('requires root to reproduce UID 1001 AutoDL tree');return;}
 const f=await fixture();try{
  const request=path.join(f.tmp,'.voder-tools/node/bin/node');
  const version=path.join(f.tmp,'.voder-tools/node-versions/node-v24.13.0-linux-x64');
  const actual=path.join(version,'bin/node');
  await fs.mkdir(path.dirname(actual),{recursive:true,mode:0o755});
  await fs.writeFile(actual,'node-binary-from-managed-distribution',{mode:0o755});
  await fs.chown(version,1001,1001);await fs.chown(path.join(version,'bin'),1001,1001);await fs.chown(actual,1001,1001);
  await fs.unlink(request);await fs.symlink(actual,request);
  const profile=await createProfile({folder:f.folder,agentRoot:f.agent,storageRoot:f.tmp,targetFile:path.join(f.repo,'target.py'),testFile:path.join(f.repo,'test_target.py')});
  const cfg=JSON.parse(await fs.readFile(profile.path));
  assert.notEqual(cfg.nodeBinary,actual);assert.match(cfg.nodeBinary,/\.voder-chat-node-cache\/[a-f0-9]{64}\.node$/);
  const st=await fs.lstat(cfg.nodeBinary);assert.equal(st.uid,0);assert.equal(st.mode&0o077,0);
  assert.equal((await validateProfile(profile.path,f.folder)).target,'target.py');
  // Changing the original owner-controlled executable doesn't alter the cached bytes.
  await fs.writeFile(actual,'completely different untrusted bytes');
  assert.equal((await fs.readFile(cfg.nodeBinary,'utf8')),'node-binary-from-managed-distribution');
  assert.equal((await validateProfile(profile.path,f.folder)).target,'target.py');
 }finally{await f.finish();}
});

test('foreign-owned Node snapshot cache is reused without duplicated executable',async(t)=>{
 if(process.getuid?.()!==0){t.skip('requires root');return;}
 const f=await fixture();try{
  const request=path.join(f.tmp,'.voder-tools/node/bin/node');
  const directory=path.join(f.tmp,'.voder-tools/node-versions/v24/bin');
  const actual=path.join(directory,'node');await fs.mkdir(directory,{recursive:true});await fs.writeFile(actual,'fixture immutable',{mode:0o755});
  await fs.chown(path.dirname(directory),1001,1001);await fs.chown(directory,1001,1001);await fs.chown(actual,1001,1001);
  await fs.unlink(request);await fs.symlink(actual,request);
  const args={folder:f.folder,agentRoot:f.agent,storageRoot:f.tmp,targetFile:path.join(f.repo,'target.py'),testFile:path.join(f.repo,'test_target.py')};
  const one=await createProfile(args),two=await createProfile(args);
  const x=JSON.parse(await fs.readFile(one.path));const y=JSON.parse(await fs.readFile(two.path));
  assert.equal(x.nodeBinary,y.nodeBinary);
  assert.deepEqual((await fs.readdir(path.dirname(x.nodeBinary))).filter(n=>n.endsWith('.node')).length,1);
 }finally{await f.finish();}
});

test('corrupted private Node snapshot fails closed during profile validation',async(t)=>{
 if(process.getuid?.()!==0){t.skip('requires root');return;}
 const f=await fixture();try{
  const request=path.join(f.tmp,'.voder-tools/node/bin/node');
  const directory=path.join(f.tmp,'.voder-tools/node-versions/v24/bin');const actual=path.join(directory,'node');
  await fs.mkdir(directory,{recursive:true});await fs.writeFile(actual,'immutable',{mode:0o755});await fs.chown(directory,1001,1001);await fs.chown(actual,1001,1001);
  await fs.unlink(request);await fs.symlink(actual,request);
  const profile=await createProfile({folder:f.folder,agentRoot:f.agent,storageRoot:f.tmp,targetFile:path.join(f.repo,'target.py'),testFile:path.join(f.repo,'test_target.py')});
  const cfg=JSON.parse(await fs.readFile(profile.path));await fs.writeFile(cfg.nodeBinary,'attacker-manipulated',{mode:0o500});
  await assert.rejects(validateProfile(profile.path,f.folder),e=>e.code==='CHAT_NODE_CACHE_HASH_MISMATCH');
 }finally{await f.finish();}
});

test('foreign-owned group-writable distribution is still refused with no snapshot',async(t)=>{
 if(process.getuid?.()!==0){t.skip('requires root');return;}
 const f=await fixture();try{
  const request=path.join(f.tmp,'.voder-tools/node/bin/node');const dir=path.join(f.tmp,'.voder-tools/node-versions/v24/bin');
  const actual=path.join(dir,'node');await fs.mkdir(dir,{recursive:true});await fs.writeFile(actual,'source',{mode:0o755});
  await fs.chown(dir,1001,1001);await fs.chmod(dir,0o775);
  await fs.unlink(request);await fs.symlink(actual,request);
  await assert.rejects(createProfile({folder:f.folder,agentRoot:f.agent,storageRoot:f.tmp,targetFile:path.join(f.repo,'target.py'),testFile:path.join(f.repo,'test_target.py')}),e=>e.code==='CHAT_NODE_UNSAFE_DIRECTORY');
  await assert.rejects(fs.stat(path.join(f.tmp,'.voder-chat-node-cache')),e=>e.code==='ENOENT');
 }finally{await f.finish();}
});
