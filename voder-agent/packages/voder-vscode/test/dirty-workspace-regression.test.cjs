'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs/promises'),path=require('node:path'),crypto=require('node:crypto'),{execFileSync}=require('node:child_process');
const source=process.env.VODER_WORKSPACE_REGRESSION_SOURCE||path.join(__dirname,'../src');
const{ordinaryWorkspace}=require(path.join(source,'ordinary-workspace.cjs'));
const{publishOriginalProject}=require(path.join(source,'original-project-publication.cjs'));
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
const git=(root,...args)=>execFileSync('/usr/bin/git',['-c','core.hooksPath=/dev/null','-c','commit.gpgsign=false','-C',root,...args],{encoding:'utf8',stdio:['ignore','pipe','pipe'],env:{...process.env,GIT_OPTIONAL_LOCKS:'0'}});
async function fixture(fn){
 // Publication intentionally refuses world-writable ancestors such as /tmp.
 const base=await fs.mkdtemp(path.join(await fs.realpath(require('node:os').homedir()),'.voder-dirty-regression-'));
 const root=path.join(base,'project'),storageRoot=path.join(base,'storage');
 try{await fs.mkdir(root,{mode:0o700});await fs.mkdir(storageRoot,{mode:0o700});
 await fs.writeFile(path.join(root,'submission_source_writer.py'),'print("before")\n');
 git(root,'init','-q');git(root,'add','submission_source_writer.py');git(root,'-c','user.name=Fixture','-c','user.email=fixture@invalid','commit','-qm','original baseline');
 return await fn({base,root,storageRoot});}finally{await fs.rm(base,{recursive:true,force:true});}
}
async function unrelated(root){
 const assets=['Libo.docx','VERILOOP E2 - TOWARD VERIFIABLE INTELLIGENCE.pdf','VeriLoop.xlsx','game.zip','veriloop_e2_benchmark_result.png','wanzi.pptx','研究 资料.txt'];
 const map=new Map();for(const name of assets){const b=Buffer.from('existing asset: '+name+'\0\xff');await fs.writeFile(path.join(root,name),b);map.set(name,b);}
 await fs.mkdir(path.join(root,'.voder'));await fs.writeFile(path.join(root,'.voder/.gitignore'),'*\n');
 await fs.mkdir(path.join(root,'voder-chat-e2e'));await fs.writeFile(path.join(root,'voder-chat-e2e/keep.txt'),'unrelated project');
 return map;
}
async function state(root){return{head:git(root,'rev-parse','HEAD'),index:await fs.readFile(path.join(root,'.git/index')),status:git(root,'status','--porcelain=v1','--untracked-files=all'),source:await fs.readFile(path.join(root,'submission_source_writer.py'))};}
async function assertState(root,before){const after=await state(root);assert.deepEqual(after,before);}
async function candidate(f){
 const assets=await unrelated(f.root),before=await state(f.root),copy=await ordinaryWorkspace(f);
 assert.equal(copy.isolated,true);assert.equal(copy.reason,'DIRTY_REPOSITORY');
 assert.equal(git(copy.root,'status','--porcelain=v1','--untracked-files=all'),'');
 const data=Buffer.from('print("verified candidate")\n');await fs.writeFile(path.join(copy.root,'submission_source_writer.py'),data);
 // This is a publication TEST DOUBLE; no native success or model quality is claimed.
 const result={status:'DELIVERED',verified:true,taskAcceptance:{schema:'voder.task-acceptance.v1',delivery:'DELIVERED',artifacts:[{path:'submission_source_writer.py',sha256:sha(data)}]}};
 return{assets,before,copy,data,result,sessionId:crypto.randomUUID(),originalRoot:f.root,privateRoot:copy.root,storageRoot:f.storageRoot};
}
test('untracked documents trigger a clean private copy without changing original HEAD, index or bytes',()=>fixture(async f=>{
 const assets=await unrelated(f.root),before=await state(f.root),r=await ordinaryWorkspace(f);
 assert.equal(r.isolated,true);assert.equal(r.reason,'DIRTY_REPOSITORY');assert.equal(git(r.root,'status','--porcelain=v1','--untracked-files=all'),'');
 await assertState(f.root,before);for(const[n,b]of assets)assert.deepEqual(await fs.readFile(path.join(r.root,n)),b);
}));
test('clean committed workspace retains its existing direct route',()=>fixture(async f=>{assert.deepEqual(await ordinaryWorkspace(f),{root:f.root,isolated:false});assert.deepEqual(await fs.readdir(f.storageRoot),[]);}));
test('read-only dirty workspace retains its existing direct route',()=>fixture(async f=>{await unrelated(f.root);const before=await state(f.root);assert.deepEqual(await ordinaryWorkspace({...f,readOnly:true}),{root:f.root,isolated:false});await assertState(f.root,before);assert.deepEqual(await fs.readdir(f.storageRoot),[]);}));
test('staged and unstaged edits are copied from current bytes and original index remains exact',()=>fixture(async f=>{
 await fs.writeFile(path.join(f.root,'submission_source_writer.py'),'staged\n');git(f.root,'add','submission_source_writer.py');await fs.writeFile(path.join(f.root,'submission_source_writer.py'),'unstaged\n');
 const before=await state(f.root),r=await ordinaryWorkspace(f);assert.equal(r.reason,'DIRTY_REPOSITORY');assert.equal(git(r.root,'show','HEAD:submission_source_writer.py'),'unstaged\n');await assertState(f.root,before);
}));
test('publication accepts unrelated space/Unicode baseline names and writes only the approved artifact',()=>fixture(async f=>{
 const c=await candidate(f);let approvals=0;
 const r=await publishOriginalProject({...c,approve:async s=>{approvals++;assert.equal(s.root,f.root);assert.deepEqual(s.files,['submission_source_writer.py']);return true;}});
 assert.equal(r.status,'DELIVERED');assert.equal(approvals,1);assert.deepEqual(await fs.readFile(path.join(f.root,'submission_source_writer.py')),c.data);
 assert.equal(git(f.root,'rev-parse','HEAD'),c.before.head);assert.deepEqual(await fs.readFile(path.join(f.root,'.git/index')),c.before.index);
 for(const[n,b]of c.assets)assert.deepEqual(await fs.readFile(path.join(f.root,n)),b);
 assert.equal(await fs.readFile(path.join(f.root,'voder-chat-e2e/keep.txt'),'utf8'),'unrelated project');
 assert.equal(await fs.readFile(path.join(f.root,'.voder/.gitignore'),'utf8'),'*\n');
}));
test('unverified candidate cannot publish into a dirty original project',()=>fixture(async f=>{
 const c=await candidate(f);await assert.rejects(publishOriginalProject({...c,result:{...c.result,verified:false},approve:async()=>true}),{code:'PUBLICATION_UNVERIFIED'});await assertState(f.root,c.before);
}));
test('declined publication retains all original files',()=>fixture(async f=>{
 const c=await candidate(f);await assert.rejects(publishOriginalProject({...c,approve:async()=>false}),{code:'PUBLICATION_DECLINED'});await assertState(f.root,c.before);
}));
test('original target edits after copying remain protected from publication',()=>fixture(async f=>{
 const c=await candidate(f);await fs.writeFile(path.join(f.root,'submission_source_writer.py'),'concurrent user edit\n');const before=await state(f.root);
 await assert.rejects(publishOriginalProject({...c,approve:async()=>true}),{code:'PUBLICATION_CONFLICT'});await assertState(f.root,before);
}));
test('original target edits during approval remain protected from publication',()=>fixture(async f=>{
 const c=await candidate(f);await assert.rejects(publishOriginalProject({...c,approve:async()=>{await fs.writeFile(path.join(f.root,'submission_source_writer.py'),'edit while approving\n');return true;}}),{code:'PUBLICATION_CONFLICT'});
 assert.equal(await fs.readFile(path.join(f.root,'submission_source_writer.py'),'utf8'),'edit while approving\n');
}));
test('unrelated asset baseline does not relax artifact path authorization',()=>fixture(async f=>{
 const c=await candidate(f),result={...c.result,taskAcceptance:{...c.result.taskAcceptance,artifacts:[{path:'VERILOOP E2 - TOWARD VERIFIABLE INTELLIGENCE.pdf',sha256:sha(c.assets.get('VERILOOP E2 - TOWARD VERIFIABLE INTELLIGENCE.pdf'))}]}};
 await assert.rejects(publishOriginalProject({...c,result,approve:async()=>true}),{code:'PUBLICATION_PATH'});await assertState(f.root,c.before);
}));
test('baseline traversal and duplicate entries remain rejected',()=>fixture(async f=>{
 const c=await candidate(f),p=path.join(f.storageRoot,'working-copies',path.basename(c.copy.root)+'.json'),original=JSON.parse(await fs.readFile(p,'utf8'));
 for(const row of [{path:'../escape',sha256:'a'.repeat(64)},original.files[0]]){
  await fs.writeFile(p,JSON.stringify({...original,files:[...original.files,row]}));
  await assert.rejects(publishOriginalProject({...c,approve:async()=>true}),{code:'PUBLICATION_MANIFEST'});await assertState(f.root,c.before);
 }
}));
test('untracked symlink still fails closed without touching its destination',()=>fixture(async f=>{
 const external=path.join(f.base,'external');await fs.writeFile(external,'must survive');await fs.symlink(external,path.join(f.root,'escape'));
 const before=await state(f.root);await assert.rejects(ordinaryWorkspace(f),{code:'WORKSPACE_COPY_SYMLINK'});await assertState(f.root,before);assert.equal(await fs.readFile(external,'utf8'),'must survive');
}));
