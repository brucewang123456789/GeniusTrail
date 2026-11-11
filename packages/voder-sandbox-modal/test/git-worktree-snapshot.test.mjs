import {test} from 'node:test';import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,rm,symlink,link} from 'node:fs/promises';import {join} from 'node:path';import {tmpdir} from 'node:os';import {execFileSync} from 'node:child_process';
import {snapshotWorkspace} from '../src/manifest.ts';import {executeModalTask} from '../src/execute.ts';
const git=(cwd,args,opts={})=>execFileSync('/usr/bin/git',args,{cwd,env:{...process.env,GIT_CONFIG_NOSYSTEM:'1',GIT_CONFIG_GLOBAL:'/dev/null'},...opts});
async function fixture(fn){const root=await mkdtemp(join(tmpdir(),'voder-git-boundary-'));try{await fn(root);}finally{await rm(root,{recursive:true,force:true});}}
const patch='diff --git a/a.txt b/a.txt\n--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-old\n+new\n';
async function repo(root){const main=join(root,'main'),candidate=join(root,'candidate');await mkdir(join(main,'project'),{recursive:true});await writeFile(join(main,'project/a.txt'),'old\n');await writeFile(join(main,'solution.patch'),patch);git(main,['init','-q']);git(main,['add','--all']);git(main,['-c','user.name=Fixture','-c','user.email=fixture@example.invalid','-c','commit.gpgsign=false','commit','-qm','base']);git(main,['worktree','add','--detach','--quiet',candidate]);return {main,candidate};}
async function materialize(snapshot,remote){await mkdir(remote,{recursive:true});for(const f of snapshot.files){const p=join(remote,f.relative);await mkdir(join(p,'..'),{recursive:true});await writeFile(p,f.bytes);}}
test('real clone and detached worktree produce identical portable source inputs',()=>fixture(async root=>{
 const {main,candidate}=await repo(root),pointer=await readFile(join(candidate,'.git'),'utf8');assert.match(pointer,/^gitdir: /);
 const cloned=join(root,'cloned');git(root,['clone','--no-hardlinks','--quiet',main,cloned]);
 const a=await snapshotWorkspace(main,['solution.patch']),b=await snapshotWorkspace(candidate,['solution.patch']);
 const c=await snapshotWorkspace(cloned,['solution.patch']);assert.equal(c.digest,b.digest);assert.equal(c.totalBytes,b.totalBytes);
 assert.equal(a.digest,b.digest);assert.equal(a.totalBytes,b.totalBytes);assert.deepEqual(a.files.map(f=>f.relative),b.files.map(f=>f.relative));
 assert.ok(!b.files.some(f=>f.relative.split('/').includes('.git')));assert.equal(await readFile(join(candidate,'.git'),'utf8'),pointer);
 assert.equal(git(main,['status','--porcelain']).toString(),'');
}));
test('captured fatal Git pointer failure reproduces, fixed upload validates after host metadata disappears',()=>fixture(async root=>{
 const {main,candidate}=await repo(root);const snapshot=await snapshotWorkspace(candidate),pointer=await readFile(join(candidate,'.git'));
 const broken=join(root,'remote-before'),fixed=join(root,'remote-after');await materialize(snapshot,broken);await materialize(snapshot,fixed);await writeFile(join(broken,'.git'),pointer);
 await rm(join(main,'.git'),{recursive:true,force:true});
 assert.throws(()=>git(broken,['apply','--check','--directory=project','solution.patch'],{stdio:'pipe'}),error=>/not a git repository/.test(error.stderr.toString()));
 git(fixed,['apply','--check','--directory=project','solution.patch']);assert.equal(await readFile(join(fixed,'project/a.txt'),'utf8'),'old\n');
 await writeFile(join(fixed,'invalid.patch'),patch.replace('-old\n','-missing\n'));
 assert.throws(()=>git(fixed,['apply','--check','--directory=project','invalid.patch'],{stdio:'pipe'}),error=>/patch does not apply/.test(error.stderr.toString()));
 assert.equal(await readFile(join(fixed,'project/a.txt'),'utf8'),'old\n');
}));
test('nested submodule git pointers and ordinary git directories never leak metadata',()=>fixture(async root=>{
 await mkdir(join(root,'project/submodule'),{recursive:true});await mkdir(join(root,'.git'));await writeFile(join(root,'.git/config'),'HOST_SECRET');await writeFile(join(root,'project/submodule/.git'),'gitdir: /HOST_PRIVATE_PATH\n');await writeFile(join(root,'project/submodule/a.txt'),'source');
 const s=await snapshotWorkspace(root);assert.deepEqual(s.files.map(f=>f.relative),['project/submodule/a.txt']);assert.ok(!s.checksumText.includes('HOST'));
 await assert.rejects(snapshotWorkspace(root,['project/submodule/.git']),/MISSING_REQUIRED_FILE/);
}));
test('symlink named .git still rejects instead of following or weakening path isolation',()=>fixture(async root=>{
 await writeFile(join(root,'a.txt'),'source');await symlink('/etc',join(root,'.git'));await assert.rejects(snapshotWorkspace(root),/SYMLINK_BLOCKED/);
}));
test('special file named .git still rejects',()=>fixture(async root=>{
 execFileSync('mkfifo',[join(root,'.git')]);await assert.rejects(snapshotWorkspace(root),/SPECIAL_FILE_BLOCKED/);
}));
test('regular source hardlinks and unrelated symlinks remain rejected',()=>fixture(async root=>{
 await writeFile(join(root,'a.txt'),'source');await link(join(root,'a.txt'),join(root,'b.txt'));await assert.rejects(snapshotWorkspace(root),/HARDLINK_BLOCKED/);
 await rm(join(root,'b.txt'));await symlink('a.txt',join(root,'alias'));await assert.rejects(snapshotWorkspace(root),/SYMLINK_BLOCKED/);
}));
test('executeModalTask uploads no worktree pointer and retains receipt integrity with a fixture driver',()=>fixture(async root=>{
 const {candidate}=await repo(root),uploaded=new Map();let terminated=0;
 const stream=s=>new ReadableStream({start(c){if(s)c.enqueue(s);c.close();}}),result=()=>({stdout:stream(''),stderr:stream(''),async wait(){return 0;}});
 const driver={async create(){return {sandboxId:'OFFLINE_FIXTURE',filesystem:{async writeBytes(b,p){uploaded.set(p,b)},async writeText(){}},async exec(){return result()},async terminate(){terminated++;}}}};
 const receipt=await executeModalTask({workspace:candidate,program:'/usr/bin/true',args:[],runId:'fixture-run',actionId:'fixture-action',timeoutMs:1000,memoryMiB:512,cpu:1,maxOutputBytes:1024},driver,{VODER_MODAL_APP:'fixture',VODER_MODAL_IMAGE:'python@sha256:'+'a'.repeat(64),MODAL_TOKEN_ID:'fixture',MODAL_TOKEN_SECRET:'fixture'});
 assert.equal(receipt.exitCode,0);assert.equal(terminated,1);assert.ok(uploaded.has('/workspace/solution.patch'));assert.ok(![...uploaded.keys()].some(p=>p.split('/').includes('.git')));
}));
