import {test} from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,readdir,rm,stat,access} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {spawnSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {HostCandidateVerifier} from '../src/candidate-verifier.mjs';
import {classifyFailure,DiagnosticJournal} from '../../voder-diagnostics/src/index.mjs';
const hash=x=>createHash('sha256').update(x).digest('hex');
const run=(cwd,...args)=>{
 const result=spawnSync('/usr/bin/git',args,{cwd,encoding:'utf8',env:{...process.env,GIT_AUTHOR_NAME:'Test',GIT_AUTHOR_EMAIL:'test@example.invalid',GIT_COMMITTER_NAME:'Test',GIT_COMMITTER_EMAIL:'test@example.invalid'}});
 assert.equal(result.status,0,result.stderr);
};
const good='diff --git a/project/a.txt b/project/a.txt\n--- a/project/a.txt\n+++ b/project/a.txt\n@@ -1 +1 @@\n-OLD\n+NEW\n';
const bad='diff --git a/project/a.txt b/project/a.txt\n--- a/project/a.txt\n+++ b/project/a.txt\n@@ -1,2 +1,2 @@\n-OLD\n+NEW\ndiff --git a/project/b.txt b/project/b.txt\n';
async function scenario(callback){
 const root=await mkdtemp(join(tmpdir(),'voder-preflight-test-'));
 try{
  const cwd=join(root,'repo'),store=join(root,'checks');await mkdir(join(cwd,'src'),{recursive:true});await mkdir(join(cwd,'project'));
  await writeFile(join(cwd,'src/model.patch'),'PENDING\n');await writeFile(join(cwd,'project/a.txt'),'OLD\n');
  run(cwd,'init','-q');run(cwd,'add','.');run(cwd,'commit','-qm','baseline');
  const program='/bin/cat',args=['src/model.patch'],policy=join(root,'policy.json');
  await writeFile(policy,JSON.stringify({schema:'voder.gateway.policy.v1',repository_path:cwd,target_path:'src/model.patch',validator_program:program,validator_args:args,validator_sha256:hash(`${program}\0v1\0[${args.map(x=>JSON.stringify(x)).join(', ')}]`)}));
  const cli=join(root,'fake-cli.mjs'),marker=join(root,'cli-called');
  await writeFile(cli,`import {writeFileSync} from 'node:fs';writeFileSync(${JSON.stringify(marker)},'YES');process.stderr.write('FAKE_SANDBOX_REACHED');throw new Error('FAKE_SANDBOX_FAILURE');\n`);
  const verifier=new HostCandidateVerifier({cwd,nativeGateway:{policyFile:policy,targetPath:'src/model.patch',attemptStore:store,nodeBinary:process.execPath,modalCli:cli},timeoutMs:120000});
  const candidate=content=>({relative_path:'src/model.patch',old_sha256:hash('PENDING\n'),new_content:content});
  await callback({root,cwd,store,marker,verifier,candidate});
 }finally{await rm(root,{recursive:true,force:true});}
}
async function readCheck(store){const dirs=await readdir(join(store,'candidate-checks'));assert.equal(dirs.length,1);const dir=join(store,'candidate-checks',dirs[0]);const detail=JSON.parse(await readFile(join(dir,'patch-preflight.private.json')));return {dir,detail};}
test('corrupt hunk counts return PATCH_FORMAT without Modal, with private read-only git evidence',async()=>scenario(async({cwd,store,marker,verifier,candidate})=>{
 const res=await verifier.verifyCandidate(candidate(bad));assert.equal(res.status,'FAIL');assert.equal(res.backend,'host_patch_preflight');assert.equal(res.sandboxed,false);assert.match(res.diagnostic,/PATCH_FORMAT.*corrupt patch/i);
 const {dir,detail}=await readCheck(store);assert.equal(detail.kind,'PATCH_FORMAT');assert.equal(detail.checks[0].stage,'PARSE');assert.equal((await stat(join(dir,'patch-preflight.private.json'))).mode&0o777,0o600);
 assert.deepEqual((await readdir(dir)).filter(x=>/^(receipt|request)\.json$/.test(x)),[]);
 await assert.rejects(access(marker));assert.equal(await readFile(join(cwd,'src/model.patch'),'utf8'),'PENDING\n');assert.equal(await readFile(join(cwd,'project/a.txt'),'utf8'),'OLD\n');
}));
test('valid syntax but nonexistent patch target fails fast with PATCH_CONTEXT, no sandbox',async()=>scenario(async({store,marker,verifier,candidate})=>{
 const misplaced=good.replaceAll('project/a.txt','project/not-present.txt');const res=await verifier.verifyCandidate(candidate(misplaced));
 assert.equal(res.status,'FAIL');assert.match(res.diagnostic,/PATCH_CONTEXT/);const {detail}=await readCheck(store);assert.equal(detail.checks[0].stage,'PARSE');assert.equal(detail.checks[1].stage,'APPLICABILITY');await assert.rejects(access(marker));
}));
test('valid literal patch passes host preflight, then still requires real independent sandbox receipt',async()=>scenario(async({store,marker,verifier,candidate})=>{
 await assert.rejects(verifier.verifyCandidate(candidate(good)),{code:'VERIFIER_INFRASTRUCTURE'});
 assert.equal(await readFile(marker,'utf8'),'YES');const {dir,detail}=await readCheck(store);
 assert.equal(detail.status,'PASS');assert.equal(detail.checks.at(-1).rc,0);await assert.rejects(access(join(dir,'receipt.json')));
}));
test('host-constructed patch relative to project subtree passes applicability checks',async()=>scenario(async({store,verifier,marker,candidate})=>{
 verifier.gate.patchBaseDir='project'; // Explicit trusted adapter configuration, never inferred from a directory name.
 const nested=good.replaceAll('a/project/a.txt','a/a.txt').replaceAll('b/project/a.txt','b/a.txt');
 await assert.rejects(verifier.verifyCandidate(candidate(nested)),{code:'VERIFIER_INFRASTRUCTURE'});
 const {detail}=await readCheck(store);assert.equal(detail.status,'PASS');assert.equal(detail.checks.at(-1).context,'project');assert.equal(await readFile(marker,'utf8'),'YES');
}));
test('empty/unified-diff-free text is not accepted as patch preflight PASS',async()=>scenario(async({store,marker,verifier,candidate})=>{
 const res=await verifier.verifyCandidate(candidate('just notes\n'));assert.equal(res.status,'FAIL');assert.match(res.diagnostic,/PATCH_FORMAT/);await assert.rejects(access(marker));const {detail}=await readCheck(store);assert.equal(detail.kind,'PATCH_FORMAT');
}));

test('host preflight typed failures classify as patch repair rather than opaque infrastructure',()=>{assert.equal(classifyFailure('PATCH_FORMAT: git apply rejected candidate'),'PATCH_FORMAT');assert.equal(classifyFailure('PATCH_CONTEXT: file absent'),'PATCH_CONTEXT');});

test('host preflight evidence is explicitly labeled non-Modal by diagnostics',()=>{const j=new DiagnosticJournal();const result=j.observeSandbox({status:'FAIL',backend:'host_patch_preflight',diagnostic:'PATCH_FORMAT: corrupt patch at line 373'},{candidateDigest:'a'.repeat(64)});assert.equal(result.source,'HOST_GIT_PREFLIGHT');assert.equal(result.classification,'PATCH_FORMAT');});

// Reproduce Git's dangerous behaviour: in a repository subdir, a patch may
// print "Skipped patch" yet return 0, even when nothing was verified.
test('project preflight must reject an all-skipped patch instead of rc-zero false PASS',async()=>scenario(async({store,marker,verifier,candidate})=>{
 verifier.gate.patchBaseDir='project'; // Explicit trusted adapter configuration, never inferred from a directory name.
 const wrong='diff --git a/a.txt b/a.txt\n--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-WRONG\n+NEW\n';
 const res=await verifier.verifyCandidate(candidate(wrong));
 assert.equal(res.status,'FAIL');assert.equal(res.backend,'host_patch_preflight');assert.match(res.diagnostic,/PATCH_CONTEXT/);
 const {detail}=await readCheck(store);assert.equal(detail.status,'FAIL');assert.equal(detail.checks.at(-1).context,'project');
 assert.equal(detail.checks.at(-1).expectedPaths,1);assert.equal(detail.checks.at(-1).rc,1);
 await assert.rejects(access(marker));
}));
test('project preflight cannot PASS using matching decoy at workspace root',async()=>scenario(async({cwd,store,marker,verifier,candidate})=>{
 verifier.gate.patchBaseDir='project'; // Explicit trusted adapter configuration, never inferred from a directory name.
 await writeFile(join(cwd,'a.txt'),'OLD\n');await writeFile(join(cwd,'project/a.txt'),'DIFFERENT\n');
 run(cwd,'add','.');run(cwd,'commit','-qm','test-patch-root-disambiguation');
 const diff='diff --git a/a.txt b/a.txt\n--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-OLD\n+NEW\n';
 const res=await verifier.verifyCandidate(candidate(diff));
 assert.equal(res.status,'FAIL');assert.equal(res.backend,'host_patch_preflight');assert.match(res.diagnostic,/PATCH_CONTEXT/);
 const {detail}=await readCheck(store);assert.equal(detail.checks.at(-1).context,'project');
 await assert.rejects(access(marker));
}));
