import {test} from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {createHash} from 'node:crypto';
import {spawnSync} from 'node:child_process';
import {expandCandidate,repairLiteralPatchMetadata} from '../src/candidate-edits.mjs';
import {RepositoryInspector} from '../src/repository-tools.mjs';
import {DiagnosticJournal} from '../../voder-diagnostics/src/index.mjs';
const sha=s=>createHash('sha256').update(s).digest('hex');
const gitCheck=(cwd,content)=>spawnSync('/usr/bin/git',['apply','--check','--verbose','--directory=project','-'],{cwd,input:content,encoding:'utf8'});
async function fixture(callback){
 const cwd=await mkdtemp(join(tmpdir(),'voder-recovery-'));
 try{await mkdir(join(cwd,'src'));await mkdir(join(cwd,'project'));await writeFile(join(cwd,'src/model.patch'),'PENDING\n');
  await callback(cwd,new RepositoryInspector(cwd));
 }finally{await rm(cwd,{recursive:true,force:true});}
}
const candidate=(s)=>({relative_path:'src/model.patch',old_sha256:sha('PENDING\n'),new_content:s});
const header=(file='packages/a.txt')=>`diff --git a/${file} b/${file}\n--- a/${file}\n+++ b/${file}\n`;
const good=header()+ '@@ -1,2 +1,2 @@\n one\n-old\n+new\n';
const malformed=header()+ '@@ -1,500 +1,20 @@\n one\n-old\n+new\n';
test('literal wrong hunk counts repaired automatically, verified file-by-file, no source writes',()=>fixture(async(cwd,inspector)=>{
 await writeFile(join(cwd,'project/packages.txt'),'kept');await mkdir(join(cwd,'project/packages'));await writeFile(join(cwd,'project/packages/a.txt'),'one\nold\n');
 assert.notEqual(gitCheck(cwd,malformed).status,0);
 const c=await expandCandidate(candidate(malformed),inspector);
 assert.notEqual(c.new_content,malformed);assert.equal(c.new_content,good);
 const checked=gitCheck(cwd,c.new_content);assert.equal(checked.status,0,checked.stderr);
 assert.equal(await readFile(join(cwd,'src/model.patch'),'utf8'),'PENDING\n');
 assert.equal(await readFile(join(cwd,'project/packages/a.txt'),'utf8'),'one\nold\n');
}));
test('valid literal bytes never modified',async()=>assert.equal(repairLiteralPatchMetadata(good).content,good));
test('multi-file independent hunk counts repaired without touching +/- code lines',()=>{
 const original=header('packages/a.txt')+'@@ -1,42 +1,99 @@\n-old\n+new\n'+header('packages/b.txt')+'@@ -1,9 +1,80 @@\n-x\n+y\n';
 const r=repairLiteralPatchMetadata(original);
 assert.equal(r.repairedHunks,2);
 assert.equal(r.content,header('packages/a.txt')+'@@ -1,1 +1,1 @@\n-old\n+new\n'+header('packages/b.txt')+'@@ -1,1 +1,1 @@\n-x\n+y\n');
});
test('unrepairable garbage does not get guessed, and remains denied by git',()=>fixture(async(cwd,inspector)=>{
 const broken=header()+'@@ -1,1 +1,1 @@\n-old\n+new\nINVALID_FRAGMENT\n';
 const c=await expandCandidate(candidate(broken),inspector);
 assert.equal(c.new_content,broken);assert.notEqual(gitCheck(cwd,c.new_content).status,0);
}));
test('path alias base_dir duplication is normalized only when pinned SHA matches',()=>fixture(async(cwd,inspector)=>{
 await mkdir(join(cwd,'project/packages'));await writeFile(join(cwd,'project/packages/a.txt'),'old\n');
 const c=await expandCandidate({relative_path:'src/model.patch',old_sha256:sha('PENDING\n'),unified_diff:{base_dir:'project',files:[{relative_path:'project/packages/a.txt',old_sha256:sha('old\n'),new_content:'new\n'}]}},inspector);
 assert.match(c.new_content,/diff --git a\/packages\/a.txt b\/packages\/a.txt/);
 assert.equal(gitCheck(cwd,c.new_content).status,0);
}));
test('path alias still rejects missing source and stale pins',()=>fixture(async(cwd,inspector)=>{
 await mkdir(join(cwd,'project/packages'));await writeFile(join(cwd,'project/packages/a.txt'),'old\n');
 const x=hash=>({relative_path:'src/model.patch',old_sha256:sha('PENDING\n'),unified_diff:{base_dir:'project',files:[{relative_path:'project/packages/a.txt',old_sha256:hash,new_content:'new\n'}]}});
 await assert.rejects(expandCandidate(x(sha('different\n')),inspector),{code:'STALE_SOURCE'});
 await assert.rejects(expandCandidate({...x(sha('old\n')),unified_diff:{base_dir:'project',files:[{relative_path:'project/packages/none.txt',old_sha256:sha('old\n'),new_content:'new\n'}]}},inspector),{code:'CANDIDATE_SOURCE_NOT_FOUND'});
}));
test('same git context error on distinct repaired candidates must not stop new attempts',()=>{
 const j=new DiagnosticJournal({maxIdenticalFailures:2});
 for(const digit of ['1','2','3','4']){
  const r=j.observeSandbox({status:'FAIL',backend:'host_patch_preflight',diagnostic:'PATCH_CONTEXT: source did not match'}, {candidateDigest:digit.repeat(64)});
  assert.equal(r.stop,false);assert.equal(r.identical_count,1);j.assertMayProceed();
 }
 const repeated=j.observeSandbox({status:'FAIL',backend:'host_patch_preflight',diagnostic:'PATCH_CONTEXT: source did not match'}, {candidateDigest:'4'.repeat(64)});
 assert.equal(repeated.stop,true);
});
test('same patch header but different actual source context remains rejected',()=>fixture(async(cwd,inspector)=>{
 await mkdir(join(cwd,'project/packages'));await writeFile(join(cwd,'project/packages/a.txt'),'different\n');
 const c=await expandCandidate(candidate(malformed),inspector);
 assert.notEqual(gitCheck(cwd,c.new_content).status,0);
}));
