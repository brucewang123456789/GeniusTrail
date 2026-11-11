import {test} from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,rm,symlink,readFile} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {createHash} from 'node:crypto';
import {expandCandidate} from '../src/candidate-edits.mjs';
import {RepositoryInspector} from '../src/repository-tools.mjs';
const hash=s=>createHash('sha256').update(s).digest('hex');
async function task(fn){
 const cwd=await mkdtemp(join(tmpdir(),'voder-missing-source-'));
 try{
  await mkdir(join(cwd,'project'));
  await mkdir(join(cwd,'src'));
  await writeFile(join(cwd,'src/model.patch'),'PENDING\n');
  return await fn(cwd,new RepositoryInspector(cwd));
 }finally{await rm(cwd,{recursive:true,force:true});}
}
const model=(base_dir,files,output='src/model.patch')=>({
 relative_path:output,old_sha256:hash('PENDING\n'),unified_diff:{base_dir,files}
});
const member=(path)=>({relative_path:path,old_sha256:hash('old\n'),new_content:'new\n'});
async function errorOf(p){try{await p;assert.fail('expected failure');}catch(e){if(e.code==='ERR_ASSERTION')throw e;return e;}}
test('structured direct edit nonexistent source has actionable relative-only error',()=>task(async(cwd,inspector)=>{
 const e=await errorOf(expandCandidate({relative_path:'src/ghost.ts',old_sha256:hash('old\n'),edits:[{old_text:'old',new_text:'new'}]},inspector));
 assert.equal(e.code,'CANDIDATE_SOURCE_NOT_FOUND');
 assert.match(e.message,/src\/ghost\.ts/);
 assert.match(e.message,/voder_list_files/);
 assert.doesNotMatch(e.message,new RegExp(cwd.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')));
}));
test('missing patch output preimage is explicitly classified',()=>task(async(_cwd,inspector)=>{
 const e=await errorOf(expandCandidate(model('project',[member('a.ts')],'src/missing.patch'),inspector));
 assert.equal(e.code,'CANDIDATE_SOURCE_NOT_FOUND');
 assert.match(e.message,/Patch output preimage/);
}));
test('missing patch base directory is explicitly classified',()=>task(async(_cwd,inspector)=>{
 const e=await errorOf(expandCandidate(model('wrong/project',[member('a.ts')]),inspector));
 assert.equal(e.code,'CANDIDATE_SOURCE_NOT_FOUND');
 assert.match(e.message,/Patch base directory/);
}));
test('missing nested member path is explicitly classified',()=>task(async(_cwd,inspector)=>{
 const e=await errorOf(expandCandidate(model('project',[member('missing.ts')]),inspector));
 assert.equal(e.code,'CANDIDATE_SOURCE_NOT_FOUND');
 assert.match(e.message,/Patch source/);
 assert.match(e.message,/project\/missing\.ts/);
}));
test('missing parent is classified ENOTDIR or ENOENT as missing pinned source',()=>task(async(cwd,inspector)=>{
 await writeFile(join(cwd,'project/not-a-dir'),'hi');
 const e=await errorOf(expandCandidate(model('project',[member('not-a-dir/deep.ts')]),inspector));
 assert.equal(e.code,'CANDIDATE_SOURCE_NOT_FOUND');
}));
test('symlink remains denied, not converted to missing-file error',()=>task(async(cwd,inspector)=>{
 await writeFile(join(cwd,'project/existing.ts'),'old\n');
 await symlink('existing.ts',join(cwd,'project/link.ts'));
 const e=await errorOf(expandCandidate(model('project',[member('link.ts')]),inspector));
 assert.equal(e.code,'UNTRUSTED_PATH');
}));
test('traversal remains denied, not converted to missing-file error',()=>task(async(_cwd,inspector)=>{
 const e=await errorOf(expandCandidate(model('project',[member('../missing.ts')]),inspector));
 assert.ok(['INVALID_CANDIDATE','INVALID_PATH'].includes(e.code));
}));
test('present pinned source still produces a valid patch and performs no writes',()=>task(async(cwd,inspector)=>{
 await writeFile(join(cwd,'project/a.ts'),'old\n');
 const result=await expandCandidate(model('project',[member('a.ts')]),inspector);
 assert.match(result.new_content,/diff --git a\/a\.ts b\/a\.ts/);
 assert.equal(await readFile(join(cwd,'project/a.ts'),'utf8'),'old\n');
}));
