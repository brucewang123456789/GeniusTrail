import {test} from 'node:test';import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,rm,symlink} from 'node:fs/promises';import {join} from 'node:path';import {tmpdir} from 'node:os';import {createHash} from 'node:crypto';import {execFileSync} from 'node:child_process';
import {expandCandidate} from '../src/candidate-edits.mjs';import {RepositoryInspector} from '../src/repository-tools.mjs';
import {classifyFailure} from '../../voder-diagnostics/src/index.mjs';
const sha=s=>createHash('sha256').update(s).digest('hex');
async function fixture(fn){const cwd=await mkdtemp(join(tmpdir(),'voder-patch-artifact-'));try{await mkdir(join(cwd,'src'));await mkdir(join(cwd,'project'));await writeFile(join(cwd,'src/model.patch'),'PENDING\n');await fn(cwd,new RepositoryInspector(cwd));}finally{await rm(cwd,{recursive:true,force:true});}}
const proposal=(files,extra={})=>({relative_path:'src/model.patch',old_sha256:sha('PENDING\n'),unified_diff:{base_dir:'project',files},...extra});
for(const [label,before,after] of [
 ['LF','a\nb\nc\n','a\nB\nC\nD\n'],['CRLF','a\r\nb\r\n','a\r\nB\r\nC\r\n'],['no final newline','one\ntwo','one\nTWO'],['add newline','one','one\n'],['remove newline','one\n','one'],['empty file','','one\n'],['insert beginning','one\ntwo\n','ZERO\none\ntwo\n'],['insert end','one\ntwo\n','one\ntwo\nTHREE\n']])test('host diff applies exactly with '+label,()=>fixture(async(cwd,repo)=>{
 await writeFile(join(cwd,'project/a.txt'),before);
 const result=await expandCandidate(proposal([{relative_path:'a.txt',old_sha256:sha(before),new_content:after}]),repo);
 assert.equal(await readFile(join(cwd,'src/model.patch'),'utf8'),'PENDING\n');assert.equal(await readFile(join(cwd,'project/a.txt'),'utf8'),before);
 execFileSync('git',['apply','--check','-'],{cwd:join(cwd,'project'),input:result.new_content});
 execFileSync('git',['apply','-'],{cwd:join(cwd,'project'),input:result.new_content});assert.equal(await readFile(join(cwd,'project/a.txt'),'utf8'),after);
}));
test('real git accepts 100 deterministic multi-hunk-sized transformations without model hunk counts',()=>fixture(async(cwd,repo)=>{
 for(let n=0;n<100;n++){
  const original=Array.from({length:20+n},(_,i)=>`line ${i}\n`).join('');const changed=original.replace('line 1\n','ONE\nNEW\n').replace(`line ${18+n}\n`,'tail changed\n');
  await writeFile(join(cwd,'project/a.txt'),original);
  const c=await expandCandidate(proposal([{relative_path:'a.txt',old_sha256:sha(original),new_content:changed}]),repo);
  execFileSync('git',['apply','--check','-'],{cwd:join(cwd,'project'),input:c.new_content});
  execFileSync('git',['apply','-'],{cwd:join(cwd,'project'),input:c.new_content});assert.equal(await readFile(join(cwd,'project/a.txt'),'utf8'),changed);
 }
}));
test('exact source edits produce multiple sorted patch members',()=>fixture(async(cwd,repo)=>{
 for(const name of ['a.txt','b.txt'])await writeFile(join(cwd,'project',name),'one\ntwo\n');
 const c=await expandCandidate(proposal(['b.txt','a.txt'].map(relative_path=>({relative_path,old_sha256:sha('one\ntwo\n'),edits:[{old_text:'two',new_text:'TWO\nTHREE'}]}))),repo);
 assert.ok(c.new_content.indexOf('a/a.txt')<c.new_content.indexOf('a/b.txt'));
 execFileSync('git',['apply','--check','-'],{cwd:join(cwd,'project'),input:c.new_content});
}));
test('stale project source and stale output fail before any verifier or write',()=>fixture(async(cwd,repo)=>{
 await writeFile(join(cwd,'project/a.txt'),'current\n');const member={relative_path:'a.txt',old_sha256:sha('stale\n'),new_content:'new\n'};
 await assert.rejects(expandCandidate(proposal([member]),repo),{code:'STALE_SOURCE'});
 await assert.rejects(expandCandidate(proposal([{...member,old_sha256:sha('current\n')}],{old_sha256:'f'.repeat(64)}),repo),{code:'STALE_SOURCE'});
 assert.equal(await readFile(join(cwd,'src/model.patch'),'utf8'),'PENDING\n');
}));
test('traversal, symlink and self-reference do not gain patch construction authority',()=>fixture(async(cwd,repo)=>{
 await writeFile(join(cwd,'project/a.txt'),'old\n');await symlink('a.txt',join(cwd,'project/alias.txt'));
 await assert.rejects(expandCandidate(proposal([{relative_path:'../escape',old_sha256:sha('old\n'),new_content:'new\n'}]),repo),{code:'INVALID_CANDIDATE'});
 await assert.rejects(expandCandidate(proposal([{relative_path:'alias.txt',old_sha256:sha('old\n'),new_content:'new\n'}]),repo),{code:'UNTRUSTED_PATH'});
 const c=proposal([{relative_path:'src/model.patch',old_sha256:sha('PENDING\n'),new_content:'new\n'}]);c.unified_diff.base_dir='';await assert.rejects(expandCandidate(c,repo),{code:'INVALID_PATCH_ARTIFACT'});
}));
test('ambiguous exact edits and duplicate files are rejected rather than guessing context',()=>fixture(async(cwd,repo)=>{
 await writeFile(join(cwd,'project/a.txt'),'same\nsame\n');const file={relative_path:'a.txt',old_sha256:sha('same\nsame\n'),edits:[{old_text:'same',new_text:'changed'}]};
 await assert.rejects(expandCandidate(proposal([file]),repo),{code:'AMBIGUOUS_EDIT'});
 const full={relative_path:'a.txt',old_sha256:file.old_sha256,new_content:'new\n'};await assert.rejects(expandCandidate(proposal([full,full]),repo),{code:'INVALID_PATCH_ARTIFACT'});
}));
test('actual captured Git errors get precise patch-format and base-context classes',()=>{
 assert.equal(classifyFailure('RuntimeError: patch fragment without header at line 219: @@ -1010,5 +1010,50 @@'),'PATCH_FORMAT');
 assert.equal(classifyFailure('RuntimeError: patch failed: project/compiler/compiler.go:95\npatch does not apply'),'PATCH_CONTEXT');
});
test('kernel expands structured patch before verifier and records its provenance without fabricating delivery',()=>fixture(async(cwd,repo)=>{
 const {AgentKernel}=await import('../src/kernel.mjs');const old='one\ntwo\n';await writeFile(join(cwd,'project/a.txt'),old);let checks=0;
 const kernel=new AgentKernel({cwd,broker:{},candidateVerifier:{async verifyCandidate(c){checks++;execFileSync('git',['apply','--check','-'],{cwd:join(cwd,'project'),input:c.new_content});return {status:'PASS',backend:'OFFLINE_GIT_CHECK_FIXTURE',sandboxed:true,evidence_digest:sha(c.new_content),checks:[{name:'git-apply-fixture',passed:true}]};}},modelHost:{async createSession({tools}){return {async prompt(){const checked=await tools.find(t=>t.name==='voder_verify_candidate').execute({candidate:proposal([{relative_path:'a.txt',old_sha256:sha(old),edits:[{old_text:'two',new_text:'TWO\nTHREE'}]}])});assert.equal(checked.status,'PASS');},dispose(){}};}}});
 const result=await kernel.submit({goal:'write a patch artifact'});assert.equal(checks,1);assert.equal(result.status,'BLOCKED');assert.equal(result.orchestration.attempted_native_actions,1);
 assert.ok(kernel.events.some(e=>e.type==='vgr.priority_commit_rejected'&&e.code==='NATIVE_DISPATCH_ERROR'));
 assert.ok(kernel.events.some(e=>e.type==='candidate.prepared'&&e.code==='HOST_CONSTRUCTED_PATCH'));
 assert.equal(await readFile(join(cwd,'project/a.txt'),'utf8'),old);assert.equal(await readFile(join(cwd,'src/model.patch'),'utf8'),'PENDING\n');kernel.close();
}));
