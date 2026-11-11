import {test} from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,symlink,rm} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {execFileSync} from 'node:child_process';
import {RepositoryInspector,repositoryReadTools} from '../src/repository-tools.mjs';
const paths=['project/builtin/lib.go','project/expr_test.go','project/builtin/builtin.go','project/ast/print.go','project/ast/visitor.go','project/ast/node.go'];
async function fixture(fn){const cwd=await mkdtemp(join(tmpdir(),'voder-search-scope-'));try{for(const p of paths){await mkdir(join(cwd,p,'..'),{recursive:true});await writeFile(join(cwd,p),'package x\n// go:build literal.*\nimport "strconv"\n');}await fn(new RepositoryInspector(cwd),cwd);}finally{await rm(cwd,{recursive:true,force:true});}}
test('all six real trace file paths are accepted as literal file search scopes',()=>fixture(async inspector=>{
 for(const path of paths){const result=await inspector.search({path,query:path.endsWith('expr_test.go')?'go:build':'strconv'});assert.equal(result.complete,true);assert.equal(result.files_scanned,1);assert.equal(result.matches.length,1);assert.equal(result.matches[0].path,path);}
}));
test('file, directory and root queries agree; metacharacters remain literal',()=>fixture(async inspector=>{
 const file=await inspector.search({path:paths[0],query:'literal.*'}),dir=await inspector.search({path:'project/builtin',query:'literal.*'});
 assert.deepEqual(file.matches,dir.matches.filter(x=>x.path===paths[0]));
 const root=await inspector.search({query:'strconv'}),dot=await inspector.search({path:'.',query:'strconv'});
 assert.equal(root.matches.length,6);assert.deepEqual(root.matches,dot.matches);
 assert.equal((await inspector.search({path:paths[0],query:'missing'})).complete,true);
}));
test('file scope keeps traversal, metadata, symlink and FIFO boundary closed',()=>fixture(async(inspector,cwd)=>{
 for(const path of ['/etc/passwd','../escape','.env','project/../expr_test.go','project\\expr_test.go','project//expr_test.go'])await assert.rejects(inspector.search({path,query:'x'}),{code:'INVALID_PATH'});
 await symlink(join(cwd,paths[0]),join(cwd,'alias'));await symlink(join(cwd,'project'),join(cwd,'linked'));
 for(const path of ['alias','linked/expr_test.go'])await assert.rejects(inspector.search({path,query:'x'}),{code:'UNTRUSTED_PATH'});
 execFileSync('mkfifo',[join(cwd,'pipe')]);await assert.rejects(inspector.search({path:'pipe',query:'x'}),{code:'INVALID_PATH'});
 // Files reached indirectly in traversal are still opened with O_NOFOLLOW.
 const result=await inspector.search({query:'strconv'});assert.equal(result.matches.length,6);
}));
test('binary and invalid UTF-8 fail while large text remains searchable',()=>fixture(async(inspector,cwd)=>{
 for(const [name,bytes,code] of [['binary',Buffer.from([0,1]),'NOT_TEXT'],['invalid',Buffer.from([255,254]),'NOT_TEXT']]){
  await writeFile(join(cwd,name),bytes);await assert.rejects(inspector.search({path:name,query:'x'}),{code});
 }
 await writeFile(join(cwd,'large'), 'A'.repeat(65537) + 'needle');
 assert.equal((await inspector.search({path:'large',query:'needle'})).matches.length,1);
 assert.equal((await inspector.search({query:'strconv'})).matches.length,6);
}));
test('cancel and match limit remain explicit for direct files',()=>fixture(async inspector=>{
 await assert.rejects(inspector.search({path:paths[0],query:'x'},AbortSignal.abort()),{code:'CANCELLED'});
 const bounded=await inspector.search({path:paths[0],query:'strconv',max_results:1});assert.equal(bounded.complete,false);assert.equal(bounded.limit_reason,'MATCHES');
 assert.match(repositoryReadTools('/tmp').find(x=>x.name==='voder_search_code').description,/file OR/);
}));
