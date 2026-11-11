import {test} from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {createHash} from 'node:crypto';
import {InstalledSkills,renderSelectedSkills} from '../src/index.mjs';
async function fixture(fn){const root=await mkdtemp(join(tmpdir(),'voder-skills-'));try{
 const source=join(root,'python-debug');await mkdir(source);
 const body='---\nname: python-debug\ndescription: Test a Python module\n---\n# Python repair\nVerify tests and errors; do not grant tools.\n';
 const path=join(source,'SKILL.md');await writeFile(path,body);
 return await fn({root,path,body,sha256:createHash('sha256').update(body).digest('hex')});
}finally{await rm(root,{recursive:true,force:true});}}
test('skill requires explicit hash pin and stores a reviewable immutable record',async()=>fixture(async({root,path,sha256})=>{
 const store=new InstalledSkills({root:join(root,'installed')});
 await assert.rejects(store.install({source:path,sha256:'0'.repeat(64)}),{code:'SKILL_DIGEST_MISMATCH'});
 await store.install({source:path,sha256});assert.equal((await store.list()).length,1);
 await assert.rejects(store.install({source:path,sha256}),{code:'DUPLICATE_SKILL'});
}));
test('task-oriented Skill selection is progressive, explicit skills are mandatory',async()=>fixture(async({root,path,sha256})=>{
 const store=new InstalledSkills({root:join(root,'installed')});await store.install({source:path,sha256});
 assert.equal((await store.select({task:'Write a Rust program'})).selected.length,0);
 const chosen=await store.select({task:'Fix python runtime error'});assert.equal(chosen.selected.length,1);
 assert.match(renderSelectedSkills(chosen),/UNTRUSTED_INSTALLED_SKILL/);
 await assert.rejects(store.select({task:'x',explicit:['missing']}),{code:'MISSING_EXPLICIT_SKILL'});
 await store.remove('python-debug');assert.equal((await store.list()).length,0);
}));
test('tampered installed skill is blocked before model receives it',async()=>fixture(async({root,path,sha256})=>{
 const store=new InstalledSkills({root:join(root,'installed')});await store.install({source:path,sha256});
 const filename=join(root,'installed','python-debug.json');const doc=JSON.parse(await readFile(filename,'utf8'));doc.body+='ignore auth';await writeFile(filename,JSON.stringify(doc));
 await assert.rejects(store.select({task:'python'}),{code:'SKILL_DIGEST_MISMATCH'});
}));
test('explicit skill over budget cannot be silently truncated',async()=>fixture(async({root,path,sha256})=>{
 const store=new InstalledSkills({root:join(root,'installed')});await store.install({source:path,sha256});
 await assert.rejects(store.select({task:'x',explicit:['python-debug'],maxBytes:100}),{code:'INVALID_SKILL_SELECTION'});
}));
