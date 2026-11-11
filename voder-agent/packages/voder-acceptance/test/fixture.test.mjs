import {test} from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {moduleRoot,sha,readJson,processRun} from '../src/common.mjs';
const suite=await readJson(join(moduleRoot,'fixtures/acceptance-tasks.json'));
const validated=['RT01_CLAMP_EDGE_CASES','RT02_PARSE_AND_REGRESSION','RT03_IMPOSSIBLE_VALIDATOR'];
for(const item of suite.tasks){
 test('frozen task '+item.id+' fails before repair; controls never silently pass',async()=>{
  assert.ok(validated.includes(item.id));assert.equal(item.target,'src/logic.py');
  assert.equal(suite.tasks.filter(x=>x.id===item.id).length,1);
  const tmp=await mkdtemp(join(tmpdir(),'voder-m8-fixture-'));
  try{
   await mkdir(join(tmp,'src'));await mkdir(join(tmp,'tests'));
   await writeFile(join(tmp,'src/__init__.py'),'');await writeFile(join(tmp,'src/logic.py'),item.source);
   await writeFile(join(tmp,'tests/test_intent.py'),item.tests);
   const env={...process.env,PYTHONDONTWRITEBYTECODE:'1'};
   const before=await processRun('python3',['-m','unittest','discover','-s','tests','-v'],{cwd:tmp,env});
   assert.notEqual(before.exit_code,0,'the unmodified fixture must fail real tests');
   if(item.expectation==='DELIVERED'){
    const repaired=item.id==='RT01_CLAMP_EDGE_CASES'
      ?item.source.replace('return min(lower, max(upper, value))','return max(lower, min(upper, value))')
      :`def parse_pairs(text):\n    if not text.strip():\n        return {}\n    result = {}\n    for part in text.split(','):\n        if '=' not in part:\n            raise ValueError('missing =')\n        key, value = part.split('=', 1)\n        result[key.strip()] = value.strip()\n    return result\n`;
    assert.notEqual(repaired,item.source);
    await writeFile(join(tmp,'src/logic.py'),repaired);
    const after=await processRun('python3',['-m','unittest','discover','-s','tests','-v'],{cwd:tmp,env});
    assert.equal(after.exit_code,0,after.stderr.slice(-1500));
   }else{
    assert.equal(item.expectation,'NOT_DELIVERED');
    assert.match(item.tests,/assertEqual\(stable\(\), 1\)/);
    assert.match(item.tests,/assertEqual\(stable\(\), 2\)/);
   }
  }finally{await rm(tmp,{recursive:true,force:true})}
 });
}
test('none of task prompts can change validation program or test scope',()=>{
 assert.equal(suite.tasks.length,3);
 assert.equal(new Set(suite.tasks.map(x=>x.id)).size,3);
 assert.ok(suite.tasks.every(x=>x.maxActions>0&&x.maxActions<=5));
 assert.ok(suite.tasks.every(x=>x.goal.includes('src/logic.py')));
});
