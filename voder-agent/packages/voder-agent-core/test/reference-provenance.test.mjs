import {test} from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,rm,readdir} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import context from '../src/reference-context.cjs';
import {AgentKernel} from '../src/kernel.mjs';
const content='from calc import add\nassert add(2,3)==5\n';
const ref={id:'12345678-1111',source:'attachment',kind:'text',name:'test_calc.py',content,sha256:context.digest(Buffer.from(content))};
async function inEmptyWorkspace(run){const cwd=await mkdtemp(join(tmpdir(),'voder-provenance-'));try{return await run(cwd);}finally{await rm(cwd,{recursive:true,force:true});}}
const broker={native:{async executeAuthorized(){throw Error('Read-only test cannot run native actions');}}};
test('system-level provenance contract is present only when references are supplied',async()=>inEmptyWorkspace(async cwd=>{
 let created,prompt,calls=0;
 const modelHost={async createSession(x){created=x;return {async prompt(x){calls++;prompt=x;return {draft:'Only attachment evidence inspected'};},dispose(){}};}};
 const kernel=new AgentKernel({cwd,broker,allowUnverifiedDraft:true,modelHost});
 const result=await kernel.submit({goal:'Explain only this attached test',referenceContext:[ref]});
 assert.equal(result.status,'DRAFT_UNVERIFIED');assert.equal(calls,1);
 assert.match(created.systemPrompt,/VODER REFERENCE PROVENANCE POLICY/);
 assert.match(created.systemPrompt,/filename does NOT assert/);
 assert.ok(!created.systemPrompt.includes(content));
 assert.match(prompt,/from calc import add/);
 assert.match(prompt,/"source":"attachment"/);
 assert.equal((await readdir(cwd)).length,0);
}));
test('without references the system does not carry attachment-specific prompt budget',async()=>inEmptyWorkspace(async cwd=>{
 let created;
 const modelHost={async createSession(x){created=x;return {async prompt(){return {draft:'Repository observation'};},dispose(){}};}};
 const kernel=new AgentKernel({cwd,broker,allowUnverifiedDraft:true,modelHost});
 await kernel.submit({goal:'Explain this empty workspace'});
 assert.doesNotMatch(created.systemPrompt,/VODER REFERENCE PROVENANCE POLICY/);
}));
test('rendered snapshots retain provenance but cannot be promoted to instructions',()=>{
 const out=context.renderReferences([ref]);
 assert.match(out,/source=attachment is not necessarily a workspace file/);
 assert.match(out,/"name":"test_calc.py"/);
 assert.match(out,/"source":"attachment"/);
 assert.doesNotMatch(context.referencePolicy(),/from calc import add/);
});
test('host requiredFiles still rejects missing real workspace inputs even with same-named reference',async()=>inEmptyWorkspace(async cwd=>{
 let started=0;
 const modelHost={async createSession(){started++;throw Error('No model request expected');}};
 const kernel=new AgentKernel({cwd,broker,allowUnverifiedDraft:true,modelHost});
 const result=await kernel.submit({goal:'Execute requested workspace test',requiredFiles:['test_calc.py'],referenceContext:[ref]});
 assert.equal(result.status,'WAITING_FOR_INPUT');assert.equal(started,0);
}));
