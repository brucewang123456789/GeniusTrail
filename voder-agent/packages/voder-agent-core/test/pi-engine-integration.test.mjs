import {test} from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,rm,writeFile,symlink} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {PiModelHost} from '../src/pi-host.mjs';
import {AgentKernel} from '../src/kernel.mjs';
import {RepositoryInspector,repositoryReadTools} from '../src/repository-tools.mjs';

async function fixture(fn){const cwd=await mkdtemp(join(tmpdir(),'voder-pi-engine-'));try{await mkdir(join(cwd,'src'));await writeFile(join(cwd,'src','sample.txt'),'one\nsearch needle\nthree\n');return await fn(cwd);}finally{await rm(cwd,{force:true,recursive:true})}}
const nativeTool={name:'voder_authorized_action',description:'Native patch',async execute(){throw new Error('NO_NATIVE_AUTHORITY')}};
function upstream({extra=[],exposure='direct',onPrompt}={}){
  const api={getAgentDir:()=>'/trusted',SessionManager:{inMemory:()=>({})},DefaultResourceLoader:class{constructor(options){api.loader=options}async reload(){}},defineTool:x=>x,
    createAgentSession:async options=>{
      api.options=options;
      const defs=options.customTools;
      const tools=[...defs.map(x=>x.name),...extra];
      const session={getActiveToolNames:()=>tools,getCallableToolNames:()=>tools,getAllTools:()=>tools.map(name=>({name,exposure})),async prompt(goal){return onPrompt?.(defs,goal)},dispose(){},subscribe(){return ()=>{}},abort(){}};
      return {session};
    }};
  return api;
}
test('upstream Pi receives only four explicit Voder tools; default + dynamic extension tools suppressed',async()=>fixture(async cwd=>{
 const fake=upstream();const host=new PiModelHost({upstream:fake});const session=await host.createSession({cwd,systemPrompt:'trusted prompt',tools:[nativeTool,...repositoryReadTools(cwd)]});
 assert.deepEqual(session.inspectTools().active,['voder_authorized_action','voder_read_file','voder_list_files','voder_search_code']);
 assert.equal(fake.options.noTools,'all');assert.equal(fake.loader.noExtensions,true);assert.equal(fake.loader.noSkills,true);assert.equal(fake.loader.noContextFiles,true);
 assert.deepEqual(fake.options.tools,['voder_authorized_action','voder_read_file','voder_list_files','voder_search_code']);
 assert.equal(fake.options.customTools.find(t=>t.name==='voder_authorized_action').parameters.properties.action.additionalProperties,false);
 session.dispose();
}));
test('Pi tool response goes back into the model cycle as content, no host command',async()=>fixture(async cwd=>{
 let seen=[];const fake=upstream({onPrompt:async defs=>{for(const n of ['voder_list_files','voder_search_code','voder_read_file']){
   const d=defs.find(x=>x.name===n);
   const args=n==='voder_list_files'?{path:'src'}:n==='voder_search_code'?{query:'needle'}:{path:'src/sample.txt'};
   const output=await d.execute('id',args,new AbortController().signal);
   assert.equal(output.details.status,'READ_OK');seen.push(JSON.parse(output.content[0].text).voder_tool_result);
 }}});
 const session=await new PiModelHost({upstream:fake}).createSession({cwd,systemPrompt:'trusted prompt',tools:[nativeTool,...repositoryReadTools(cwd)]});
 await session.prompt('inspect');assert.equal(seen.length,3);assert.equal(seen[0].entries[0].name,'sample.txt');
 assert.equal(seen[1].matches[0].line,2);assert.match(seen[2].content,/needle/);session.dispose();
}));
test('Pi session refuses ambient deferred tool even if not exposed to model',async()=>fixture(async cwd=>{
 const fake=upstream({extra:['mcp__rogue']});await assert.rejects(new PiModelHost({upstream:fake}).createSession({cwd,systemPrompt:'p',tools:[nativeTool,...repositoryReadTools(cwd)]}),{code:'UNSAFE_TOOLS'});
}));
test('Pi session refuses non-direct tool exposure',async()=>fixture(async cwd=>{
 const fake=upstream({exposure:'deferred'});await assert.rejects(new PiModelHost({upstream:fake}).createSession({cwd,systemPrompt:'p',tools:[nativeTool,...repositoryReadTools(cwd)]}),{code:'UNSAFE_TOOLS'});
}));
test('direct model declaration cannot request Bash or arbitrary tool by name',async()=>fixture(async cwd=>{
 const fake=upstream();await assert.rejects(new PiModelHost({upstream:fake}).createSession({cwd,systemPrompt:'p',tools:[nativeTool,{name:'bash',description:'no',execute(){}}]}),{code:'INVALID_TOOL_SET'});
}));
test('repository reader denies file traversal and metadata access',async()=>fixture(async cwd=>{
 const repo=new RepositoryInspector(cwd);for(const path of ['../etc/passwd','/etc/passwd','src/.env','src/../../etc/passwd','src\\sample.txt','.git/config']){
   await assert.rejects(repo.read({path}),{code:'INVALID_PATH'});
 }
}));
test('repository reader denies symlinked file and symlinked parent',async()=>fixture(async cwd=>{
 await symlink('/etc/passwd',join(cwd,'src','leak'));await symlink('/etc',join(cwd,'escape'));
 const repo=new RepositoryInspector(cwd);
 await assert.rejects(repo.read({path:'src/leak'}),{code:'UNTRUSTED_PATH'});
 await assert.rejects(repo.read({path:'escape/passwd'}),{code:'UNTRUSTED_PATH'});
 const listing=await repo.list({path:'src'});assert.ok(!listing.entries.some(x=>x.name==='leak'));
}));
test('repo search bounded and no symlink-derived content',async()=>fixture(async cwd=>{
 const repo=new RepositoryInspector(cwd);const result=await repo.search({query:'needle',max_results:1});
 assert.equal(result.matches[0].path,'src/sample.txt');assert.equal(result.truncated,true);
 await assert.rejects(repo.search({query:'',max_results:40}),{code:'INVALID_QUERY'});
 await assert.rejects(repo.search({query:'needle',max_results:41}),{code:'INVALID_BUDGET'});
}));
test('native tool error does not become a fabricated verified receipt',async()=>fixture(async cwd=>{
 const fake=upstream({onPrompt:async defs=>{
   const r=await defs.find(x=>x.name==='voder_authorized_action').execute('call1',{action:{kind:'patch',relative_path:'src/sample.txt',old_sha256:'0'.repeat(64),new_content:'new'}},new AbortController().signal);
   assert.equal(r.details.status,'DENIED');assert.ok(!r.content[0].text.includes('COMMITTED'));
 }});
 const host=new PiModelHost({upstream:fake});const kernel=new AgentKernel({modelHost:host,broker:{native:{},dispatch(){throw Error('native not available')}},cwd,sessionId:'task-1'});
 const result=await kernel.submit({goal:'repair'});assert.equal(result.status,'BLOCKED');
 assert.ok(kernel.events.some(x=>x.type==='vgr.action_requested'));kernel.close();
}));
test('new kernel task does not reuse prior model session, goals or tools',async()=>fixture(async cwd=>{
 let created=0,disposed=0;
 const host={async createSession(){created++;return {async prompt(){},dispose(){disposed++}}}};
 const kernel=new AgentKernel({cwd,modelHost:host,broker:{},sessionId:'task-1'});
 assert.equal((await kernel.submit({goal:'first'})).status,'BLOCKED');
 assert.equal((await kernel.submit({goal:'second'})).status,'BLOCKED');
 assert.equal(created,2);assert.equal(disposed,1);kernel.close();assert.equal(disposed,2);
}));
test('cancelled session cannot issue side effects',async()=>fixture(async cwd=>{
 let tools;
 const kernel=new AgentKernel({cwd,modelHost:{async createSession(o){tools=o.tools;return {async prompt(){await kernel.cancel();},async abort(){},dispose(){}}}},broker:{dispatch(){throw Error('must not execute')}},sessionId:'task-1'});
 const result=await kernel.submit({goal:'cancel'});assert.equal(result.status,'CANCELLED');
 await assert.rejects(tools[0].execute({kind:'patch',relative_path:'src/sample.txt',old_sha256:'0'.repeat(64),new_content:'x'}),{code:'CANCELLED'});
}));

test('stale task callback cannot use a later task action budget',async()=>fixture(async cwd=>{
 const calls=[];
 const host={async createSession({tools}){calls.push(tools[0].execute);return {async prompt(){},dispose(){}}}};
 const kernel=new AgentKernel({cwd,modelHost:host,broker:{async dispatch(){throw Error('must not execute')}},sessionId:'task-1'});
 await kernel.submit({goal:'original task'});await kernel.submit({goal:'different task'});
 await assert.rejects(calls[0]({kind:'patch',relative_path:'src/sample.txt',old_sha256:'0'.repeat(64),new_content:'x'}),{code:'STALE_TASK'});
 kernel.close();
}));

test('multi-step model attempt only settles as verified after native authenticated commit',async()=>fixture(async cwd=>{
 let attempts=0;let commitChecks=0;let feedback=[];
 const native={async executeAuthorized(a){attempts++;return {authority:'VODER_NATIVE_VGR',
   session_id:a.session_id,action_id:a.action_id,run_id:`run-${a.action_id}`,
   status:attempts===1?'OBSERVED':'COMMITTED',journal_digest:'a'.repeat(64),
   ...(attempts===2?{commit_digest:'b'.repeat(64)}:{})};}};
 const commitVerifier={async confirmCommit(receipt,input){commitChecks++;return {
   verified:true,authority:'VODER_NATIVE_VGR_STATE',session_id:input.session_id,
   action_id:input.action_id,run_id:receipt.run_id,commit_digest:receipt.commit_digest};}};
 const {VgrBroker}=await import('../src/broker.mjs');
 const broker=new VgrBroker({native,sessionId:'s1',workspaceRoot:cwd,commitVerifier});
 const fake=upstream({onPrompt:async defs=>{
   const read=defs.find(x=>x.name==='voder_read_file');
   feedback.push(await read.execute('r',{path:'src/sample.txt'},new AbortController().signal));
   const action=defs.find(x=>x.name==='voder_authorized_action');
   for(const content of ['candidate1','candidate2']){
     const raw=await action.execute('tool',{action:{kind:'patch',relative_path:'src/sample.txt',old_sha256:'a'.repeat(64),new_content:content}},new AbortController().signal);
     feedback.push(raw);
   }
 }});
 const kernel=new AgentKernel({modelHost:new PiModelHost({upstream:fake}),broker,cwd,sessionId:'s1'});
 const result=await kernel.submit({goal:'repair',maxActions:2});
 assert.equal(result.status,'VERIFIED');assert.equal(commitChecks,1);assert.equal(attempts,2);assert.equal(feedback.length,3);
 assert.equal(kernel.events.filter(x=>x.type==='vgr.action_requested').length,2);
 kernel.close();
}));
