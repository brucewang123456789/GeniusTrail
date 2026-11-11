import {test} from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,rm} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {VoderSDK,VoderControlPlane,VoderJournal} from '../src/index.mjs';

test('durable SDK result retains real blocker, phase and counts without source, prompts or recovery advice',async()=>{
 const cwd=await mkdtemp(join(tmpdir(),'voder-budget-journal-'));let host;
 try{
  await mkdir(join(cwd,'src'));await writeFile(join(cwd,'src/a.txt'),'PRIVATE_SOURCE_CANARY_2026');
  const sdk=new VoderSDK({cwd,native:{async executeAuthorized(){throw Error('No native calls allowed in this offline test');}},orchestrationOptions:{maxToolCalls:4},modelHost:{async createSession({tools,toolControl}){return {async prompt(){
   await tools.find(t=>t.name==='voder_read_file').execute({path:'src/a.txt'});
   await tools.find(t=>t.name==='voder_list_files').execute({path:'src'});
   assert.equal(toolControl().phase,'FINALIZING');
   throw Object.assign(Error('PRIVATE_FAILURE_MESSAGE'),{code:'FINALIZATION_STALLED'});
  },dispose(){},async abort(){}};}}});
  const dir=join(cwd,'private'),journal=await new VoderJournal({dir,workspace:cwd}).initialize();host=new VoderControlPlane({sdk,journal});
  const {sessionId}=await host.createSession();await host.submitTask(sessionId,{goal:'PRIVATE_TASK_CANARY_2026'});const result=await host.wait(sessionId);
  assert.equal(result.status,'BLOCKED');assert.equal(result.blocked_reason,'FINALIZATION_STALLED');assert.equal(result.verified,false);
  assert.equal(result.orchestration.tool_calls,2);assert.equal(result.orchestration.remaining_tool_calls,2);assert.equal(result.orchestration.phase,'FINALIZING');
  const stored=await new VoderJournal({dir,workspace:cwd}).get(sessionId);
  assert.deepEqual(stored.result.orchestration,result.orchestration);
  const settled=stored.events.find(e=>e.type==='task.settled');assert.equal(settled.reason,'FINALIZATION_STALLED');assert.equal(settled.orchestration.verification_calls,0);
  assert.ok(stored.events.some(e=>e.type==='orchestration.phase'&&e.orchestration.phase==='FINALIZING'));
  const raw=await readFile(join(dir,'sessions',sessionId,'snapshot.json'),'utf8');
  for(const secret of ['PRIVATE_SOURCE_CANARY_2026','PRIVATE_TASK_CANARY_2026','PRIVATE_FAILURE_MESSAGE','next_step','new_content'])assert.ok(!raw.includes(secret),secret);
 }finally{host?.dispose();await rm(cwd,{recursive:true,force:true});}
});
