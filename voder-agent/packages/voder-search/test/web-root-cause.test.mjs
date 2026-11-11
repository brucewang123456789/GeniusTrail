import {test} from 'node:test';
import assert from 'node:assert/strict';
import {EvidenceSearchService} from '../src/index.mjs';
import {RoutedPublicProvider} from '../src/public-provider.mjs';
import {ToolEconomy} from '../../voder-tools/src/index.mjs';
import {VoderSDK} from '../../voder-sdk/src/index.mjs';
import {mkdtemp,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
const url='https://www.nobelprize.org/prizes/physics/2026/summary/';
const fail=code=>Object.assign(Error(code),{code});
test('original Chinese query recovers after premature fetch and reaches search and fetch',async()=>{
 let searches=0,fetches=0;
 const svc=new EvidenceSearchService({cwd:process.cwd(),allowWeb:true,provider:{async search(){searches++;return [{title:'Official physics',url,excerpt:'Official evidence'}];},async extract(u){fetches++;return {url:u,text:'Observed official page'};}}});
 const task=svc.newTask('请搜索网页，核实2026年诺贝尔物理学奖得主是谁');
 assert.equal(task.explicitWeb,true);
 const economy=new ToolEconomy();const [tool]=svc.makeTools({task,toolEconomy:economy});
 await assert.rejects(tool.execute({mode:'fetch',query:'https://127.0.0.1/'}),{code:'UNSAFE_SOURCE_URL'});
 const found=await tool.execute({mode:'web',query:'2026 Nobel Prize Physics'});
 assert.equal(found.data.results[0].url,url);
 const page=await tool.execute({mode:'fetch',query:url});
 assert.equal(page.data.results[0].excerpt,'Observed official page');
 assert.equal(searches,1);assert.equal(fetches,1);assert.equal(economy.failures.get(tool.name),0);
});
test('external factual question needs no keyword or local repo miss',async()=>{
 let calls=0;const svc=new EvidenceSearchService({cwd:process.cwd(),allowWeb:true,provider:{async search(){calls++;return [];}}});
 await svc.query(svc.newTask('Who won the 2026 Nobel Prize in Physics?'),{mode:'web',query:'2026 Nobel Physics'});assert.equal(calls,1);
});
test('parallel actual adapter failures are counted without lost updates',async()=>{
 const e=new ToolEconomy();let release;const gate=new Promise(r=>release=r);
 const calls=[1,2].map(n=>e.run('adapter',{n},async()=>{await gate;throw fail('WEB_PROVIDER_FAILED');}));release();await Promise.allSettled(calls);
 assert.equal(e.failures.get('adapter'),2);
 await assert.rejects(e.run('adapter',{n:3},async()=>({})),{code:'TOOL_CIRCUIT_OPEN'});
});
test('policy refusals do not poison adapter or prevent same-input recovery',async()=>{
 const e=new ToolEconomy();const args={mode:'fetch',query:url};
 for(let n=0;n<3;n++)await assert.rejects(e.run('search',args,async()=>{throw fail('SOURCE_NOT_DISCOVERED');}),{code:'SOURCE_NOT_DISCOVERED'});
 assert.equal(e.failures.get('search')??0,0);
 assert.equal((await e.run('search',args,async()=>({ok:true}))).data.ok,true);
 assert.equal(e.snapshot().failed_calls,3);
});
test('router sends general facts to general search and language docs to original official backend',async()=>{
 const seen=[];const backend=n=>({async search(q){seen.push(n);return [];},async extract(u){seen.push(n);return {url:u,text:'page'};}});
 const r=new RoutedPublicProvider({official:backend('official'),web:backend('public')});
 await r.search('Nobel Prize 2026');await r.search('Python pathlib');await r.extract(url);await r.extract('https://docs.python.org/3/library/pathlib.html');
 assert.deepEqual(seen,['public','official','public','official']);
});
test('host and user network denials still produce zero provider calls',async()=>{
 let calls=0;const provider={async search(){calls++;return [];}};
 for(const allowWeb of [false,true]){
  const s=new EvidenceSearchService({cwd:process.cwd(),allowWeb,provider});
  await assert.rejects(s.query(s.newTask('不要搜索网页'),{mode:'web',query:'Nobel Physics'}),{code:'WEB_FORBIDDEN_BY_TASK'});
 }
 const off=new EvidenceSearchService({cwd:process.cwd(),provider});await assert.rejects(off.query(off.newTask('搜索网页'),{mode:'web',query:'Nobel Physics'}),{code:'WEB_DISABLED'});assert.equal(calls,0);
});
test('SDK scheduler accounts for failed extension reads and preserves VGR isolation',async()=>{
 const cwd=await mkdtemp(join(tmpdir(),'web-root-cause-'));let native=0,checked=false;
 const sdk=new VoderSDK({cwd,allowUnverifiedDraft:true,native:{async executeAuthorized(){native++;throw Error('unexpected');}},searchOptions:{allowWeb:true,provider:{async search(){throw fail('WEB_PROVIDER_FAILED');}}},modelHost:{async createSession({tools}){return {async prompt(){const t=tools.find(t=>t.name==='voder_ext_search_evidence');await assert.rejects(t.execute({mode:'web',query:'Nobel Physics'}),{code:'WEB_PROVIDER_FAILED'});checked=true;},dispose(){}};}}});
 try{const s=sdk.createSession();await sdk.submitTask(s.sessionId,{goal:'搜索网页核实诺奖'});const k=sdk.sessions.get(s.sessionId);assert.equal(checked,true);assert.equal(k.scheduler.snapshot().tool_calls,0);assert.equal(k.scheduler.snapshot().retrieval_calls,1);assert.equal(k.scheduler.snapshot().retrieval_no_progress,1);assert.equal(k.toolEconomy.calls,1);assert.equal(k.toolEconomy.sharedCalls,0);assert.equal(native,0);}finally{sdk.dispose();await rm(cwd,{recursive:true,force:true});}
});
