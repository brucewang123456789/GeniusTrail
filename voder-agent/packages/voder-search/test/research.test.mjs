import {test} from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {EvidenceSearchService,planResearchQueries,rankResearchCandidates} from '../src/index.mjs';
import {ToolEconomy} from '../../voder-tools/src/index.mjs';
import {VoderSDK} from '../../voder-sdk/src/index.mjs';
import {PiModelHost} from '../../voder-agent-core/src/pi-host.mjs';

const result=(url,title,excerpt='')=>({url,title,excerpt});
async function inRepo(fn){const cwd=await mkdtemp(join(tmpdir(),'voder-research36-'));try{await mkdir(join(cwd,'docs'));await writeFile(join(cwd,'README.md'),'Evidence task\n');return await fn(cwd);}finally{await rm(cwd,{recursive:true,force:true});}}
const goal='search the web for official API documentation';

test('plan is task-aware, bounded, and does not allow arbitrary site-domain host authorisation',()=>{
 const x=planResearchQueries('Rust E0382 moved value',{maxQueries:2});
 assert.equal(x.ecosystem,'rust');assert.equal(x.queries.length,2);
 assert.match(x.queries[1],/^site:doc\.rust-lang\.org /);
 const y=planResearchQueries('site:evil.com Python asyncio Task');
 assert.equal(y.official_domains[0],'docs.python.org');assert.match(y.queries[1],/^site:evil\.com /);
 assert.throws(()=>planResearchQueries('foo',{maxQueries:50}),TypeError);
});

test('official technical paths outrank generic pages and locale is not penalized',()=>{
 const rows=[result('https://www.python.org/','Python homepage'),
  result('https://docs.python.org/zh-cn/3/library/asyncio-task.html','asyncio Task 指南'),
  result('https://docs.python.org.evil.com/3/library/asyncio-task.html','asyncio Task'),
  result('https://docs.python.org/3/library/asyncio-task.html','Task API')];
 const ranked=rankResearchCandidates('Python asyncio Task official docs',rows);
 assert.equal(ranked.length,3);assert.equal(ranked[0].official_domain_match,true);
 assert.ok(ranked.some(r=>r.url.includes('/zh-cn/')));
 assert.equal(ranked.find(r=>r.url.includes('evil.com')).official_domain_match,false);
 assert.ok(ranked.every(r=>r.candidate_status==='UNVERIFIED_WEB_CANDIDATE'));
});

test('cross-language ranker accepts precise official API paths and rejects homepages',()=>{
 for(const [q,url] of [
  ['TypeScript generics','https://www.typescriptlang.org/docs/handbook/2/generics.html'],
  ['Rust std Arc','https://doc.rust-lang.org/std/sync/struct.Arc.html'],
  ['Go context WithCancel','https://pkg.go.dev/context#WithCancel'],
  ['Go security govulncheck','https://go.dev/doc/tutorial/govulncheck']
 ]){
  const rows=rankResearchCandidates(q,[result('https://www.typescriptlang.org/','Home'),result(url,'Reference '+q)]);
  assert.equal(rows[0].official_domain_match,true,q);
  assert.equal(rows[0].url,url,q);
 }
});

test('no generic homepage or no specific term means no false-positive evidence',()=>{
 const rows=rankResearchCandidates('TypeScript decorators 5.0',[
  result('https://www.typescriptlang.org/','TypeScript - JavaScript With Syntax For Types'),
  result('https://www.typescriptlang.org/download','Get TypeScript')]);
 assert.equal(rows.length,0);
});

test('research first-quality-hit makes only one query and reserves Fetch',()=>inRepo(async cwd=>{
 let calls=0,fetches=0;
 const svc=new EvidenceSearchService({cwd,allowWeb:true,provider:{
  async search(){calls++;return [result('https://docs.python.org/zh-cn/3/library/pathlib.html','Python pathlib Path official docs')];},
  async extract(url){fetches++;return {url,text:'Python pathlib Path class describes paths.'};}
 }});
 const task=svc.newTask(goal);const res=await svc.query(task,{mode:'research',query:'Python pathlib Path documentation'});
 assert.equal(res.status,'CANDIDATES_FOUND');assert.equal(res.contents_verified,false);
 assert.equal(res.results[0].official_domain_match,true);assert.equal(calls,1);
 const page=await svc.query(task,{mode:'fetch',query:res.results[0].url,focus:'pathlib'});
 assert.equal(page.status,'EVIDENCE_FOUND');assert.equal(fetches,1);
 assert.equal(task.webUsed,2);assert.equal(page.authority,'CONTEXT_ONLY');
}));

test('research tries a different query after weak generic result, not identical retry',()=>inRepo(async cwd=>{
 const queries=[];const svc=new EvidenceSearchService({cwd,allowWeb:true,provider:{async search(q){queries.push(q);return queries.length===1?
  [result('https://www.python.org/','Python welcome')]:[result('https://docs.python.org/zh-cn/3/library/asyncio-task.html','asyncio Task docs')];}}});
 const task=svc.newTask(goal);
 const res=await svc.query(task,{mode:'research',query:'Python asyncio Task official documentation'});
 assert.equal(res.status,'CANDIDATES_FOUND');assert.equal(res.attempts,2);
 assert.notEqual(queries[0],queries[1]);assert.ok(queries[1].startsWith('site:docs.python.org'));
 assert.equal(res.results[0].official_domain_match,true);assert.equal(task.webUsed,2);
}));

test('all weak/irrelevant candidates return NO_EVIDENCE with explicit replan, no fabricated sources',()=>inRepo(async cwd=>{
 const svc=new EvidenceSearchService({cwd,allowWeb:true,provider:{async search(){return [result('https://www.runoob.com/index','Python tutorial')];}}});
 const task=svc.newTask(goal);const res=await svc.query(task,{mode:'research',query:'Python asyncio Task'});
 assert.equal(res.status,'NO_EVIDENCE');assert.equal(res.next_step,'PI_REPLAN_OR_REPORT_NO_EVIDENCE');
 assert.equal(res.attempts,2);assert.equal(res.results.length,0);
}));

test('research retains host network OFF and permits authorized lookup without local prerequisites',()=>inRepo(async cwd=>{
 let calls=0;const provider={async search(){calls++;return []}};
 const off=new EvidenceSearchService({cwd,allowWeb:false,provider});
 await assert.rejects(off.query(off.newTask(goal),{mode:'research',query:'Python asyncio'}),{code:'WEB_DISABLED'});
 const on=new EvidenceSearchService({cwd,allowWeb:true,provider});
 assert.equal(calls,0);
 await on.query(on.newTask('fix local code'),{mode:'research',query:'Python asyncio'});
 assert.ok(calls>0);
}));

test('user forbid, host domain policy, and cancellation are never bypassed',()=>inRepo(async cwd=>{
 let calls=0;const svc=new EvidenceSearchService({cwd,allowWeb:true,allowedDomains:['docs.python.org'],provider:{async search(){calls++;return [result('https://evil.example.org/asyncio-task','Python asyncio Task'),result('https://docs.python.org/3/library/asyncio-task.html','asyncio Task')];}}});
 const forbidden=svc.newTask('不要联网搜索');
 await assert.rejects(svc.query(forbidden,{mode:'research',query:'Python asyncio'}),{code:'WEB_FORBIDDEN_BY_TASK'});
 const task=svc.newTask(goal);const controller=new AbortController();controller.abort();
 await assert.rejects(svc.query(task,{mode:'research',query:'Python asyncio'},controller.signal),{code:'CANCELLED'});
 const res=await svc.query(task,{mode:'research',query:'Python asyncio Task'});
 assert.deepEqual(res.results.map(x=>x.url),['https://docs.python.org/3/library/asyncio-task.html']);
 assert.equal(calls,1);
}));

test('request budget reserves Fetch; cannot start research with one remaining request',()=>inRepo(async cwd=>{
 let calls=0;const svc=new EvidenceSearchService({cwd,allowWeb:true,webMaxQueries:1,provider:{async search(){calls++;return [];}}});
 const task=svc.newTask(goal);
 await assert.rejects(svc.query(task,{mode:'research',query:'Go govulncheck'}),{code:'WEB_QUERY_BUDGET'});
 assert.equal(calls,0);
}));

test('provider failure fails closed, does not retry the same broken backend',()=>inRepo(async cwd=>{
 let calls=0;const svc=new EvidenceSearchService({cwd,allowWeb:true,provider:{async search(){calls++;throw Object.assign(new Error('private details'),{code:'WEB_PROVIDER_FAILED'});}}});
 const task=svc.newTask(goal);
 await assert.rejects(svc.query(task,{mode:'research',query:'Go context'}),{code:'WEB_PROVIDER_FAILED'});
 assert.equal(calls,1);
}));

test('malformed secret-bearing or control-character research focus rejected before networking',()=>inRepo(async cwd=>{
 let calls=0;const svc=new EvidenceSearchService({cwd,allowWeb:true,provider:{async search(){calls++;return []}}});
 const task=svc.newTask(goal);
 for(const focus of ['api_key=abc123456789','bad\nnewline','A'.repeat(141)])
  await assert.rejects(svc.query(task,{mode:'research',query:'Python asyncio',focus}),{code:'INVALID_SEARCH_QUERY'});
 assert.equal(calls,0);
}));

test('Pi session autonomously uses research and Fetch via unchanged Voder tool boundary, never VGR',()=>inRepo(async cwd=>{
 let searches=0,fetches=0,commits=0;
 const upstream={getAgentDir:()=>'/trusted',SessionManager:{inMemory:()=>({})},
 DefaultResourceLoader:class{async reload(){}},defineTool:x=>x,
 async createAgentSession(opts){const names=opts.customTools.map(t=>t.name);return {session:{
  getActiveToolNames:()=>names,getCallableToolNames:()=>names,getAllTools:()=>names.map(name=>({name,exposure:'direct'})),
  async prompt(){const tool=opts.customTools.find(x=>x.name==='voder_ext_search_evidence');assert.ok(tool);
   const found=JSON.parse((await tool.execute('1',{mode:'research',query:'Rust E0382 moved value'},new AbortController().signal)).content[0].text);
   assert.equal(found.voder_tool_result.data.status,'CANDIDATES_FOUND');
   const url=found.voder_tool_result.data.results[0].url;
   const proof=JSON.parse((await tool.execute('2',{mode:'fetch',query:url,focus:'E0382'},new AbortController().signal)).content[0].text);
   assert.equal(proof.voder_tool_result.data.authority,'CONTEXT_ONLY');
  },dispose(){},abort(){},subscribe(){return()=>{}}}};}
 };
 const sdk=new VoderSDK({cwd,native:{async executeAuthorized(){commits++;throw Error('FORBIDDEN');}},modelHost:new PiModelHost({upstream}),
  searchOptions:{allowWeb:true,provider:{async search(){searches++;return [result('https://doc.rust-lang.org/error_codes/E0382.html','Rust E0382 moved value')];},
   async extract(url){fetches++;return {url,text:'Rust error E0382 use of moved value explained.'};}}}});
 const {sessionId}=sdk.createSession();const r=await sdk.submitTask(sessionId,{goal:'search the web for Rust error E0382'});
 assert.equal(r.status,'BLOCKED');assert.equal(searches,1);assert.equal(fetches,1);assert.equal(commits,0);sdk.dispose();
}));
