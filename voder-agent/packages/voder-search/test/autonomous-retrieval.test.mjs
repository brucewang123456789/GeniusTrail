import {test} from 'node:test';
import assert from 'node:assert/strict';
import {EvidenceSearchService} from '../src/index.mjs';
import {ToolEconomy,HostToolRegistry} from '../../voder-tools/src/index.mjs';
import {EvidenceScheduler} from '../../voder-orchestration/src/index.mjs';
const url='https://docs.python.org/3/library/asyncio.html';
function setup({body='A'.repeat(12000)+' Search QPS: 100; Fetch QPS: 25. '+ 'B'.repeat(9000),budgetPolicy={}}={}){
 let requests=0;const provider={async search(q){requests++;return [{url,title:'Official docs '+q,excerpt:q}]},async extract(u){requests++;return {url:u,text:body}}};
 const service=new EvidenceSearchService({cwd:process.cwd(),allowWeb:true,provider,directProvider:provider,budgetPolicy,webCacheTtlMs:0,negativeCacheTtlMs:0});
 const task=service.newTask('search web official docs'),economy=new ToolEconomy({maxCalls:2,maxReturnBytes:4096}),scheduler=new EvidenceScheduler({maxToolCalls:4,maxReadBytes:1024});
 const tool=service.makeTools({task,toolEconomy:economy,inspectRead:(...args)=>scheduler.inspectRead(...args)})[0];
 return {service,task,economy,scheduler,tool,requests:()=>requests};
}
test('50 distinct evidence calls pass both old tool ceilings and leave code/commit reserve unchanged',async()=>{
 const x=setup();for(let i=0;i<50;i++)await x.tool.execute({mode:'web',query:'Python docs '+i});
 assert.equal(x.requests(),50);assert.equal(x.economy.sharedCalls,0);assert.equal(x.economy.retrievalCalls,50);
 assert.equal(x.scheduler.calls,0);assert.equal(x.scheduler.snapshot().remaining_tool_calls,4);assert.equal(x.scheduler.verifications,0);
 assert.equal(x.scheduler.retrievalNoProgress,0);
});
test('ordinary extension tools still hit their original finite budget',async()=>{
 const x=setup();for(let i=0;i<30;i++)await x.tool.execute({mode:'web',query:'Python docs '+i});
 const t=new HostToolRegistry().register({name:'voder_ext_other',description:'Read',parameters:{type:'object',properties:{query:{type:'string',maxLength:50}},required:['query'],additionalProperties:false},execute:async args=>args}).makeTools({economy:x.economy})[0];
 await t.execute({query:'a'});await t.execute({query:'b'});await assert.rejects(t.execute({query:'c'}),{code:'TOOL_CALL_BUDGET'});
 assert.throws(()=>new HostToolRegistry().register({name:'voder_ext_other',description:'Read',budgetClass:'retrieval',capability:'search',parameters:{type:'object',properties:{},required:[],additionalProperties:false},execute:async()=>({})}),{code:'INVALID_TOOL_REGISTRATION'});
});
test('cached cursor and targeted reads reach the page tail with zero additional network requests',async()=>{
 const x=setup({budgetPolicy:{max_requests:1}});const first=await x.tool.execute({mode:'fetch',query:url});
 const row=first.data.results[0];assert.equal(row.excerpt.length,2000);assert.ok(row.passage.next_cursor);assert.equal(row.passage.truncated,true);
 let cursor=row.passage.next_cursor,tail='';while(cursor){const r=await x.tool.execute({mode:'read',query:url,cursor});tail+=r.data.results[0].excerpt;cursor=r.data.results[0].passage.next_cursor;}
 assert.match(tail,/Search QPS: 100/);assert.equal(x.requests(),1);assert.equal(x.task.webUsed,1);
 const focus=await x.tool.execute({mode:'read',query:url,focus:'QPS',max_chars:1000});assert.match(focus.data.results[0].excerpt,/QPS: 100/);
});
test('cached cursors are bound to source content and URL, never used as filesystem paths',async()=>{
 const x=setup();await x.service.query(x.task,{mode:'fetch',query:url});
 for(const cursor of ['../../etc/passwd','0'.repeat(64)+':2000'])await assert.rejects(x.service.query(x.task,{mode:'read',query:url,cursor}),{code:'INVALID_SEARCH_QUERY'});
 await assert.rejects(x.service.query(x.task,{mode:'read',query:'https://docs.python.org/other',cursor:'x'}),{code:'WEB_SOURCE_NOT_FOUND'});
 await assert.rejects(x.service.query(x.task,{mode:'read',query:'https://127.0.0.1/'}),{code:'UNSAFE_SOURCE_URL'});assert.equal(x.requests(),1);
});
test('cached page windows preserve surrogate pairs and bounded metadata',async()=>{
 const x=setup({body:'😀'.repeat(6000)}),r=await x.service.query(x.task,{mode:'fetch',query:url,max_chars:1001});
 assert.equal(r.results[0].excerpt.length,1000);assert.equal(r.results[0].passage.end,1000);
 const next=await x.service.query(x.task,{mode:'read',query:url,cursor:r.results[0].passage.next_cursor,max_chars:1001});assert.equal(next.results[0].passage.start,1000);
 assert.ok(Buffer.byteLength(JSON.stringify(next))<32768);
});
test('repeated empty or identical evidence recommends finishing with gaps instead of spending code reserve',async()=>{
 const s=new EvidenceScheduler({maxNoProgress:3});
 for(let i=0;i<3;i++)await s.inspectRead('voder_ext_search_evidence',{mode:'web',query:'empty '+i},async()=>({data:{results:[]}}));
 await assert.rejects(s.inspectRead('voder_ext_search_evidence',{mode:'web',query:'empty 4'},async()=>({})),{code:'RETRIEVAL_NO_PROGRESS'});
 assert.equal(s.calls,0);assert.equal(s.noProgress,0);assert.equal(s.phase,'EXPLORING');
});
test('duplicate pending/successful evidence calls remain blocked and user cancellation is preserved',async()=>{
 const x=setup();await x.tool.execute({mode:'web',query:'Python docs'});await assert.rejects(x.tool.execute({mode:'web',query:'Python docs'}),{code:'DUPLICATE_NO_PROGRESS'});
 const c=new AbortController();c.abort();await assert.rejects(x.tool.execute({mode:'web',query:'different'},c.signal),{code:'CANCELLED'});
 assert.equal(x.requests(),1);
});
test('retrieval remains on demand and user-configured finite request limits still apply',async()=>{
 const x=setup({budgetPolicy:{max_search_requests:1}});assert.equal(x.requests(),0);await x.tool.execute({mode:'web',query:'Python docs'});
 await assert.rejects(x.tool.execute({mode:'web',query:'another'}),{code:'WEB_QUERY_BUDGET'});assert.equal(x.requests(),1);
 assert.equal(x.scheduler.calls,0);
});

test('tool fetch reuses cached source for new focus and can refetch after cache eviction',async()=>{
 const x=setup();const first=await x.tool.execute({mode:'fetch',query:url});
 const repeat=await x.tool.execute({mode:'fetch',query:url});assert.equal(repeat.data.cache_hit,true);assert.equal(x.requests(),1);
 x.task.documents.delete(url);await x.tool.execute({mode:'fetch',query:url});assert.equal(x.requests(),2);
});
