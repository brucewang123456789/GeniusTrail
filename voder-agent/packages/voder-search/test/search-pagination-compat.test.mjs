import {test} from 'node:test';
import assert from 'node:assert/strict';
import {EvidenceSearchService} from '../src/index.mjs';
import {ToolEconomy} from '../../voder-tools/src/index.mjs';
const url='https://docs.python.org/3/library/asyncio.html';
function make(){let calls=0;const provider={async search(query){calls++;return [{url,title:'Rust reqwest documentation',excerpt:'Official Rust reqwest docs '+query}]},async extract(u){calls++;return {url:u,text:'A'.repeat(6000)}}};
 const service=new EvidenceSearchService({cwd:process.cwd(),allowWeb:true,provider,directProvider:provider,budgetPolicy:{},webCacheTtlMs:0});
 const task=service.newTask('web search official Rust docs');return {service,task,get calls(){return calls},tool:service.makeTools({task,toolEconomy:new ToolEconomy()})[0]};}
test('real host tool accepts discovery with irrelevant optional pagination fields',async()=>{
 const x=make();for(const [i,extra]of [{},{max_chars:2000},{cursor:''},{max_chars:6000,cursor:'previous-cursor'}].entries()){
  const out=await x.tool.execute({mode:'web',query:'rust reqwest '+i,...extra});assert.equal(out.data.results[0].url,url);
 }assert.equal(x.calls,4);
});
test('research tolerates auto-populated passage defaults without weakening query or URL validation',async()=>{
 const x=make();const out=await x.tool.execute({mode:'research',query:'rust reqwest',max_chars:2000,cursor:''});assert.ok(out.data.results.length);
 await assert.rejects(x.tool.execute({mode:'web',query:'',max_chars:2000}),{code:'INVALID_SEARCH_QUERY'});
 await assert.rejects(x.tool.execute({mode:'fetch',query:'https://127.0.0.1/',cursor:''}),{code:'UNSAFE_SOURCE_URL'});
});
test('empty optional cursor is absent for fetch/read while real cursor validation stays strict',async()=>{
 const x=make();const first=await x.tool.execute({mode:'fetch',query:url,cursor:'',max_chars:1000});assert.equal(first.data.results[0].excerpt.length,1000);
 const read=await x.tool.execute({mode:'read',query:url,cursor:'',focus:'A',max_chars:1000});assert.equal(read.data.network_requests,0);
 await assert.rejects(x.tool.execute({mode:'read',query:url,cursor:'wrong'}),{code:'INVALID_SEARCH_QUERY'});
 await assert.rejects(x.tool.execute({mode:'fetch',query:url,cursor:'real-nonempty-cursor'}),{code:'INVALID_SEARCH_QUERY'});
 assert.equal(x.calls,1);
});
