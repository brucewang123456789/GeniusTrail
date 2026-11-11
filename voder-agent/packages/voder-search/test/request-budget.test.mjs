import {test} from 'node:test';
import assert from 'node:assert/strict';
import {EvidenceSearchService} from '../src/index.mjs';
import {normalizeBudget,selectBudget} from '../src/request-budget.mjs';
import {normalizeSearchConfig} from '../src/search-config.mjs';
const url='https://docs.python.org/3/library/asyncio.html';
const hit={url,title:'Python asyncio documentation',excerpt:'Official Python documentation asynchronous tasks'};
const provider={async search(){return [hit]},async extract(u){return {url:u,text:'Official Python documentation content'}}};
const make=(budgetPolicy,extra={})=>new EvidenceSearchService({cwd:process.cwd(),allowWeb:true,provider,directProvider:provider,budgetPolicy,webCacheTtlMs:0,negativeCacheTtlMs:0,...extra});
test('configured unlimited policy overrides legacy entrypoint value 3; explicit host policy can restrict it',()=>{
 assert.equal(selectBudget({configured:{},legacy:3}).max_requests,null);
 assert.equal(selectBudget({configured:{},explicit:{max_requests:7},legacy:3}).max_requests,7);
 assert.equal(selectBudget({legacy:3}).max_requests,3);
});
test('budget normalization rejects ambiguous, unknown and malformed limits',()=>{
 for(const v of [null,[],{wat:3},{max_requests:-1},{max_requests:Infinity},{max_search_requests:'3'},{task_timeout_ms:0},{task_timeout_ms:2147483648}])assert.throws(()=>normalizeBudget(v),{code:'WEB_CONFIG_INVALID'});
 assert.equal(normalizeBudget({max_fetch_requests:0}).max_fetch_requests,0);
 assert.equal(normalizeBudget({max_requests:100000}).max_requests,100000);
});
test('configuration accepts optional budget without changing legacy credentials or presets',()=>{
 const config={version:1,providers:[{id:'ds',preset:'deepseek',credentials:{api_key:{value:'test-not-real'}}}],budget:{max_requests:null,max_search_requests:25}};
 const before=JSON.stringify(config);const out=normalizeSearchConfig(config);
 assert.equal(out.providers[0].ready,true);assert.equal(JSON.stringify(config),before);
 assert.throws(()=>normalizeSearchConfig({...config,budget:{max_requests:-1}}),{code:'WEB_CONFIG_INVALID'});
});
test('unlimited task completes 24 searches plus four official pages; counts remain exact',async()=>{
 const s=make({}),t=s.newTask('web search official documentation');
 for(let i=0;i<24;i++)await s.query(t,{mode:'web',query:'Python asyncio documentation '+i});
 for(const host of ['docs.python.org','api-docs.deepseek.com','brave.com','modelcontextprotocol.io'])await s.query(t,{mode:'fetch',query:'https://'+host+'/docs'});
 assert.equal(t.webUsed,28);assert.equal(t.webSearchUsed,24);assert.equal(t.webFetchUsed,4);
 assert.equal(s._budgetSnapshot(t).remaining,null);assert.equal(s._budgetSnapshot(t).fetch_reserved,false);
});
test('finite total retains a source-fetch reservation and meters each backend fallback',async()=>{
 const p={...provider,async search(q,{acquireRequest}){acquireRequest();return [hit]}};
 const s=make({max_requests:3},{provider:p}),t=s.newTask('web search');
 await s.query(t,{mode:'web',query:'Python asyncio documentation'});
 await assert.rejects(s.query(t,{mode:'web',query:'Python documentation other'}),{code:'WEB_QUERY_BUDGET'});
 await s.query(t,{mode:'fetch',query:url});assert.equal(t.webUsed,3);
});
test('search and fetch limits are independent; exhausted search still permits fetch',async()=>{
 const s=make({max_search_requests:2,max_fetch_requests:4}),t=s.newTask('web search');
 for(let i=0;i<2;i++)await s.query(t,{mode:'web',query:'Python asyncio documentation '+i});
 await assert.rejects(s.query(t,{mode:'web',query:'third search'}),{code:'WEB_QUERY_BUDGET'});
 for(let i=0;i<4;i++)await s.query(t,{mode:'fetch',query:url});
 await assert.rejects(s.query(t,{mode:'fetch',query:url}),{code:'WEB_QUERY_BUDGET'});
 assert.equal(t.webUsed,6);
});
test('fetch disabled by budget does not reserve a slot or charge blocked calls',async()=>{
 const s=make({max_requests:2,max_fetch_requests:0}),t=s.newTask('web search');
 for(let i=0;i<2;i++)await s.query(t,{mode:'web',query:'Python docs '+i});
 await assert.rejects(s.query(t,{mode:'fetch',query:url}),{code:'WEB_QUERY_BUDGET'});assert.equal(t.webUsed,2);
});
test('concurrent fetches cannot exceed finite limit and retain exact pending counts',async()=>{
 let release;const wait=new Promise(r=>{release=r});
 const s=make({max_fetch_requests:2},{provider:{...provider,async extract(u){await wait;return {url:u,text:'body'}}}}),t=s.newTask('web search');
 // A discovered source selects this injected provider, never the real transport.
 t.sourceUrls.add(url);
 const calls=[s.query(t,{mode:'fetch',query:url}),s.query(t,{mode:'fetch',query:url}),s.query(t,{mode:'fetch',query:url})];
 const done=Promise.allSettled(calls);release();const rows=await done;
 assert.equal(rows.filter(r=>r.status==='fulfilled').length,2);assert.equal(rows[2].reason.code,'WEB_QUERY_BUDGET');
 assert.equal(t.webFetchUsed,2);assert.equal(t.webFetchPending,0);
});
test('task time budget cancels in-flight search and does not count as provider outage',async()=>{
 const s=make({task_timeout_ms:35},{provider:{async search(q,{signal}){return new Promise((resolve,reject)=>signal.addEventListener('abort',()=>reject(Object.assign(new Error(),{code:'CANCELLED'})),{once:true}))}}});
 const t=s.newTask('web search');const keep=setTimeout(()=>{},200);
 try{await assert.rejects(s.query(t,{mode:'web',query:'Python documentation'}),{code:'WEB_QUERY_BUDGET'});}finally{clearTimeout(keep)}
 assert.equal(s.providerFailureCount,0);
});
test('user cancellation remains CANCELLED instead of time-budget exhaustion',async()=>{
 const s=make({task_timeout_ms:1000}),t=s.newTask('web search'),c=new AbortController();c.abort();
 await assert.rejects(s.query(t,{mode:'web',query:'Python docs'},c.signal),{code:'CANCELLED'});assert.equal(t.webUsed,0);
});
test('unlimited budget retains web authorization, task prohibition and SSRF rejection',async()=>{
 const s=make({}),t=s.newTask('web search');
 await assert.rejects(s.query(t,{mode:'fetch',query:'https://127.0.0.1/'}),{code:'UNSAFE_SOURCE_URL'});
 await assert.rejects(s.query(s.newTask('禁止联网'),{mode:'web',query:'Python docs'}),{code:'WEB_FORBIDDEN_BY_TASK'});
 const off=make({},{allowWeb:false});await assert.rejects(off.query(off.newTask('web search'),{mode:'web',query:'Python docs'}),{code:'WEB_DISABLED'});
});

test('metered fallback cannot overdraw search cap or open the provider circuit',async()=>{
 const p={...provider,async search(q,{acquireRequest}){acquireRequest();return [hit]}};
 const s=make({max_search_requests:1},{provider:p}),t=s.newTask('web search');
 await assert.rejects(s.query(t,{mode:'web',query:'Python docs'}),{code:'WEB_QUERY_BUDGET'});
 assert.equal(t.webSearchUsed,1);assert.equal(s.providerFailureCount,0);
 await s.query(t,{mode:'fetch',query:url});assert.equal(t.webFetchUsed,1);
});
test('exhausted fetch cap does not reserve an unusable slot after fetch failure',async()=>{
 const p={...provider,async extract(){throw Object.assign(new Error(),{code:'WEB_ACCESS_DENIED'})}};
 const s=make({max_requests:3,max_fetch_requests:1},{provider:p,directProvider:p}),t=s.newTask('web search');
 await assert.rejects(s.query(t,{mode:'fetch',query:url}),{code:'WEB_ACCESS_DENIED'});
 for(let i=0;i<2;i++)await s.query(t,{mode:'web',query:'Python docs '+i});
 assert.equal(t.webUsed,3);assert.equal(s._budgetSnapshot(t).fetch_reserved,false);
});
