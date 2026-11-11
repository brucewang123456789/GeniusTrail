import {test} from 'node:test';
import assert from 'node:assert/strict';
import {RustWebProvider,runNative} from '../src/rust-web.mjs';
import {RoutedPublicProvider} from '../src/public-provider.mjs';
import {EvidenceSearchService,TavilyWebProvider} from '../src/index.mjs';
const q='site:science.example.org/research nebula spectrum';
const url='https://science.example.org/research/nebula-spectrum';
const row={url,title:'Nebula spectrum observations',excerpt:'Evidence from spectrum measurements'};
const cfg=()=>({providers:['bigmodel','deepseek'],keys:{bigmodel:'test-one',deepseek:'test-two'}});
const err=code=>Object.assign(new Error('secret response body must not escape'),{code});
const make=(runImpl,options={})=>new RustWebProvider({runImpl,configImpl:cfg,...options});
const page=(u=url)=>({url:u,text:'Page text',raw_sha256:'a'.repeat(64)});

test('default general route is Rust; existing official documentation route is preserved',async()=>{
 const p=new RoutedPublicProvider();assert.ok(p.web instanceof RustWebProvider);
 let official=0;const routed=new RoutedPublicProvider({official:{search:async()=>{official++;return[]}},web:make(async()=>({rows:[row]}))});
 await routed.search('python pathlib');assert.equal(official,1);await routed.search(q);assert.equal(official,1);
});
test('failed primary falls back once and charges the SDK budget; a fetch remains reserved',async()=>{
 const calls=[];const web=make(async job=>{calls.push(job);if(job.provider==='bigmodel')throw err('WEB_CONNECT_FAILED');return{rows:[row]}});
 const service=new EvidenceSearchService({cwd:process.cwd(),allowWeb:true,provider:new RoutedPublicProvider({web})});
 const t=service.newTask('web search');const out=await service.query(t,{mode:'web',query:q});
 assert.equal(out.results[0].url,url);assert.equal(t.webUsed,2);assert.equal(out.budget.remaining,1);assert.equal(out.budget.fetch_reserved,true);
 assert.deepEqual(calls.map(x=>x.provider),['bigmodel','deepseek']);assert.ok(calls.every(x=>x.op==='search'));assert.equal(out.discovery_attempts[0].code,'WEB_CONNECT_FAILED');
});
test('without a host request token no hidden fallback is possible',async()=>{
 let calls=0;const p=make(async()=>{calls++;throw err('WEB_CONNECT_FAILED')});
 await assert.rejects(p.search(q),{code:'WEB_CONNECT_FAILED'});assert.equal(calls,1);
});
test('budget refusal never spends reserved fetch on a backup',async()=>{
 let calls=0;const p=make(async()=>{calls++;throw err('WEB_CONNECT_FAILED')});
 const s=new EvidenceSearchService({cwd:process.cwd(),allowWeb:true,webMaxQueries:2,provider:new RoutedPublicProvider({web:p})});const t=s.newTask('search web');
 await assert.rejects(s.query(t,{mode:'web',query:q}),{code:'WEB_CONNECT_FAILED'});assert.equal(calls,1);assert.equal(t.webUsed,1);
});
test('valid empty response differs from outages and backup failure does not turn empty into success',async()=>{
 const p=make(async j=>j.provider==='bigmodel'?{rows:[]}:Promise.reject(err('WEB_TIMEOUT')));
 const out=await p.search(q,{acquireRequest:()=>true});assert.deepEqual(out,[]);assert.equal(out.discovery_attempts[1].status,'FAILED');
 const broken=make(async()=>{throw err('WEB_DNS_FAILED')});await assert.rejects(broken.search(q,{acquireRequest:()=>true}),{code:'WEB_DNS_FAILED'});
});
test('host scopes reject provider URL spoofing and cross-domain results',async()=>{
 const p=make(async()=>({rows:[{...row,url:'https://science.example.org.evil.com/research/nebula-spectrum'},{...row,url:'https://internal.local/'},row]}));
 assert.deepEqual((await p.search(q,{sourceAllowed:u=>new URL(u).hostname==='science.example.org'})).map(r=>r.url),[url]);
});
test('auth cooldown is backend and credential scoped; backup stays available',async()=>{
 let clock=0;const calls=[];let primaryKey='old';const p=make(async j=>{calls.push(j.provider);if(j.provider==='bigmodel')throw err('WEB_AUTH_FAILED');return{rows:[row]}},{now:()=>clock,configImpl:()=>({providers:['bigmodel','deepseek'],keys:{bigmodel:primaryKey,deepseek:'second'}})});
 await p.search(q,{acquireRequest:()=>true});await p.search(q,{acquireRequest:()=>true});assert.deepEqual(calls,['bigmodel','deepseek','deepseek']);
 primaryKey='new';await p.search(q,{acquireRequest:()=>true});assert.deepEqual(calls.slice(-2),['bigmodel','deepseek']);
 clock=61000;await p.search(q,{acquireRequest:()=>true});assert.deepEqual(calls.slice(-2),['bigmodel','deepseek']);
});
test('missing credentials are configuration failure with zero native calls',async()=>{
 let calls=0;const p=make(async()=>{calls++},{configImpl:()=>({providers:['bigmodel'],keys:{}})});
 await assert.rejects(p.search(q),{code:'WEB_CREDENTIAL_MISSING'});assert.equal(calls,0);
});
test('provider query limits never silently truncate the query',async()=>{
 let calls=0;const p=make(async()=>{calls++;return{rows:[]}},{configImpl:()=>({providers:['deepseek'],keys:{deepseek:'test'}})});
 await assert.rejects(p.search('中'.repeat(129)),{code:'INVALID_SEARCH_QUERY'});assert.equal(calls,0);
});
test('cancellation propagates without invoking backup',async()=>{
 const c=new AbortController();let calls=0;const p=make(async()=>{calls++;c.abort();throw err('WEB_CONNECT_FAILED')});
 await assert.rejects(p.search(q,{signal:c.signal,acquireRequest:()=>true}),{code:'CANCELLED'});assert.equal(calls,1);
});
test('redirect domain is checked before its DNS or request',async()=>{
 const calls=[];const p=make(async j=>{calls.push(j.url);return{redirect:'https://evil.example.net/target',status:302}});
 await assert.rejects(p.extract(url,{urlAllowed:u=>new URL(u).hostname==='science.example.org'}),{code:'WEB_DOMAIN_DENIED'});assert.deepEqual(calls,[url]);
});
test('redirect chain preserves original URL, final URL, raw hash and untrusted page provenance',async()=>{
 const target='https://science.example.org/research/final';let calls=0;const p=make(async()=>++calls===1?{redirect:target,status:301}:page(target));
 const out=await p.extract(url);assert.equal(out.url,url);assert.equal(out.final_url,target);assert.equal(out.redirects.length,1);assert.equal(out.evidence_kind,'FETCHED_PAGE');
});
test('redirect loops and oversized chains are refused',async()=>{
 const loop=make(async()=>({redirect:url,status:302}));await assert.rejects(loop.extract(url),{code:'WEB_REDIRECT_LOOP'});
 let calls=0;const p=make(async()=>({redirect:'https://science.example.org/r/'+(++calls),status:302}));await assert.rejects(p.extract(url),{code:'WEB_REDIRECT_LIMIT'});assert.equal(calls,5);
});
test('single wall-clock deadline spans provider fallback and redirect chain',async()=>{
 let clock=0;const p=make(async()=>{clock+=11900;return{redirect:'https://science.example.org/final',status:302}},{now:()=>clock});
 await assert.rejects(p.extract(url),{code:'WEB_TIMEOUT'});
});
test('native messages and provider secrets never become SDK error messages',async()=>{
 const p=make(async()=>{throw err('UNKNOWN_WITH_SECRET')});const s=new EvidenceSearchService({cwd:process.cwd(),allowWeb:true,provider:new RoutedPublicProvider({web:p})});
 await assert.rejects(s.query(s.newTask('web search'),{mode:'web',query:q}),e=>e.code==='WEB_PROVIDER_FAILED'&&!/secret|test-one|test-two/.test(e.message));
});
test('search vendor credentials cannot route page fetch to a vendor extract endpoint',async()=>{
 const provider=new TavilyWebProvider({apiKey:'test',fetchImpl:async()=>{throw new Error('remote fetch forbidden')}});let called=false;
 const s=new EvidenceSearchService({cwd:process.cwd(),allowWeb:true,provider,directProvider:{extract:async u=>{called=true;return page(u)}}});const t=s.newTask('web search');t.sourceUrls.add(url);
 assert.equal((await s.query(t,{mode:'fetch',query:url})).status,'EVIDENCE_FOUND');assert.equal(called,true);
});
test('the built Rust process refuses unsafe URLs before any connection',async()=>{
 for(const u of ['https://127.0.0.1/','http://example.org/','https://example.org/?token=1234'])await assert.rejects(runNative({op:'fetch',url:u,timeout_ms:1000}),{code:'UNSAFE_SOURCE_URL'});
});
test('native pre-cancellation and concurrency backpressure are bounded',async()=>{
 const c=new AbortController();c.abort();await assert.rejects(runNative({op:'fetch',url,timeout_ms:1000},{signal:c.signal}),{code:'CANCELLED'});
 const jobs=Array.from({length:5},()=>runNative({op:'fetch',url:'https://127.0.0.1/',timeout_ms:1000}));const out=await Promise.allSettled(jobs);
 assert.equal(out.filter(x=>x.status==='rejected'&&x.reason.code==='WEB_NATIVE_BUSY').length,1);
});


test('a stalled primary retains time for a metered backup within the shared deadline',async()=>{
 let clock=0,charges=0;const timeouts=[];
 const p=make(async job=>{timeouts.push(job.timeout_ms);clock+=job.timeout_ms;if(job.provider==='bigmodel')throw err('WEB_TIMEOUT');return{rows:[row]}},{now:()=>clock});
 const out=await p.search(q,{acquireRequest:()=>{charges++;return true}});
 assert.equal(out[0].url,url);assert.deepEqual(timeouts,[6000,6000]);assert.equal(charges,1);assert.equal(clock,12000);
});
test('cancelled half-open recovery releases its reservation',async()=>{
 let clock=0,phase=0;const p=make(async()=>{if(phase===0)throw err('WEB_CONNECT_FAILED');if(phase===1)throw err('CANCELLED');return{rows:[row]}},{now:()=>clock,configImpl:()=>({providers:['bigmodel'],keys:{bigmodel:'test'}})});
 await assert.rejects(p.search(q),{code:'WEB_CONNECT_FAILED'});clock=31000;phase=1;
 await assert.rejects(p.search(q),{code:'CANCELLED'});phase=2;
 assert.equal((await p.search(q))[0].url,url);
});
