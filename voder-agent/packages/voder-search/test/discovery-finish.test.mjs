import {test} from 'node:test';
import assert from 'node:assert/strict';
import {EvidenceSearchService} from '../src/index.mjs';
import {KeylessWebProvider,parseDuckDuckGoHtml} from '../src/keyless-web.mjs';
const target='https://science.example.org/research/nebula-spectrum';
const q='site:science.example.org/research nebula spectrum';
const rss=rows=>'<rss><channel>'+rows.map(r=>`<item><title>${r.title}</title><link>${r.url}</link><description>${r.excerpt??''}</description></item>`).join('')+'</channel></rss>';
const good={url:target,title:'Nebula spectrum observations',excerpt:'Spectrum measurements'};
const bad={url:'https://calendar.example.net/2026/',title:'2026 calendar'};
const html=(href=target)=>`<html><a href="${href}" class="result__a">Nebula &amp; spectrum</a><a class="result__snippet">Measurements of a distant nebula</a></html>`;
const make=opts=>new EvidenceSearchService({cwd:process.cwd(),allowWeb:true,...opts});
const deferred=()=>{let resolve;return {promise:new Promise(r=>resolve=r),resolve:(v)=>resolve(v)}};

test('secondary parser handles source wrappers and attribute order without executing markup',()=>{
 const wrapper='//duckduckgo.com/l/?uddg='+encodeURIComponent(target)+'&amp;rut=abcd';
 const rows=parseDuckDuckGoHtml(html(wrapper));assert.equal(rows[0].url,target);assert.equal(rows[0].title,'Nebula & spectrum');assert.match(rows[0].excerpt,/Measurements/);
 assert.equal(parseDuckDuckGoHtml('<div class="no-results">No results found</div>').length,0);
 assert.throws(()=>parseDuckDuckGoHtml('<form id="challenge-form">captcha</form>'),{code:'WEB_SEARCH_CHALLENGE'});
 assert.throws(()=>parseDuckDuckGoHtml('<html>sign in</html>'),{code:'WEB_PROVIDER_INVALID'});
 for(const u of ['javascript:alert(1)','http://science.example.org/','https://127.0.0.1/','https://internal.local/'])assert.equal(parseDuckDuckGoHtml(html(u)).length,0);
});
test('weak scoped primary search uses one metered secondary request and retains domain/path checks',async()=>{
 const calls=[];const p=new KeylessWebProvider({getImpl:async(u,o)=>{calls.push([u,o]);return calls.length===1?rss([bad]):html();}});
 const s=make({provider:p});const t=s.newTask('web search');const out=await s.query(t,{mode:'web',query:q});
 assert.deepEqual(out.results.map(r=>r.url),[target]);assert.equal(t.webUsed,2);assert.equal(t.webSearchUsed,2);assert.equal(out.budget.remaining,1);assert.equal(out.budget.fetch_reserved,true);
 assert.equal(out.discovery_attempts[0].eligible,0);assert.equal(out.discovery_attempts[1].status,'CANDIDATES_FOUND');
 const endpoint=new URL(calls[1][0]);assert.equal(endpoint.hostname,'html.duckduckgo.com');assert.equal(endpoint.pathname,'/html/');assert.equal(endpoint.searchParams.get('q'),q);
 assert.equal(calls[1][1].urlAllowed('https://evil.example.com/html/?q='+encodeURIComponent(q)),false);
 assert.equal(calls[1][1].urlAllowed('https://html.duckduckgo.com/html/?q=changed'),false);
 assert.ok(calls[1][1].timeoutMs<=12000);
});
test('unrelated year-only hits trigger metered fallback for ordinary fact queries',async()=>{
 let calls=0;const p=new KeylessWebProvider({getImpl:async()=>++calls===1?rss([bad]):html()});
 const s=make({provider:p});const out=await s.query(s.newTask('web search'),{mode:'web',query:'2026 nebula spectrum'});
 assert.equal(calls,2);assert.equal(out.results[0].url,target);
});
test('strong existing primary result never invokes a second engine',async()=>{
 let calls=0;const p=new KeylessWebProvider({getImpl:async()=>{calls++;return rss([good]);}});
 const s=make({provider:p});const t=s.newTask('web search');const out=await s.query(t,{mode:'web',query:q});
 assert.equal(calls,1);assert.equal(t.webUsed,1);assert.equal(out.results[0].url,target);
});
test('budget refusal keeps primary outcome and reserved fetch; it is not an adapter failure',async()=>{
 let calls=0;const p=new KeylessWebProvider({getImpl:async()=>{calls++;return rss([bad]);}});
 const s=make({provider:p,webMaxQueries:2});const t=s.newTask('web search');const out=await s.query(t,{mode:'web',query:q});
 assert.equal(out.status,'NO_EVIDENCE');assert.equal(calls,1);assert.equal(t.webUsed,1);assert.equal(out.discovery_attempts[1].status,'SKIPPED_BUDGET');assert.equal(s.providerFailureCount,0);
});
test('secondary failure is explicit and does not erase a usable primary response',async()=>{
 let calls=0;const p=new KeylessWebProvider({getImpl:async()=>++calls===1?rss([bad]):'<form id="challenge-form">challenge</form>'});
 const s=make({provider:p});const out=await s.query(s.newTask('web search'),{mode:'web',query:'2026 nebula spectrum'});
 assert.equal(out.results[0].url,bad.url);assert.deepEqual(out.discovery_attempts[1],{provider:'duckduckgo-html',status:'FAILED',code:'WEB_SEARCH_CHALLENGE'});assert.equal(s.providerFailureCount,0);
});
test('host domain policy cannot be bypassed by a secondary result or a spoofed host',async()=>{
 let calls=0;const p=new KeylessWebProvider({getImpl:async()=>++calls===1?rss([bad]):html('https://science.example.org.evil.com/research/nebula-spectrum')+html(target)});
 const s=make({provider:p,allowedDomains:['science.example.org']});const out=await s.query(s.newTask('web search'),{mode:'web',query:q});
 assert.deepEqual(out.results.map(x=>x.url),[target]);
});
test('standalone provider calls without host metering cannot silently issue a second request',async()=>{
 let calls=0;const p=new KeylessWebProvider({getImpl:async()=>{calls++;return rss([bad])}});
 assert.deepEqual(await p.search(q),[]);assert.equal(calls,1);
});
test('configured SearXNG never silently switches to a public secondary engine',async()=>{
 let calls=0;const p=new KeylessWebProvider({engine:'searxng',searxngUrl:'http://127.0.0.1:8888',getImpl:async()=>{calls++;return '{"results":[]}'}});
 const s=make({provider:p});await s.query(s.newTask('web search'),{mode:'web',query:q});assert.equal(calls,1);
});
test('in-flight fetch satisfies reservation while research and fetch share the exact total budget',async()=>{
 const first=deferred(),page=deferred();let searches=0;
 const s=make({provider:{async search(){searches++;if(searches===1)await first.promise;return [];}},directProvider:{async extract(url){await page.promise;return {url,text:'Public source body'};}}});
 const t=s.newTask('web search');const research=s.query(t,{mode:'research',query:'nebula spectrum'});
 const fetch=s.query(t,{mode:'fetch',query:target});assert.equal(t.webFetchPending,1);assert.equal(s._budgetSnapshot(t).fetch_reserved,false);
 first.resolve();const result=await research;assert.equal(result.attempts,2);assert.equal(searches,2);assert.equal(t.webUsed,3);
 page.resolve();assert.equal((await fetch).status,'EVIDENCE_FOUND');assert.equal(t.webFetchPending,0);
 await assert.rejects(s.query(t,{mode:'fetch',query:target}),{code:'WEB_QUERY_BUDGET'});
});
test('research returns prior candidates when concurrent work exhausts discovery capacity',async()=>{
 const gate=deferred();let calls=0;
 const s=make({provider:{async search(){calls++;if(calls===1){await gate.promise;return [good];}return [];}}});
 const t=s.newTask('web search');const first=s.query(t,{mode:'research',query:'nebula spectrum'});
 await s.query(t,{mode:'web',query:'another discovery'});gate.resolve();const out=await first;
 assert.equal(out.status,'CANDIDATES_FOUND');assert.equal(out.results[0].url,target);assert.equal(out.partial,true);assert.equal(out.stop_reason,'WEB_QUERY_BUDGET');assert.equal(out.attempts,1);assert.equal(t.webUsed,2);
});
test('failed fetch releases pending reservation state; cancellation still propagates',async()=>{
 const s=make({directProvider:{async extract(){throw Object.assign(new Error('timeout'),{code:'WEB_TIMEOUT'})}}});const t=s.newTask('web search');
 await assert.rejects(s.query(t,{mode:'fetch',query:target}),{code:'WEB_TIMEOUT'});assert.equal(t.webFetchPending,0);assert.equal(t.fetched,false);assert.equal(s._budgetSnapshot(t).fetch_reserved,true);
 const c=new AbortController();let n=0;const p=new KeylessWebProvider({getImpl:async()=>{n++;c.abort();return rss([bad]);}});
 await assert.rejects(p.search(q,{signal:c.signal,acquireRequest:()=>{throw Object.assign(new Error('cancel'),{code:'CANCELLED'})}}),{code:'CANCELLED'});assert.equal(n,1);
});
