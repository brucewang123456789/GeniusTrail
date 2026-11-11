import {test} from 'node:test';
import assert from 'node:assert/strict';
import {EventEmitter} from 'node:events';
import {Readable} from 'node:stream';
import {EvidenceSearchService} from '../src/index.mjs';
import {KeylessWebProvider,pinnedGet,publicAddress,checkedPublicUrl} from '../src/keyless-web.mjs';
import {RoutedPublicProvider} from '../src/public-provider.mjs';
import {planResearchQueries,rankResearchCandidates} from '../src/research.mjs';
import {ToolEconomy} from '../../voder-tools/src/index.mjs';
const url='https://www.nobelprize.org/prizes/physics/2026/summary/';
const err=code=>Object.assign(new Error('untrusted private exception'),{code});
const hit={url,title:'2026 Nobel Prize Physics',excerpt:'Official announcement'};
const svc=opts=>new EvidenceSearchService({cwd:process.cwd(),allowWeb:true,...opts});
const direct={async extract(u){return {url:u,text:'Observed public source text'};}};
// Simulate actual request/response stream lifecycle, not a replaced pinnedGet.
function transport(routes,calls=[]){return (u,options,callback)=>{
 calls.push({url:u.href,options});const req=new EventEmitter();req.destroy=()=>{req.destroyed=true;};
 req.end=()=>queueMicrotask(()=>{
  const spec=routes[u.href];if(!spec){req.emit('error',err('ECONNRESET'));return;}
  const res=spec.hang?new Readable({read(){}}):Readable.from(spec.chunks??[Buffer.from(spec.body??'body')]);
  res.statusCode=spec.status??200;res.headers={'content-type':'text/html',...(spec.headers??{})};
  req.destroy=()=>{req.destroyed=true;res.destroy();};callback(res);
  if(spec.abort)queueMicrotask(()=>spec.abort());
 });return req;
};}
test('known public URL works with empty discovery and records direct provenance',async()=>{
 let searches=0,fetches=0;
 const s=svc({provider:{async search(){searches++;return [];}},directProvider:{async extract(u,opts){fetches++;assert.equal(opts.urlAllowed(u),true);return direct.extract(u);}}});
 const t=s.newTask('搜索网页，从官网核实');const p=await s.query(t,{mode:'fetch',query:url});
 assert.equal(searches,0);assert.equal(fetches,1);assert.equal(t.webUsed,1);assert.equal(t.webFetchUsed,1);
 assert.equal(t.sourceUrls.has(url),false);assert.equal(p.budget.fetch_reserved,false);assert.equal(p.results[0].source_provenance,'DIRECT_PUBLIC_URL');assert.equal(p.authority,'CONTEXT_ONLY');
});
test('host OFF, user prohibition, invalid focus, cancellation and domain denial do not call direct transport',async()=>{
 let calls=0;const provider={async search(){return []}};const dp={async extract(){calls++;throw Error('must not run')}};
 for(const [s,t,args,signal,code] of [
  [new EvidenceSearchService({cwd:process.cwd(),directProvider:dp}),null,{mode:'fetch',query:url},undefined,'WEB_DISABLED'],
  [svc({provider,directProvider:dp}), '不要联网搜索',{mode:'fetch',query:url},undefined,'WEB_FORBIDDEN_BY_TASK'],
  [svc({provider,directProvider:dp}),null,{mode:'fetch',query:url,focus:'bad\nfocus'},undefined,'INVALID_SEARCH_QUERY'],
  [svc({provider,directProvider:dp}),null,{mode:'fetch',query:url},AbortSignal.abort(),'CANCELLED'],
  [svc({provider,directProvider:dp,allowedDomains:['python.org']}),null,{mode:'fetch',query:url},undefined,'WEB_DOMAIN_DENIED']
 ])await assert.rejects(s.query(s.newTask(t??'web search'),args,signal),{code});
 assert.equal(calls,0);
});
test('default three-request budget always reserves a source request after two weak searches',async()=>{
 const s=svc({provider:{async search(){return []}},directProvider:direct});const t=s.newTask('web search');
 await s.query(t,{mode:'web',query:'first'});await s.query(t,{mode:'web',query:'second'});
 await assert.rejects(s.query(t,{mode:'web',query:'third'}),{code:'WEB_QUERY_BUDGET'});
 const p=await s.query(t,{mode:'fetch',query:url});assert.equal(p.status,'EVIDENCE_FOUND');
 assert.equal(t.webUsed,3);assert.equal(t.webSearchUsed,2);
 await assert.rejects(s.query(t,{mode:'fetch',query:url}),{code:'WEB_QUERY_BUDGET'});
});
test('parallel discovery cannot consume reserved Fetch quota',async()=>{
 let release;const pending=new Promise(r=>release=r);let calls=0;
 const s=svc({provider:{async search(){calls++;await pending;return []}},directProvider:direct});const t=s.newTask('web search');
 const a=s.query(t,{mode:'web',query:'one'}),b=s.query(t,{mode:'web',query:'two'});
 await assert.rejects(s.query(t,{mode:'web',query:'three'}),{code:'WEB_QUERY_BUDGET'});
 release();await Promise.all([a,b]);await s.query(t,{mode:'fetch',query:url});assert.equal(calls,2);assert.equal(t.webUsed,3);
});
test('cache hits do not consume external request quota',async()=>{
 let calls=0;const s=svc({webMaxQueries:2,provider:{async search(){calls++;return [hit]}}});const t=s.newTask('web search');
 await s.query(t,{mode:'web',query:'Nobel'});const b=await s.query(t,{mode:'web',query:'Nobel'});
 assert.equal(b.cache_hit,true);assert.equal(t.webUsed,1);assert.equal(calls,1);
});
test('site restriction rejects spoofed and unrelated domains before publishing candidates',async()=>{
 const s=svc({provider:{async search(){return [
  {url:'https://www.nobelprize.org.evil.com/prizes/physics/',title:'Physics Nobel 2026'},
  {url:'https://mba.example.com/physics/2026/',title:'Physics Nobel 2026'},hit,
  {url:'https://www.nobelprize.org/prizes/chemistry/2026/',title:'Physics Nobel 2026'}];}}});
 const t=s.newTask('web search');const q='site:nobelprize.org/prizes/physics/ Nobel 2026 Physics';
 const out=await s.query(t,{mode:'web',query:q});assert.deepEqual(out.results.map(r=>r.url),[url]);
 assert.deepEqual([...t.sourceUrls],[url]);
 const plan=planResearchQueries(q);assert.ok(plan.queries.every(x=>x.includes('site:nobelprize.org/prizes/physics/')));
 assert.equal(rankResearchCandidates('2026 Nobel Physics',[{url:'https://calendar.example.com/2026/',title:'2026 calendar'}]).length,0);
 assert.deepEqual(rankResearchCandidates(q,[hit,{url:'https://calendar.example.com/2026/',title:'2026'}]).map(r=>r.url),[url]);
});
test('explicit unrelated site routes to general search even with Python or Go keyword',async()=>{
 let official=0,web=0;const p=new RoutedPublicProvider({official:{async search(){official++;return []}},web:{async search(){web++;return []}}});
 await p.search('site:nobelprize.org Go research');await p.search('site:nobelprize.org Python');
 assert.equal(official,0);assert.equal(web,2);
});
test('keyless search filters site before truncating its source list',async()=>{
 let items='';for(let i=0;i<6;i++)items+=`<item><title>2026 calendar</title><link>https://calendar.example.com/${i}</link></item>`;
 items+=`<item><title>2026 physics</title><link>${url}</link></item>`;
 const p=new KeylessWebProvider({getImpl:async()=>`<rss><channel>${items}</channel></rss>`});
 assert.deepEqual((await p.search('site:nobelprize.org physics 2026')).map(x=>x.url),[url]);
});
test('failed discovery circuit cannot block direct source fetch through ToolEconomy',async()=>{
 const s=svc({provider:{async search(){throw err('WEB_PROVIDER_FAILED')}},directProvider:direct});
 const economy=new ToolEconomy();const t=s.newTask('web search');const [tool]=s.makeTools({task:t,toolEconomy:economy});
 for(const q of ['first','second'])await assert.rejects(tool.execute({mode:'web',query:q}),{code:'WEB_PROVIDER_FAILED'});
 // New task avoids the first task's spent discovery quota while reusing the service circuit.
 await assert.rejects(s.query(s.newTask('web search'),{mode:'web',query:'third'}),{code:'WEB_PROVIDER_CIRCUIT_OPEN'});
 const page=await tool.execute({mode:'fetch',query:url});assert.equal(page.data.status,'EVIDENCE_FOUND');
 assert.equal(economy.failures.get(tool.name),0);
});
test('404 is a source outcome; it does not poison another URL or imply the prize is unannounced',async()=>{
 let calls=0;const s=svc({directProvider:{async extract(u){calls++;if(u.endsWith('/missing'))throw err('WEB_SOURCE_NOT_FOUND');return direct.extract(u);}}});
 const t=s.newTask('web search');await assert.rejects(s.query(t,{mode:'fetch',query:url+'missing'}),{code:'WEB_SOURCE_NOT_FOUND'});
 const p=await s.query(t,{mode:'fetch',query:url});assert.equal(p.status,'EVIDENCE_FOUND');assert.equal(calls,2);assert.equal(s.providerCircuits.size,0);
});
test('source circuit is isolated by origin and recovers with one bounded half-open probe',async()=>{
 const s=svc({providerFailureLimit:1,directProvider:{async extract(u){if(u.includes('nobelprize'))throw err('WEB_TIMEOUT');return direct.extract(u)}}});
 await assert.rejects(s.query(s.newTask('web search'),{mode:'fetch',query:url}),{code:'WEB_TIMEOUT'});
 const t=s.newTask('web search');await assert.rejects(s.query(t,{mode:'fetch',query:url}),{code:'WEB_PROVIDER_CIRCUIT_OPEN'});assert.equal(t.webUsed,0);
 assert.equal((await s.query(t,{mode:'fetch',query:'https://example.org/page'})).status,'EVIDENCE_FOUND');
 const state=s.providerCircuits.get('fetch:https://www.nobelprize.org');state.at-=31000;
 let release;const gate=new Promise(r=>release=r);s.directProvider={async extract(u){await gate;return direct.extract(u)}};
 const p=s.query(s.newTask('web search'),{mode:'fetch',query:url});
 await assert.rejects(s.query(s.newTask('web search'),{mode:'fetch',query:url}),{code:'WEB_PROVIDER_CIRCUIT_OPEN'});
 release();await p;assert.equal(s.providerCircuits.size,0);
});
test('source redirects retain original and final URL and hash; sockets always use pinned lookup',async()=>{
 const calls=[];const routes={[url]:{status:301,headers:{location:'/prizes/physics/2026/press-release/'}},
  'https://www.nobelprize.org/prizes/physics/2026/press-release/':{body:'<p>Observed announcement</p>'}};
 const p=new KeylessWebProvider({getImpl:(u,o)=>pinnedGet(u,{...o,requestImpl:transport(routes,calls)})});
 const out=await p.extract(url);assert.equal(out.url,url);assert.match(out.final_url,/press-release/);assert.equal(out.redirects.length,1);
 assert.match(out.raw_sha256,/^[a-f0-9]{64}$/);assert.equal(calls.length,2);
 assert.ok(calls.every(c=>typeof c.options.lookup==='function'&&c.options.rejectUnauthorized===true&&c.options.agent===false));
});
test('redirect policy rejects unsafe destination, downgrade, domain escape, loops and excessive chain',async()=>{
 for(const dest of ['http://www.nobelprize.org/x','https://127.0.0.1/x','https://user:pass@example.org/','https://internal.local/']){
  const calls=[];await assert.rejects(pinnedGet(url,{sourceRedirects:true,requestImpl:transport({[url]:{status:302,headers:{location:dest}}},calls)}),{code:'UNSAFE_SOURCE_URL'});
  assert.equal(calls.length,1);
 }
 const calls=[];await assert.rejects(pinnedGet(url,{sourceRedirects:true,urlAllowed:u=>new URL(u).hostname==='www.nobelprize.org',requestImpl:transport({[url]:{status:302,headers:{location:'https://evil.example.com/'}}},calls)}),{code:'WEB_DOMAIN_DENIED'});assert.equal(calls.length,1);
 await assert.rejects(pinnedGet(url,{sourceRedirects:true,requestImpl:transport({[url]:{status:302,headers:{location:url}}})}),{code:'WEB_REDIRECT_LOOP'});
 await assert.rejects(pinnedGet(url,{sourceRedirects:true,maxRedirects:0,requestImpl:transport({[url]:{status:302,headers:{location:'/next'}}})}),{code:'WEB_REDIRECT_LIMIT'});
});
test('cross-domain redirect rechecks public transport and preserves allowed final provenance',async()=>{
 const final='https://news.example.org/announcement';const calls=[];
 const result=await pinnedGet(url,{sourceRedirects:true,returnMetadata:true,requestImpl:transport({[url]:{status:307,headers:{location:final}},[final]:{body:'source'}},calls)});
 assert.equal(result.url,final);assert.equal(calls[1].options.servername,'news.example.org');
});
test('HTTP statuses are explicit sanitized outcomes, not network diagnoses',async()=>{
 for(const [status,code] of [[404,'WEB_SOURCE_NOT_FOUND'],[410,'WEB_SOURCE_NOT_FOUND'],[403,'WEB_ACCESS_DENIED'],[429,'WEB_RATE_LIMITED'],[503,'WEB_HTTP_ERROR']])
 await assert.rejects(pinnedGet(url,{requestImpl:transport({[url]:{status}})}),e=>e.code===code&&e.httpStatus===status&&!String(e).includes('private'));
});
test('abort tears down a stalled body immediately and leaves no active source request',async()=>{
 const c=new AbortController(),calls=[];await assert.rejects(pinnedGet(url,{signal:c.signal,requestImpl:transport({[url]:{hang:true,abort:()=>c.abort()}},calls)}),{code:'CANCELLED'});
 assert.equal(c.signal.aborted,true);
});
test('one deadline bounds redirected stalled body',async()=>{
 const started=Date.now();await assert.rejects(pinnedGet(url,{timeoutMs:1000,sourceRedirects:true,requestImpl:transport({[url]:{status:302,headers:{location:'/slow'}},'https://www.nobelprize.org/slow':{hang:true}})}),{code:'WEB_TIMEOUT'});
 assert.ok(Date.now()-started<2000);
});
test('response size, encoding and MIME refusal release transport',async()=>{
 for(const [opts,spec,code] of [[{maxBytes:2},{body:'too large'},'SEARCH_RESPONSE_TOO_LARGE'],[{}, {headers:{'content-type':'image/png'}},'WEB_CONTENT_TYPE_DENIED'],[{}, {headers:{'content-encoding':'gzip'}},'WEB_CONTENT_ENCODING_DENIED']])
 await assert.rejects(pinnedGet(url,{...opts,requestImpl:transport({[url]:spec})}),{code});
});
test('IPv6 mapped and translation addresses and malformed URL labels are rejected',()=>{
 for(const ip of ['::ffff:0808:0808','64:ff9b::808:808','2002:0a00:0001::1','2001::1','::2'])assert.equal(publicAddress(ip),false,ip);
 for(const u of ['https://-bad.example.org/','https://good.example.org/\nadmin','https://good.example.org/?token=abc','https://good.example.org/?auth_token=abc','https://good.example.org/?x=ghp_AAAAAAAAAAAAAAAAAAAA'])assert.throws(()=>checkedPublicUrl(u),{code:'UNSAFE_SOURCE_URL'});
});
