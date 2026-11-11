import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createServer} from 'node:http';
import {normalizeSearchConfig,buildSearchJob,parseSearchResponse,SEARCH_PRESETS} from '../src/search-config.mjs';
import {RustWebProvider,runNative} from '../src/rust-web.mjs';
import {RoutedPublicProvider} from '../src/public-provider.mjs';
import {EvidenceSearchService} from '../src/index.mjs';
const row={url:'https://science.example.org/research/nebula-spectrum',title:'Nebula spectrum observations',excerpt:'Measured nebula spectrum evidence'};
const mapping={items:'/results',url:'/url',title:'/title',excerpt:'/excerpt'};
const base=()=>({id:'my-search',endpoint:'https://search.example.org/v2/query',method:'GET',params:{q:'{{query}}'},response:mapping});
const cfg=p=>({version:1,providers:[p]});
const normalized=p=>normalizeSearchConfig(cfg(p),{}).providers[0];
const ready=(raw,options={})=>new RustWebProvider({configImpl:()=>raw,...options});
async function server(t,handler){
 const calls=[];const srv=createServer(async(req,res)=>{const chunks=[];for await(const b of req)chunks.push(b);const r={method:req.method,url:new URL(req.url,'http://127.0.0.1'),headers:req.headers,body:Buffer.concat(chunks).toString()};calls.push(r);await handler(r,res);});
 await new Promise(r=>srv.listen(0,'127.0.0.1',r));t.after(()=>new Promise(r=>srv.close(r)));
 return {calls,endpoint:'http://127.0.0.1:'+srv.address().port+'/search'};
}
const json=(res,obj,status=200)=>{res.writeHead(status,{'content-type':'application/json'});res.end(JSON.stringify(obj));};

test('seven presets are optional conveniences rather than an allowlist of services',()=>{
 assert.deepEqual(SEARCH_PRESETS,['brave','tavily','serper','serpapi','bigmodel','deepseek','searxng']);
 assert.equal(normalized(base()).id,'my-search');assert.equal(normalized(base()).ready,true);
});
test('preset schemas compile to the documented methods, credentials and result shapes',async()=>{
 const fixtures={brave:{web:{results:[{url:row.url,title:row.title,description:row.excerpt}]}},tavily:{results:[{url:row.url,title:row.title,content:row.excerpt}]},serper:{organic:[{link:row.url,title:row.title,snippet:row.excerpt}]},serpapi:{organic_results:[{link:row.url,title:row.title,snippet:row.excerpt}]},bigmodel:{search_result:[{link:row.url,title:row.title,content:row.excerpt}]},deepseek:{results:[{url:row.url,title:row.title,summary:row.excerpt}]},searxng:{results:[{url:row.url,title:row.title,content:row.excerpt}]}};
 for(const preset of SEARCH_PRESETS){const p=normalized({id:preset,preset,...(preset==='searxng'?{endpoint:'https://search.example.org/search'}:{credentials:{api_key:{value:'host-secret'}}})});const job=await buildSearchJob(p,'nebula spectrum',1000);assert.equal(job.op,'search_custom');assert.equal((await parseSearchResponse(p,{payload:fixtures[preset]}))[0].url,row.url);if(preset==='deepseek')assert.equal(job.body.need_content,false);if(preset==='tavily')assert.equal(job.body.include_raw_content,false);}
});
test('a query placeholder cannot recursively expand into host credentials or change the API hostname',async()=>{
 const p=normalized({...base(),params:{q:'{{query}}',api_key:'{{secret:api_key}}'},credentials:{api_key:{value:'protected-key'}}});
 const job=await buildSearchJob(p,'{{secret:api_key}} & q=other',1000);assert.equal(job.params.q,'{{secret:api_key}} & q=other');assert.equal(job.params.api_key,'protected-key');assert.equal(job.url,base().endpoint);
});
test('nested JSON templates preserve JSON scalar types without evaluating expressions',async()=>{
 const p=normalized({...base(),method:'POST',params:{},body:{query:{text:'{{query}}'},options:{count:8,enabled:false},auth:'{{secret:token}}'},credentials:{token:{env:'CUSTOM_TOKEN'}}});
 assert.equal(p.ready,false);
 const n=normalizeSearchConfig(cfg({...base(),method:'POST',params:{},body:{q:'{{query}}',n:8},credentials:{}})).providers[0];const job=await buildSearchJob(n,'"x"\\y',1000);assert.equal(job.body.n,8);assert.equal(job.body.q,'"x"\\y');
});
test('malformed mappings and unsupported fields fail before any request',()=>{
 for(const p of [{...base(),endpoint:'https://{{query}}/'},{...base(),headers:{Host:'evil.example.org'}},{...base(),params:{q:{nested:true}}},{...base(),params:{q:'{{eval:process.env}}'}},{...base(),response:{...mapping,items:'/__proto__/x'}},{...base(),unexpected:true},{...base(),adapter:'../outside.mjs'}])assert.throws(()=>normalized(p),{code:'WEB_CONFIG_INVALID'});
});
test('configured HTTPS API ports do not relax the public page URL rules',async()=>{
 const p=normalized({...base(),endpoint:'https://search.example.org:8443/search'});assert.equal((await buildSearchJob(p,'q',1000)).url,p.endpoint);
 const web=ready(cfg(p));await assert.rejects(web.extract('https://science.example.org:8443/'),{code:'UNSAFE_SOURCE_URL'});
});
test('private and loopback endpoints require the explicit loopback-only option',()=>{
 for(const endpoint of ['http://127.0.0.1:8000/search','http://169.254.169.254/','https://10.0.0.1/'])assert.throws(()=>normalized({...base(),endpoint}),{code:'WEB_CONFIG_INVALID'});
 assert.equal(normalized({...base(),endpoint:'http://127.0.0.1:8000/search',allow_loopback:true}).allow_loopback,true);
 assert.throws(()=>normalized({...base(),endpoint:'http://169.254.169.254/',allow_loopback:true}),{code:'WEB_CONFIG_INVALID'});
});
test('JSON pointers support arrays, escaped keys and fallback snippet fields',async()=>{
 const p=normalized({...base(),response:{items:'/a~1b/0/items',url:'/source/url',title:'/meta/title',excerpt:['/missing','/text']}});
 const rows=await parseSearchResponse(p,{payload:{'a/b':[{items:[{source:{url:row.url},meta:{title:row.title},text:row.excerpt}]}]}});assert.equal(rows[0].excerpt,row.excerpt);
});
test('missing result arrays and bad URL mappings cannot become a false successful empty result',async()=>{
 const p=normalized(base());assert.deepEqual(await parseSearchResponse(p,{payload:{results:[]}}),[]);
 await assert.rejects(parseSearchResponse(p,{payload:{error:'upstream error'}}),{code:'WEB_PROVIDER_INVALID'});
 await assert.rejects(parseSearchResponse(p,{payload:{results:[{unexpected_url:row.url}]}}),{code:'WEB_PROVIDER_INVALID'});
});
test('a real Rust GET reaches the configured port with query auth encoded exactly once',async t=>{
 const {calls,endpoint}=await server(t,(r,res)=>json(res,{results:[row]}));
 const web=ready(cfg({...base(),endpoint,allow_loopback:true,params:{q:'{{query}}',api_key:'{{secret:key}}'},credentials:{key:{value:'host-key&x=1'}}}));
 const out=await web.search('nebula spectrum & q=other');assert.equal(out[0].url,row.url);assert.equal(calls.length,1);assert.equal(calls[0].method,'GET');assert.equal(calls[0].url.searchParams.get('q'),'nebula spectrum & q=other');assert.equal(calls[0].url.searchParams.get('api_key'),'host-key&x=1');assert.equal(calls[0].url.searchParams.getAll('api_key').length,1);
 assert.ok(!JSON.stringify(out.discovery_attempts).includes('host-key'));assert.equal(out.discovery_attempts[0].network_receipt.attempts[0].tls_validated,false);
});
test('a real Rust JSON POST supports arbitrary bearer authentication and nested payload',async t=>{
 const {calls,endpoint}=await server(t,(r,res)=>json(res,{data:{hits:[row]}}));
 const web=ready(cfg({...base(),endpoint,allow_loopback:true,method:'POST',params:{},headers:{Authorization:'Bearer {{secret:token}}'},credentials:{token:{value:'host-token'}},body:{input:{query:'{{query}}'},count:8},response:{...mapping,items:'/data/hits'}}));
 assert.equal((await web.search('nebula spectrum'))[0].url,row.url);assert.equal(calls[0].headers.authorization,'Bearer host-token');assert.equal(JSON.parse(calls[0].body).input.query,'nebula spectrum');
});
test('a real Rust form POST supports body authentication without leaking it to receipts',async t=>{
 const {calls,endpoint}=await server(t,(r,res)=>json(res,{results:[row]}));
 const web=ready(cfg({...base(),endpoint,allow_loopback:true,method:'POST',params:{},body_format:'form',body:{q:'{{query}}',key:'{{secret:key}}',count:8},credentials:{key:{value:'form-secret'}}}));
 const out=await web.search('nebula spectrum');const fields=new URLSearchParams(calls[0].body);assert.equal(fields.get('key'),'form-secret');assert.equal(fields.get('count'),'8');assert.match(calls[0].headers['content-type'],/form-urlencoded/);assert.ok(!JSON.stringify(out.discovery_attempts).includes('form-secret'));
});
test('real API redirects never replay API credentials',async t=>{
 const {calls,endpoint}=await server(t,(r,res)=>{res.writeHead(302,{location:'/other'});res.end()});
 const web=ready(cfg({...base(),endpoint,allow_loopback:true,headers:{'X-API-Key':'test-secret'}}));
 await assert.rejects(web.search('nebula spectrum'),{code:'WEB_REDIRECT_DENIED'});assert.equal(calls.length,1);
});
test('real structured errors are distinct from empty results and provider error bodies stay private',async t=>{
 const {calls,endpoint}=await server(t,(r,res)=>json(res,{error:'sensitive upstream body'},401));
 const web=ready(cfg({...base(),endpoint,allow_loopback:true}));
 await assert.rejects(web.search('nebula spectrum'),e=>e.code==='WEB_AUTH_FAILED'&&!e.message.includes('sensitive'));await assert.rejects(web.search('nebula spectrum'),{code:'WEB_PROVIDER_CIRCUIT_OPEN'});assert.equal(calls.length,1);
});
test('real wrong content types fail clearly instead of being parsed as search results',async t=>{
 const {endpoint}=await server(t,(r,res)=>{res.writeHead(200,{'content-type':'text/html'});res.end('<html>login</html>')});
 await assert.rejects(ready(cfg({...base(),endpoint,allow_loopback:true})).search('nebula spectrum'),{code:'WEB_CONTENT_TYPE_DENIED'});
});
test('a host-installed text adapter handles an alternative response protocol through the same Rust transport',async t=>{
 const {calls,endpoint}=await server(t,(r,res)=>{res.writeHead(200,{'content-type':'text/plain'});res.end([row.url,row.title,row.excerpt].join('\t'))});
 const web=ready(cfg({...base(),endpoint,allow_loopback:true,method:'POST',params:{},body_format:'text',body:'{{query}}',headers:{'Content-Type':'text/plain'},response_format:'text',adapter:'adapters/example-text.mjs'}));
 assert.equal((await web.search('nebula spectrum'))[0].url,row.url);assert.equal(calls[0].body,'nebula spectrum');
});
test('switching configured providers invalidates the discovery cache in the same SDK instance',async t=>{
 const a=await server(t,(r,res)=>json(res,{results:[row]}));const b=await server(t,(r,res)=>json(res,{results:[{...row,url:row.url+'-new'}]}));
 let current=cfg({...base(),endpoint:a.endpoint,allow_loopback:true});const web=new RustWebProvider({configImpl:()=>current});
 const sdk=new EvidenceSearchService({cwd:process.cwd(),allowWeb:true,provider:new RoutedPublicProvider({web})});
 const first=await sdk.query(sdk.newTask('web search'),{mode:'web',query:'nebula spectrum'});assert.equal(first.results[0].url,row.url);
 current=cfg({...base(),endpoint:b.endpoint,allow_loopback:true});const next=await sdk.query(sdk.newTask('web search'),{mode:'web',query:'nebula spectrum'});assert.equal(next.results[0].url,row.url+'-new');assert.equal(b.calls.length,1);
});
test('custom native search permissions cannot turn a public page fetch into loopback access',async()=>{
 await assert.rejects(runNative({op:'fetch',url:'http://127.0.0.1:8000/search',allow_loopback:true,timeout_ms:1000}),{code:'UNSAFE_SOURCE_URL'});
});
test('missing credential references are not sent as empty keys',async()=>{
 let calls=0;const web=ready(cfg({...base(),headers:{Authorization:'Bearer {{secret:key}}'},credentials:{key:{env:'VODER_NONEXISTENT_TEST_SECRET'}}}),{runImpl:async()=>{calls++}});
 await assert.rejects(web.search('nebula spectrum'),{code:'WEB_CREDENTIAL_MISSING'});assert.equal(calls,0);
});

test('there is no mandatory vendor or credential when a backend requires no authentication',async()=>{
 const p=normalizeSearchConfig(cfg({...base(),preset:'searxng'}),{}).providers[0];assert.equal(p.ready,true);assert.deepEqual(p.secrets,{});
 const empty=ready({version:1,providers:[]});await assert.rejects(empty.search('nebula spectrum'),{code:'WEB_SEARCH_NOT_CONFIGURED'});
});
test('multiple custom services share the same reserved fetch budget and fallback path',async()=>{
 const seen=[];const first={...base(),id:'arbitrary-primary'},second={...base(),id:'arbitrary-backup',endpoint:'https://backup.example.org/search'};
 const web=ready({version:1,providers:[first,second]},{runImpl:async job=>{seen.push(job.url);if(job.url===first.endpoint)throw Object.assign(new Error(),{code:'WEB_CONNECT_FAILED'});return{payload:{results:[row]}}}});
 const sdk=new EvidenceSearchService({cwd:process.cwd(),allowWeb:true,provider:new RoutedPublicProvider({web})});const task=sdk.newTask('web search');
 const out=await sdk.query(task,{mode:'web',query:'nebula spectrum'});assert.deepEqual(seen,[first.endpoint,second.endpoint]);assert.equal(out.results[0].url,row.url);assert.equal(task.webUsed,2);assert.equal(out.budget.fetch_reserved,true);
});
test('raw protocol responses require a host adapter instead of silently applying JSON mappings',()=>{
 assert.throws(()=>normalized({...base(),response_format:'text'}),{code:'WEB_CONFIG_INVALID'});
});
