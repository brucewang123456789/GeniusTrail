import {test} from 'node:test';
import assert from 'node:assert/strict';
import {KeylessWebProvider, parseBingRss, checkedPublicUrl, publicAddress, pinnedGet} from '../src/keyless-web.mjs';
import {EvidenceSearchService,OfficialMultisourceProvider} from '../src/index.mjs';

const RSS=`<?xml version="1.0"?><rss version="2.0"><channel><title>Search</title>
 <item><title>Python &amp; docs</title><link>https://docs.python.org/3/library/pathlib.html</link><description><![CDATA[<b>pathlib</b> Path reference]]></description></item>
 <item><title>intranet</title><link>http://127.0.0.1/admin</link><description>drop</description></item>
</channel></rss>`;
test('Bing RSS parser decodes source details and refuses untrusted/private links',()=>{
 assert.deepEqual(parseBingRss(RSS),[{url:'https://docs.python.org/3/library/pathlib.html',title:'Python & docs',excerpt:'pathlib Path reference'}]);
 for(const raw of ['not rss','<html>login</html>'])assert.throws(()=>parseBingRss(raw),{code:'WEB_PROVIDER_INVALID'});
});
test('keyless Bing provider performs only pinned web search endpoint and never asks for key',async()=>{
 const calls=[];const provider=new KeylessWebProvider({getImpl:async(url,opts)=>{calls.push([url,opts]);return RSS;}});
 const results=await provider.search('python pathlib');
 const got=new URL(calls[0][0]);assert.equal(got.host,'www.bing.com');assert.equal(got.pathname,'/search');
 assert.equal(got.searchParams.get('q'),'python pathlib');assert.equal(got.searchParams.get('format'),'rss');
 assert.equal(results.length,1);assert.equal(results[0].url,'https://docs.python.org/3/library/pathlib.html');
 assert.ok(!JSON.stringify(calls).includes('Authorization'));
});
test('keyless provider extracts only verified public HTTPS source, strips active and injected HTML scripts',async()=>{
 let n=0;const provider=new KeylessWebProvider({getImpl:async (url)=>{n++;assert.equal(url,'https://docs.python.org/3/library/pathlib.html');
 return '<html><script>HIDDEN_MALICIOUS</script><body><h1>Pathlib</h1><p>Official reference</p></body></html>';}});
 const out=await provider.extract('https://docs.python.org/3/library/pathlib.html');
 assert.equal(n,1);assert.match(out.text,/Pathlib/);assert.match(out.text,/Official reference/);assert.doesNotMatch(out.text,/HIDDEN_MALICIOUS/);
 for(const u of ['https://localhost/x','https://127.0.0.1/x','https://10.1.2.3/x','http://docs.python.org/','https://foo.internal/x','https://user:pass@docs.python.org/x'])
  await assert.rejects(provider.extract(u),{code:'UNSAFE_SOURCE_URL'});
 assert.equal(n,1);
});
test('optional SearXNG has explicit host config and emits bounded JSON search',async()=>{
 assert.throws(()=>new KeylessWebProvider({engine:'searxng'}),{code:'INVALID_SEARCH_PROVIDER'});
 assert.throws(()=>new KeylessWebProvider({engine:'searxng',searxngUrl:'http://10.0.0.7:8888'}),{code:'INVALID_SEARCH_PROVIDER'});
 assert.throws(()=>new KeylessWebProvider({engine:'searxng',searxngUrl:'https://u:p@searx.example.com'}),{code:'INVALID_SEARCH_PROVIDER'});
 const calls=[];const s=new KeylessWebProvider({engine:'searxng',searxngUrl:'http://127.0.0.1:8888',
  getImpl:async(url,opts)=>{calls.push([url,opts]);return JSON.stringify({results:[{url:'https://docs.python.org/docs',title:'Docs',content:'reference'}]});}});
 assert.equal((await s.search('python docs'))[0].excerpt,'reference');
 assert.equal(new URL(calls[0][0]).hostname,'127.0.0.1');assert.equal(calls[0][1].allowLoopback,true);
 assert.equal(new URL(calls[0][0]).searchParams.get('format'),'json');
});
test('public address pinning rejects private, reserved and cloud metadata ranges',()=>{
 for(const a of ['127.0.0.1','169.254.169.254','10.1.3.1','172.23.0.1','192.168.1.1','100.64.0.1','198.19.0.2','203.0.113.2','::1','fe80::1','fc00::1','::ffff:127.0.0.1'])
  assert.equal(publicAddress(a),false,a);
 for(const a of ['8.8.8.8','1.1.1.1','2606:4700:4700::1111'])assert.equal(publicAddress(a),true,a);
});
test('https transport refuses IP literals, insecure scheme and cancellation before DNS/network',async()=>{
 for(const u of ['https://127.0.0.1/x','http://docs.python.org/x','https://internal.local/x'])
   await assert.rejects(pinnedGet(u),{code:'UNSAFE_SOURCE_URL'});
 const c=new AbortController();c.abort();
 await assert.rejects(pinnedGet('https://docs.python.org/',{signal:c.signal}),{code:'CANCELLED'});
});
test('default routes public queries while host network permission remains mandatory',async()=>{
 const service=new EvidenceSearchService({cwd:process.cwd(),allowWeb:false});
 assert.equal(service.provider.constructor.name,'RoutedPublicProvider');
 await assert.rejects(service.query(service.newTask('search the web'),{mode:'web',query:'python pathlib'}),{code:'WEB_DISABLED'});
 const allowed=new EvidenceSearchService({cwd:process.cwd(),allowWeb:true});
 assert.equal(allowed.provider.constructor.name,'RoutedPublicProvider');
});
test('real service preserves task discovery -> fetch -> untrusted evidence -> no VGR authority',async()=>{
 const provider=new KeylessWebProvider({getImpl:async (url)=>new URL(url).hostname==='www.bing.com'?RSS:'<html><body><h1>Official Pathlib</h1><p>Path objects</p></body></html>'});
 const service=new EvidenceSearchService({cwd:process.cwd(),allowWeb:true,provider,webMaxQueries:2,allowedDomains:['docs.python.org']});
 const t=service.newTask('web search Python docs');
 await assert.rejects(service.query(t,{mode:'fetch',query:'https://127.0.0.1/'}),{code:'UNSAFE_SOURCE_URL'});
 const result=await service.query(t,{mode:'web',query:'pathlib'});
 assert.equal(result.results.length,1);
 const page=await service.query(t,{mode:'fetch',query:result.results[0].url,focus:'Path objects'});
 assert.equal(page.authority,'CONTEXT_ONLY');assert.equal(page.results[0].trust_level,'UNTRUSTED_WEB_DATA');
 assert.match(page.results[0].content_hash,/^[0-9a-f]{64}$/);
 await assert.rejects(service.query(t,{mode:'web',query:'third query'}),{code:'WEB_QUERY_BUDGET'});
});
import {createServer} from 'node:http';
test('self-hosted SearXNG HTTP loopback works without secrets and never permits redirects',async()=>{
 const server=createServer((req,res)=>{
  if(req.url?.startsWith('/search?')){res.setHeader('Content-Type','application/json');res.end(JSON.stringify({results:[{url:'https://docs.python.org/3/',title:'Python',content:'documentation'}]}));return;}
  res.writeHead(302,{Location:'http://127.0.0.1:8080/private'}).end('redirect');
 });
 await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
 const addr=server.address();
 try{
  const service=new KeylessWebProvider({engine:'searxng',searxngUrl:`http://127.0.0.1:${addr.port}`});
  const rows=await service.search('official python');
  assert.equal(rows.length,1);assert.equal(rows[0].url,'https://docs.python.org/3/');
  await assert.rejects(pinnedGet(`http://127.0.0.1:${addr.port}/redirect`,{
   allowLoopback:true,contentTypes:['application/json']}),{code:'WEB_REDIRECT_DENIED'});
 }finally{await new Promise((resolve,reject)=>server.close(err=>err?reject(err):resolve()));}
});
