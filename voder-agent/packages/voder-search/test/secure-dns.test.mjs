import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createServer} from 'node:http';
import {execFile} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {normalizeNetwork} from '../src/network-config.mjs';
import {normalizeSearchConfig} from '../src/search-config.mjs';
import {createNativeRunner,runNative,RustWebProvider,networkSummary} from '../src/rust-web.mjs';
const binary=fileURLToPath(new URL('../native/bin/voder-retrieval',import.meta.url));
function question(wire){let end=12;const labels=[];while(wire[end]){const n=wire[end++];labels.push(wire.subarray(end,end+n).toString());end+=n;}end++;const kind=wire.readUInt16BE(end);return{host:labels.join('.'),kind,bytes:wire.subarray(12,end+4)};}
function reply(q,{ips=['3.33.153.106','15.197.138.111'],ad=true,privateAnswer=false,wrong=false,rcode=0,ttl=60}={}){
 const records=q.kind===1?ips:[];const a=[];
 for(const ip of records){const header=Buffer.alloc(12);header.writeUInt16BE(0xc00c,0);header.writeUInt16BE(1,2);header.writeUInt16BE(1,4);header.writeUInt32BE(ttl,6);header.writeUInt16BE(4,10);a.push(header,Buffer.from((privateAnswer?'127.0.0.1':ip).split('.').map(Number)));}
 const header=Buffer.alloc(12);header.writeUInt16BE(0,0);header.writeUInt16BE(0x8180|(ad?0x20:0)|rcode,2);header.writeUInt16BE(1,4);header.writeUInt16BE(records.length,6);
 const bytes=Buffer.from(q.bytes);if(wrong)bytes[1]^=1;
 return Buffer.concat([header,bytes,...a]);
}
async function doh(t,options={}){
 const calls=[];const srv=createServer(async(req,res)=>{const chunks=[];for await(const b of req)chunks.push(b);const url=new URL(req.url,'http://127.0.0.1');const wire=req.method==='GET'?Buffer.from(url.searchParams.get('dns'),'base64url'):Buffer.concat(chunks);const q=question(wire);calls.push({method:req.method,headers:req.headers,host:q.host,type:q.kind});
  if(options.delay&&(options.delayType===undefined||options.delayType===q.kind))await new Promise(r=>setTimeout(r,options.delay));
  res.writeHead(options.status??200,{'content-type':options.contentType??'application/dns-message',...(options.age!==undefined?{age:String(options.age)}:{})});res.end(options.body??reply(q,options));
 });await new Promise(r=>srv.listen(0,'127.0.0.1',r));t.after(()=>new Promise(r=>{srv.closeAllConnections();srv.close(r)}));
 return {calls,resolver:{id:options.id??'local-doh',url:'http://127.0.0.1:'+srv.address().port+'/dns-query',allow_loopback:true,bootstrap_ips:[],method:options.method??'POST'}};
}
const policy=resolvers=>normalizeNetwork({dns:{resolvers,timeout_ms:600}});
const resolve=(network,extra={})=>runNative({op:'resolve',url:'https://api.search.brave.com/',network,timeout_ms:1800,...extra});
function raw(job){return new Promise((resolve,reject)=>{const child=execFile(binary,[],{maxBuffer:2*1024*1024},(e,stdout)=>e?reject(e):resolve(JSON.parse(stdout)));child.stdin.end(JSON.stringify(job));});}

test('secure DNS is the default and contains no Brave or historical target IP pin',()=>{
 const n=normalizeNetwork();assert.equal(n.dns.mode,'secure');assert.equal(n.dns.resolvers.length,4);assert.ok(!JSON.stringify(n).includes('3.33.153.106'));assert.ok(!JSON.stringify(n).includes('128.242.240.253'));
});
test('global and per-provider DNS policies are configurable without changing provider protocol',()=>{
 const cfg=normalizeSearchConfig({version:1,network:{dns:{mode:'system',address_family:'ipv4'}},providers:[{id:'search',preset:'brave',network:{dns:{mode:'secure'}}}]});assert.equal(cfg.network.dns.mode,'system');assert.equal(cfg.providers[0].network.dns.mode,'secure');assert.equal(cfg.providers[0].network.dns.address_family,'ipv4');
});
test('resolver endpoints, bootstrap addresses, bounds and unknown fields are validated',()=>{
 for(const bad of [{dns:{mode:'auto-unsafe'}},{dns:{address_family:'magic'}},{dns:{resolvers:[]}},{connect_timeout_ms:999999},{dns:{resolvers:[{id:'bad',url:'https://dns.example.org/query',bootstrap_ips:['127.0.0.1']}]}},{dns:{resolvers:[{id:'bad',url:'http://169.254.169.254/',allow_loopback:true}]}},{dns:{password:'secret'}}])assert.throws(()=>normalizeNetwork(bad),{code:'WEB_CONFIG_INVALID'});
});
test('real native RFC8484 resolves the reported Brave case through the configured resolver',async t=>{
 const d=await doh(t);const out=await resolve(policy([d.resolver]));assert.deepEqual(out.network_receipt.dns_addresses,['3.33.153.106','15.197.138.111']);assert.equal(out.network_receipt.dns.mode,'SECURE_DOH');assert.equal(d.calls.length,2);assert.ok(d.calls.every(c=>c.host==='api.search.brave.com'&&c.method==='POST'));assert.ok(d.calls.every(c=>c.headers['content-type']==='application/dns-message'));assert.ok(!d.calls.some(c=>c.headers.authorization));
});
test('a custom RFC8484 GET resolver is supported with correctly encoded wire queries',async t=>{
 const d=await doh(t,{method:'GET'});await resolve(policy([d.resolver]));assert.ok(d.calls.every(c=>c.method==='GET'));
});
test('native fallback uses a healthy resolver when another endpoint fails',async t=>{
 const bad=await doh(t,{id:'bad',status:503}),good=await doh(t,{id:'good'});const out=await resolve(policy([bad.resolver,good.resolver]));assert.equal(out.network_receipt.dns.resolvers.find(r=>r.resolver==='bad').status,'FAILED');assert.equal(out.network_receipt.dns.resolvers.find(r=>r.resolver==='good').status,'RESOLVED');assert.deepEqual(out.network_receipt.dns_addresses,['3.33.153.106','15.197.138.111']);
});
test('resolver authenticated-data answers take precedence over unsigned inconsistent addresses',async t=>{
 const unsigned=await doh(t,{id:'unsigned',ad:false,ips:['128.242.240.253']}),signed=await doh(t,{id:'signed'});const out=await resolve(policy([unsigned.resolver,signed.resolver]));assert.ok(!out.network_receipt.dns_addresses.includes('128.242.240.253'),JSON.stringify(out.network_receipt));assert.equal(out.network_receipt.dns.resolvers.length,2);
});
test('a private record invalidates the entire resolver answer while safe alternatives remain usable',async t=>{
 const poisoned=await doh(t,{id:'poisoned',privateAnswer:true}),good=await doh(t,{id:'safe'});const out=await resolve(policy([poisoned.resolver,good.resolver]));assert.ok(!out.network_receipt.dns_addresses.includes('127.0.0.1'));assert.equal(out.network_receipt.dns.resolvers.find(r=>r.resolver==='poisoned').code,'DNS_PUBLIC_ADDRESS_REQUIRED');
});
test('all failed secure resolvers produce a typed failure without system fallback',async t=>{
 const bad=await doh(t,{status:503});await assert.rejects(resolve(policy([bad.resolver])),e=>e.code==='WEB_SECURE_DNS_UNAVAILABLE'&&e.network_receipt.mode==='secure');
});
test('strict authenticated-data policy rejects unsigned responses',async t=>{
 const d=await doh(t,{ad:false});const n=policy([d.resolver]);n.dns.require_ad=true;await assert.rejects(resolve(n),{code:'WEB_SECURE_DNS_UNAVAILABLE'});
});
test('a mismatched DNS question or unexpected content type is not accepted',async t=>{
 const d=await doh(t,{wrong:true});await assert.rejects(resolve(policy([d.resolver])),{code:'WEB_SECURE_DNS_UNAVAILABLE'});
 const html=await doh(t,{contentType:'text/html'});await assert.rejects(resolve(policy([html.resolver])),{code:'WEB_SECURE_DNS_UNAVAILABLE'});
});
test('TTL expiry, hostname and policy binding govern native DNS cache reuse',async t=>{
 const d=await doh(t);const n=policy([d.resolver]);
 const job={op:'resolve',url:'https://api.search.brave.com/',network:n,timeout_ms:1800};const first=await raw(job);assert.equal(first.ok,true);const cache=first.data.network_receipt.dns_cache;assert.ok(cache.expires_ms>Date.now());
 const count=d.calls.length;const hit=await raw({...job,cached_dns:cache});assert.equal(hit.data.network_receipt.dns.cache_hit,true);assert.equal(d.calls.length,count);
 const miss=await raw({...job,cached_dns:{...cache,host:'wrong.example.org'}});assert.equal(miss.data.network_receipt.dns.cache_hit,false);assert.equal(d.calls.length,count+2);
 const expired=await raw({...job,cached_dns:{...cache,expires_ms:Date.now()-1}});assert.equal(expired.data.network_receipt.dns.cache_hit,false);
 const privateCache=await raw({...job,cached_dns:{...cache,addresses:['127.0.0.1']}});assert.ok(!privateCache.data.network_receipt.dns_addresses.includes('127.0.0.1'));
});
test('cached DNS never grants private source access or disables URL safety',async()=>{
 await assert.rejects(runNative({op:'fetch',url:'https://127.0.0.1/',timeout_ms:1000,cached_dns:{host:'127.0.0.1',policy_hash:'forged',expires_ms:Date.now()+60000,addresses:['127.0.0.1']}}),{code:'UNSAFE_SOURCE_URL'});
});
test('cache refresh reconnects only before an HTTP request could have been sent',async()=>{
 let clock=1000,calls=[];
 const cached={host:'example.org',policy_hash:'test',expires_ms:50000,addresses:['8.8.8.8']};
 const runner=createNativeRunner(async job=>{calls.push(job);if(calls.length===2){clock+=1000;throw Object.assign(new Error('connect'),{code:'WEB_CONNECT_FAILED',network_receipt:{attempts:[{result:'WEB_CONNECT_FAILED',retry_before_request:true}],dns:{cache_hit:true}}});}return{network_receipt:{dns_cache:{...cached},attempts:[{result:'HTTP_RESPONSE',tls_validated:true}]}};},{now:()=>clock});
 const job={op:'fetch',url:'https://example.org/',timeout_ms:6000};await runner(job);const out=await runner(job);assert.equal(calls.length,3);assert.ok(calls[1].cached_dns);assert.equal(calls[2].cached_dns,undefined);assert.equal(calls[2].timeout_ms,5000);assert.equal(out.network_receipt.cache_refresh,true);
});
test('an ambiguous timeout after sending a request is never automatically replayed',async()=>{
 const cached={host:'example.org',policy_hash:'test',expires_ms:50000,addresses:['8.8.8.8']};let count=0;
 const runner=createNativeRunner(async()=>{count++;if(count===1)return{network_receipt:{dns_cache:cached,attempts:[{result:'HTTP_RESPONSE',tls_validated:true}]}};throw Object.assign(new Error(),{code:'WEB_TIMEOUT',network_receipt:{attempts:[{result:'WEB_TIMEOUT',retry_before_request:false}]}});},{now:()=>1000});
 const job={op:'fetch',url:'https://example.org/',timeout_ms:6000};await runner(job);await assert.rejects(runner(job),{code:'WEB_TIMEOUT'});assert.equal(count,2);
});
test('cache entries stay internal to the native runner and do not include search credentials',async()=>{
 const runner=createNativeRunner(async()=>({network_receipt:{dns_cache:{host:'example.org',policy_hash:'test',expires_ms:50000,addresses:['8.8.8.8']},attempts:[{result:'HTTP_RESPONSE',tls_validated:true}]}}),{now:()=>1000});
 const out=await runner({op:'fetch',url:'https://example.org/',timeout_ms:1000});assert.equal(out.network_receipt.dns_cache,undefined);
});
test('changing only the network policy changes a provider cooldown identity',async()=>{
 let current={version:1,providers:[{id:'search',preset:'brave',credentials:{api_key:{value:'test-key'}}}]};let calls=0;
 const p=new RustWebProvider({configImpl:()=>current,runImpl:async()=>{calls++;throw Object.assign(new Error(),{code:'WEB_CONNECT_FAILED'})}});
 await assert.rejects(p.search('nebula spectrum'),{code:'WEB_CONNECT_FAILED'});current={...current,network:{dns:{mode:'system'}}};await assert.rejects(p.search('nebula spectrum'),{code:'WEB_CONNECT_FAILED'});assert.equal(calls,2);
});


test('a slow AAAA response cannot discard a valid IPv4 answer in auto mode',async t=>{
 const d=await doh(t,{delay:900,delayType:28});const started=Date.now();const out=await resolve(policy([d.resolver]));assert.ok(Date.now()-started<1200);assert.ok(out.network_receipt.dns_addresses.includes('3.33.153.106'));assert.equal(out.network_receipt.dns.queries_scheduled,2);
});
test('cancellation terminates the real native DNS operation promptly',async t=>{
 const d=await doh(t,{delay:900});const controller=new AbortController();const pending=runNative({op:'resolve',url:'https://api.search.brave.com/',network:policy([d.resolver]),timeout_ms:1800},{signal:controller.signal});setTimeout(()=>controller.abort(),30);await assert.rejects(pending,{code:'CANCELLED'});
});
test('normalization returns owned bootstrap arrays rather than mutating module defaults',()=>{
 const a=normalizeNetwork();a.dns.resolvers[0].bootstrap_ips[0]='127.0.0.1';assert.equal(normalizeNetwork().dns.resolvers[0].bootstrap_ips[0],'8.8.8.8');
});

test('HTTP cache age is subtracted from DNS TTL before creating a cache lease',async t=>{
 const d=await doh(t,{age:70,ttl:60});const out=await resolve(policy([d.resolver]));assert.equal(out.network_receipt.dns.ttl,0);
});

test('detailed DNS diagnostics cannot overflow the existing model tool return budget',()=>{
 const receipt={host:'example.org',dns_addresses:Array(64).fill('2606:4700:4700::1111'),attempts:Array(6).fill({result:'WEB_CONNECT_FAILED',selected_ip:'8.8.8.8',stage:'CONNECT',retry_before_request:true}),dns:{mode:'SECURE_DOH',resolvers:Array.from({length:4},(_,i)=>({resolver:'r'+i,status:'RESOLVED',queries:[{addresses:Array(64).fill('2606:4700:4700::1111')}]}))}};
 const summary=networkSummary(receipt);assert.ok(Buffer.byteLength(JSON.stringify(summary))<4096);assert.equal(receipt.dns_addresses.length,64);assert.equal(summary.dns_addresses.length,6);
});
