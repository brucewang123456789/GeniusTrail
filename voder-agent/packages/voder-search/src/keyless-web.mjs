/**
 * Keyless, bounded public web discovery and source fetch for Voder.
 * The searched public endpoint is not a contracted/search-SLA API.
 * No model-controlled hosts, no private-network egress; one narrowly scoped Bing RSS redirect.
 * All retrieved content is untrusted context (never a VGR receipt).
 */
import {siteScope,matchesSiteScope,rankResearchCandidates} from './research.mjs';
import {createHash} from 'node:crypto';
import {request as httpsRequest} from 'node:https';
import {request as httpRequest} from 'node:http';
import {lookup as dnsLookup} from 'node:dns';
import {isIP, BlockList} from 'node:net';

const SECRET = /(?:api[_-]?key|access[_-]?token|auth[_-]?token|password|secret)\s*[:=]\s*\S{8,}|\bsk-[A-Za-z0-9_-]{16,}|\bgh[opurs]_[A-Za-z0-9]{16,}/i;
const LIMIT_SEARCH = 200 * 1024;
const LIMIT_PAGE = 600 * 1024;
const BLOCK = new BlockList();
for (const [net,pre] of [
 ['0.0.0.0',8],['10.0.0.0',8],['100.64.0.0',10],['127.0.0.0',8],
 ['169.254.0.0',16],['172.16.0.0',12],['192.0.0.0',24],
 ['192.0.2.0',24],['192.168.0.0',16],['198.18.0.0',15],
 ['198.51.100.0',24],['203.0.113.0',24],['224.0.0.0',4],['240.0.0.0',4]
])BLOCK.addSubnet(net,pre,'ipv4');
for(const [net,pre] of [['::',128],['::1',128],['fc00::',7],['fe80::',10],
 ['ff00::',8],['2001:db8::',32]])BLOCK.addSubnet(net,pre,'ipv6');
export function publicAddress(value){
 if(typeof value!=='string')return false;
 const v=value.split('%')[0];
 const fam=isIP(v);
 if(!fam)return false;
 if(fam===6){
  // Accept only global unicast IPv6; reject mapped, NAT64, 6to4 and Teredo.
  const global=new BlockList();global.addSubnet('2000::',3,'ipv6');
  const tunnel=new BlockList();tunnel.addSubnet('2001::',32,'ipv6');tunnel.addSubnet('2002::',16,'ipv6');
  if(!global.check(v,'ipv6')||tunnel.check(v,'ipv6'))return false;
 }
 return !BLOCK.check(v,fam===4?'ipv4':'ipv6');
}
function fail(code){const e=new Error(code);e.code=code;throw e;}
export function checkedPublicUrl(raw){
 if(typeof raw!=='string'||raw.length>1300||/[\x00-\x20\x7f]/.test(raw)||SECRET.test(raw))fail('UNSAFE_SOURCE_URL');
 let u;try{u=new URL(raw);}catch{fail('UNSAFE_SOURCE_URL');}
 const host=u.hostname.toLowerCase();
 if(u.protocol!=='https:'||u.username||u.password||u.port||isIP(host)||
 !/^[a-z0-9.-]+$/.test(host)||!host.includes('.')||host.includes('..')||
 host.startsWith('.')||host.endsWith('.')||host.length>253||host.split('.').some(x=>x.length>63||x.startsWith('-')||x.endsWith('-'))||
 /(?:^|\.)(?:localhost|local|internal|localdomain|invalid|onion)$/.test(host))fail('UNSAFE_SOURCE_URL');
 for(const [k,v] of u.searchParams){if(/^(?:api_?key|token|access_?token|auth_?token|secret|password|credential|authorization)$/i.test(k)||SECRET.test(v))fail('UNSAFE_SOURCE_URL');}
 u.hash='';return u.href;
}
const decode = s => s.replace(/<!\[CDATA\[([\s\S]*?)\]\]>/g,'$1').replace(/&(#x[\da-f]+|#\d+|amp|lt|gt|quot|apos|nbsp|ndash|mdash);/gi,(_,v)=>{
 const names={amp:'&',lt:'<',gt:'>',quot:'"',apos:"'",nbsp:' ',ndash:'–',mdash:'—'};
 if(v[0]==='#'){
  const n=v[1]?.toLowerCase()==='x'?parseInt(v.slice(2),16):parseInt(v.slice(1),10);
  return Number.isSafeInteger(n)&&n>0&&n<=0x10ffff&&!(n>=0xd800&&n<=0xdfff)?String.fromCodePoint(n):' ';
 }
 return names[v.toLowerCase()]??' ';
});
function flatten(raw){
 return decode(raw).replace(/<(?:script|style|noscript|svg|iframe|template|form|nav|footer|header)\b[^>]*>[\s\S]*?<\/\s*(?:script|style|noscript|svg|iframe|template|form|nav|footer|header)\s*>/gi,' ')
  .replace(/<(?:br|\/p|\/div|\/li|\/h[1-6]|\/section|\/article)\b[^>]*>/gi,'\n')
  .replace(/<[^>]*>/g,' ').replace(/\r/g,'').replace(/[\t ]+/g,' ').replace(/\n\s*\n\s*\n/g,'\n\n').trim();
}
export function parseBingRss(xml){
 if(typeof xml!=='string'||!/<rss\b/i.test(xml)||!/<channel\b/i.test(xml))fail('WEB_PROVIDER_INVALID');
 const matches=xml.match(/<item(?:\s[^>]*)?>[\s\S]*?<\/item>/gi)??[];
 const rows=[];
 const tag=(x,n)=>(x.match(new RegExp(`<${n}(?:\\s[^>]*)?>([\\s\\S]*?)<\\/${n}>`,'i'))??[])[1]??'';
 for(const item of matches.slice(0,16)){
  const url=decode(tag(item,'link')).trim();
  try{checkedPublicUrl(url);}catch{continue;}
  const title=flatten(tag(item,'title')).slice(0,160);
  const excerpt=flatten(tag(item,'description')).slice(0,1200);
  if(title||excerpt)rows.push({url,title,excerpt});
  if(rows.length>=16)break;
 }
 return rows;
}
/** Parse the public HTML search response only; never execute its markup.
 * Redirect wrappers are decoded as source references, not followed as requests. */
export function parseDuckDuckGoHtml(html){
 if(typeof html!=='string')fail('WEB_PROVIDER_INVALID');
 if(/id=["'](?:challenge-form|anomaly-form)["']|anomaly\.js|bots use DuckDuckGo/i.test(html))fail('WEB_SEARCH_CHALLENGE');
 const cleaned=html.replace(/<!--[^]*?-->|<(?:script|style)\b[^>]*>[^]*?<\/(?:script|style)\s*>/gi,' ');
 const anchors=[...cleaned.matchAll(/<a\b([^>]{0,8192})>([^]*?)<\/a\s*>/gi)].filter(m=>/\bclass\s*=\s*["'][^"']*\bresult__a\b/i.test(m[1])).slice(0,16);
 if(!anchors.length&&!/no-results|No results found/i.test(cleaned))fail('WEB_PROVIDER_INVALID');
 const rows=[],seen=new Set();
 for(let i=0;i<anchors.length;i++){
  const anchor=anchors[i],raw=decode((anchor[1].match(/\bhref\s*=\s*["']([^"']*)["']/i)??[])[1]??'');
  let u;try{u=new URL(raw,'https://html.duckduckgo.com/');}catch{continue;}
  if(['duckduckgo.com','www.duckduckgo.com','html.duckduckgo.com'].includes(u.hostname)&&u.pathname==='/l/'){
   const target=u.searchParams.get('uddg');if(!target)continue;
   try{u=new URL(target);}catch{continue;}
  }
  let url;try{url=checkedPublicUrl(u.href);}catch{continue;}
  if(seen.has(url))continue;seen.add(url);
  const tail=cleaned.slice(anchor.index+anchor[0].length,anchors[i+1]?.index??cleaned.length);
  const description=(tail.match(/<(?:a|div|span)\b[^>]*class\s*=\s*["'][^"']*\bresult__snippet\b[^"']*["'][^>]*>([^]*?)<\/(?:a|div|span)\s*>/i)??[])[1]??'';
  const title=flatten(anchor[2]).slice(0,160),excerpt=flatten(description).slice(0,1200);
  if(title||excerpt)rows.push({url,title,excerpt});
 }
 return rows;
}
function validateQuery(query){if(typeof query!=='string'||!query.trim()||query.length>180||/[\x00-\x1f\x7f]/.test(query)||SECRET.test(query))fail('INVALID_SEARCH_QUERY');return query.trim();}
/** Node lookup callback has two distinct contracts: options.all=true expects an
 * array of {address,family}; otherwise it expects (address,family). DNS is
 * resolved exactly once, every returned address is checked, then one approved
 * address is passed to the socket so Node cannot resolve a different target.
 * The last argument is a test-only resolver injection, never host supplied.
 */
export function resolvePinned(host,options,cb,lookupImpl=dnsLookup){
 lookupImpl(host,{all:true},(err,addresses)=>{
  if(err)return cb(Object.assign(new Error('WEB_DNS_FAILED'),{code:'WEB_DNS_FAILED'}));
  if(!Array.isArray(addresses)||!addresses.length||
    addresses.some(a=>!a||![4,6].includes(a.family)||!publicAddress(a.address)))
   return cb(Object.assign(new Error('DNS_PUBLIC_ADDRESS_REQUIRED'),{code:'DNS_PUBLIC_ADDRESS_REQUIRED'}));
  const family=options?.family??0;
  if(![0,4,6].includes(family))
   return cb(Object.assign(new Error('DNS_PUBLIC_ADDRESS_REQUIRED'),{code:'DNS_PUBLIC_ADDRESS_REQUIRED'}));
  const usable=family?addresses.filter(a=>a.family===family):addresses;
  if(!usable.length)
   return cb(Object.assign(new Error('DNS_PUBLIC_ADDRESS_REQUIRED'),{code:'DNS_PUBLIC_ADDRESS_REQUIRED'}));
  const chosen=usable.find(a=>a.family===4)??usable[0];
  // When net autoSelectFamily is active, all:true is mandatory: a plain IP
  // string is rejected by Node with ERR_INVALID_IP_ADDRESS.
  if(options?.all===true)return cb(null,[{address:chosen.address,family:chosen.family}]);
  return cb(null,chosen.address,chosen.family);
 });
}
// Only the trusted host can opt into an HTTP loopback SearXNG service.
function trustedSearxngEndpoint(raw){
 if(typeof raw!=='string'||raw.length>1300)fail('INVALID_SEARCH_PROVIDER');
 let u;try{u=new URL(raw);}catch{fail('INVALID_SEARCH_PROVIDER');}
 if(u.username||u.password||u.hash||u.search|| !['','/'].includes(u.pathname))fail('INVALID_SEARCH_PROVIDER');
 if(u.protocol==='http:' && !['127.0.0.1','[::1]'].includes(u.hostname))fail('INVALID_SEARCH_PROVIDER');
 if(u.protocol!=='https:' && u.protocol!=='http:')fail('INVALID_SEARCH_PROVIDER');
 if(u.protocol==='https:')checkedPublicUrl(u.origin+'/');
 return u.origin;
}
/** Explicitly scoped Bing RSS regional redirect: identical path/query, no other host,
 * no scheme downgrade, no credentials or fragment. Never enabled for fetch/SearXNG. */
export function approvedBingRssRedirect(source,location){
 let from,to;
 try{from=new URL(source);to=new URL(location);}catch{fail('WEB_REDIRECT_DENIED');}
 if(from.protocol!=='https:'||from.hostname!=='www.bing.com'||from.pathname!=='/search'||
    from.searchParams.get('format')!=='rss'||!from.searchParams.has('q')||
    from.username||from.password||from.port||from.hash||
    to.protocol!=='https:'||to.hostname!=='cn.bing.com'||to.pathname!=='/search'||
    to.username||to.password||to.port||to.hash||
    to.search!==from.search)fail('WEB_REDIRECT_DENIED');
 checkedPublicUrl(to.href);
 return to.href;
}
/** Each hop is validated before DNS; each socket receives only pinned public IPs.
 * One deadline and one abort signal cover headers, body and the redirect chain.
 * requestImpl is a host/test seam; never exposed to model tool arguments. */
export async function pinnedGet(url,{signal,maxBytes=LIMIT_SEARCH,timeoutMs=12000,contentTypes=['application/rss+xml','application/xml','text/xml','application/json','text/html','text/plain'],allowLoopback=false,bingRssRedirect=false,sourceRedirects=false,maxRedirects=4,urlAllowed=()=>true,returnMetadata=false,requestImpl}={}){
 if(!Number.isSafeInteger(maxBytes)||maxBytes<1||maxBytes>LIMIT_PAGE||!Number.isSafeInteger(timeoutMs)||timeoutMs<1000||timeoutMs>30000||!Number.isSafeInteger(maxRedirects)||maxRedirects<0||maxRedirects>5||typeof urlAllowed!=='function')fail('INVALID_SEARCH_POLICY');
 const deadline=Date.now()+timeoutMs,visited=new Set(),redirects=[];
 let current=url;
 while(true){
  if(signal?.aborted)fail('CANCELLED');
  let u;try{u=new URL(current);}catch{fail('UNSAFE_SOURCE_URL');}
  const loopback=allowLoopback&&u.protocol==='http:'&&['127.0.0.1','[::1]'].includes(u.hostname);
  if(!loopback)checkedPublicUrl(current);
  if(!urlAllowed(current))fail('WEB_DOMAIN_DENIED');
  if(visited.has(u.href))fail('WEB_REDIRECT_LOOP');
  visited.add(u.href);
  const remaining=deadline-Date.now();if(remaining<=0)fail('WEB_TIMEOUT');
  const out=await new Promise((resolve,reject)=>{
   let done=false,req,res,timer;
   const finish=(err,result)=>{
    if(done)return;done=true;clearTimeout(timer);signal?.removeEventListener('abort',onAbort);
    if(err){req?.destroy();res?.destroy();reject(err);}else resolve(result);
   };
   const onAbort=()=>finish(Object.assign(new Error('CANCELLED'),{code:'CANCELLED'}));
   timer=setTimeout(()=>finish(Object.assign(new Error('WEB_TIMEOUT'),{code:'WEB_TIMEOUT'})),remaining);
   signal?.addEventListener('abort',onAbort,{once:true});
   try{
    req=(requestImpl??(loopback?httpRequest:httpsRequest))(u,{method:'GET',agent:false,
     headers:{Accept:contentTypes.join(', '),'Accept-Encoding':'identity','User-Agent':'Voder-EvidenceSearch/2.0'},
     ...(loopback?{}:{lookup:resolvePinned,rejectUnauthorized:true,servername:u.hostname})},incoming=>{
     res=incoming;const status=res.statusCode??0;
     if([301,302,303,307,308].includes(status)){
      const location=String(res.headers.location??'');
      finish(null,{status,location});res.destroy();return;
     }
     if(status<200||status>299){
      const code=status===404||status===410?'WEB_SOURCE_NOT_FOUND':status===403?'WEB_ACCESS_DENIED':status===429?'WEB_RATE_LIMITED':'WEB_HTTP_ERROR';
      finish(Object.assign(new Error(code),{code,httpStatus:status}));return;
     }
     const ct=String(res.headers['content-type']??'').split(';')[0].toLowerCase().trim();
     if(!contentTypes.some(t=>ct===t||(t==='text/html'&&ct==='application/xhtml+xml'))){finish(Object.assign(new Error('WEB_CONTENT_TYPE_DENIED'),{code:'WEB_CONTENT_TYPE_DENIED'}));return;}
     if(res.headers['content-encoding']&&res.headers['content-encoding']!=='identity'){
      finish(Object.assign(new Error('WEB_CONTENT_ENCODING_DENIED'),{code:'WEB_CONTENT_ENCODING_DENIED'}));return;
     }
     const chunks=[];let size=0,ended=false;
     res.on('data',buf=>{
      size+=buf.length;
      if(size>maxBytes){finish(Object.assign(new Error('SEARCH_RESPONSE_TOO_LARGE'),{code:'SEARCH_RESPONSE_TOO_LARGE'}));return;}
      chunks.push(buf);
     });
     res.on('end',()=>{ended=true;finish(null,{status,body:Buffer.concat(chunks).toString('utf8'),raw_sha256:createHash('sha256').update(Buffer.concat(chunks)).digest('hex')});});
     res.on('error',err=>finish(err));
     res.on('aborted',()=>finish(Object.assign(new Error('WEB_RESPONSE_INCOMPLETE'),{code:'WEB_RESPONSE_INCOMPLETE'})));
     res.on('close',()=>{if(!ended&&!done)finish(Object.assign(new Error('WEB_RESPONSE_INCOMPLETE'),{code:'WEB_RESPONSE_INCOMPLETE'}));});
    });
    req.on('error',err=>finish(err));
    if(signal?.aborted)onAbort();
    if(!done)req.end();
   }catch(err){finish(err);}
  });
  if(out.location!==undefined){
   if(redirects.length>=maxRedirects)fail('WEB_REDIRECT_LIMIT');
   let next;
   if(bingRssRedirect&&!redirects.length)next=approvedBingRssRedirect(current,out.location);
   else if(sourceRedirects){
    try{next=new URL(out.location,current).href;}catch{fail('WEB_REDIRECT_DENIED');}
    next=checkedPublicUrl(next);
    if(!urlAllowed(next))fail('WEB_DOMAIN_DENIED');
   }else fail('WEB_REDIRECT_DENIED');
   redirects.push({from:current,to:next,status:out.status});current=next;continue;
  }
  return returnMetadata?{...out,url:current,redirects}:out.body;
 }
}
/** No paid search API, no model/API keys.  Search may be blocked or unavailable.
 * `getImpl` is a test seam only; production defaults to pinned HTTPS transport.
 */
export class KeylessWebProvider {
 constructor({engine='bing-rss',searxngUrl,timeoutMs=12000,getImpl=pinnedGet}={}){
  if(!['bing-rss','searxng'].includes(engine)||!Number.isSafeInteger(timeoutMs)||timeoutMs<1000||timeoutMs>30000||typeof getImpl!=='function')fail('INVALID_SEARCH_PROVIDER');
  this.engine=engine;this.timeoutMs=timeoutMs;this.get=getImpl;
  this.searxngUrl=engine==='searxng'?trustedSearxngEndpoint(searxngUrl):undefined;
 }
 async search(query,{signal,acquireRequest,sourceAllowed=()=>true}={}){
  query=validateQuery(query);
  const scopes=siteScope(query),deadline=Date.now()+this.timeoutMs;
  const u=this.engine==='bing-rss'?new URL('https://www.bing.com/search'):new URL(this.searxngUrl+'/search');
  u.searchParams.set('q',query);
  if(this.engine==='bing-rss')u.searchParams.set('format','rss');
  else u.searchParams.set('format','json');
  const raw=await this.get(u.href,{signal,timeoutMs:this.timeoutMs,maxBytes:LIMIT_SEARCH,
    ...(this.engine==='searxng'&&u.protocol==='http:'?{allowLoopback:true}:{}),
    contentTypes:this.engine==='bing-rss'?['application/rss+xml','application/xml','text/xml','text/plain']:['application/json'],
      bingRssRedirect:this.engine==='bing-rss'});
  let rows;
  if(this.engine==='bing-rss')rows=parseBingRss(raw);
  else{
   let obj;try{obj=JSON.parse(raw);}catch{fail('WEB_PROVIDER_INVALID');}
   if(!Array.isArray(obj?.results))fail('WEB_PROVIDER_INVALID');
   rows=obj.results.slice(0,16).map(r=>({url:r?.url,title:r?.title,excerpt:r?.content??r?.snippet??''}));
  }
  const eligible=rows.filter(r=>{
   try{return !!checkedPublicUrl(r.url)&&sourceAllowed(r.url)&&matchesSiteScope(r.url,scopes);}catch{return false;}
  });
  const quality=rankResearchCandidates(query,eligible,{maxResults:8});
  const attempts=[{provider:this.engine,status:quality.length?'CANDIDATES_FOUND':eligible.length?'WEAK_RESULTS':'NO_SCOPED_RESULTS',returned:rows.length,eligible:eligible.length}];
  const finish=out=>Object.defineProperty(out.slice(0,4),'discovery_attempts',{value:attempts,enumerable:false});
  // A secondary request is never hidden from the host budget. Strong existing
  // results and explicitly configured SearXNG retain the original path.
  if(this.engine!=='bing-rss'||quality.length||typeof acquireRequest!=='function')return finish(eligible);
  const remaining=deadline-Date.now();
  if(remaining<1000){attempts.push({provider:'duckduckgo-html',status:'SKIPPED_DEADLINE'});return finish(eligible);}
  try{
   if(acquireRequest()===false){attempts.push({provider:'duckduckgo-html',status:'SKIPPED_BUDGET'});return finish(eligible);}
  }catch(e){
   if(e?.code==='CANCELLED')throw e;
   if(e?.code!=='WEB_QUERY_BUDGET')throw e;
   attempts.push({provider:'duckduckgo-html',status:'SKIPPED_BUDGET'});return finish(eligible);
  }
  if(signal?.aborted)fail('CANCELLED');
  const fallback=new URL('https://html.duckduckgo.com/html/');fallback.searchParams.set('q',query);
  try{
   const html=await this.get(fallback.href,{signal,timeoutMs:remaining,maxBytes:LIMIT_SEARCH,
    contentTypes:['text/html','application/xhtml+xml'],sourceRedirects:true,maxRedirects:2,
    urlAllowed:raw=>{try{const next=new URL(raw);return next.origin===fallback.origin&&next.pathname===fallback.pathname&&next.searchParams.get('q')===query;}catch{return false;}}});
   const parsed=parseDuckDuckGoHtml(html);
   const secondary=parsed.filter(r=>sourceAllowed(r.url)&&matchesSiteScope(r.url,scopes));
   const ranked=rankResearchCandidates(query,secondary,{maxResults:8});
   attempts.push({provider:'duckduckgo-html',status:ranked.length?'CANDIDATES_FOUND':secondary.length?'WEAK_RESULTS':'NO_SCOPED_RESULTS',returned:parsed.length,eligible:secondary.length});
   // Do not discard legitimate primary hits if the backup is also weak.
   const merged=[...ranked,...eligible,...secondary],seen=new Set();
   return finish(merged.filter(r=>{if(seen.has(r.url))return false;seen.add(r.url);return true;}));
  }catch(e){
   if(e?.code==='CANCELLED'||signal?.aborted)fail('CANCELLED');
   const codes=new Set(['WEB_SEARCH_CHALLENGE','WEB_TIMEOUT','WEB_PROVIDER_INVALID','WEB_RATE_LIMITED','WEB_ACCESS_DENIED','WEB_DNS_FAILED','DNS_PUBLIC_ADDRESS_REQUIRED','WEB_DOMAIN_DENIED','WEB_REDIRECT_DENIED','WEB_REDIRECT_LIMIT','WEB_REDIRECT_LOOP','WEB_CONTENT_TYPE_DENIED','WEB_HTTP_ERROR','SEARCH_RESPONSE_TOO_LARGE']);
   attempts.push({provider:'duckduckgo-html',status:'FAILED',code:codes.has(e?.code)?e.code:'WEB_PROVIDER_FAILED'});
   return finish(eligible);
  }
 }

 async extract(url,{focus='',signal,urlAllowed=()=>true}={}){
  const approved=checkedPublicUrl(url);
  if(typeof focus!=='string'||focus.length>140||/[\x00-\x1f\x7f]/.test(focus)||SECRET.test(focus))fail('INVALID_SEARCH_QUERY');
  const page=await this.get(approved,{signal,timeoutMs:this.timeoutMs,maxBytes:LIMIT_PAGE,
    sourceRedirects:true,urlAllowed,returnMetadata:true,
    contentTypes:['text/html','text/plain','text/markdown','application/xhtml+xml']});
  const raw=typeof page==='string'?page:page.body;
  const text=flatten(raw);
  if(!text)fail('WEB_EXTRACTION_FAILED');
  return {url:approved,final_url:page?.url??approved,redirects:page?.redirects??[],
   raw_sha256:page?.raw_sha256??createHash('sha256').update(raw).digest('hex'),text:text.slice(0,350000)};
 }
}
