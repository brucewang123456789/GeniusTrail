/** First-party technical discovery probe; NOT a drop-in production search provider. */
import {createHash} from 'node:crypto';
import {officialGet,officialUrl} from './official-http.mjs';
import {rustSymbolRerank} from './rust-rank.mjs';
import {parseSphinxInventory,findSphinxEntries,parseGoPkgsiteSearch,goSearchApiUrl} from './sphinx-go.mjs';
const SENSITIVE_QUERY=/(?:\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|password|secret)\s*[:=]\s*\S{8,}|\bsk-[a-zA-Z0-9_-]{16,}|\bgh[opurs]_[A-Za-z0-9]{16,})/i;
const IGNORE=new Set(['the','and','for','official','reference','docs','doc','documentation','api','guide','language','programming','python','typescript','rust','go','golang','std','library','class','code','with','path','search','site','html','error','security','how','use','to']);
const words=s=>[...new Set((String(s).toLowerCase().match(/[a-z][a-z0-9_-]{1,}/g)??[]).filter(x=>!IGNORE.has(x)))].slice(0,12);
export const sha256=data=>createHash('sha256').update(data).digest('hex');
const unhtml=s=>s.replace(/&amp;/gi,'&').replace(/&lt;/gi,'<').replace(/&gt;/gi,'>').replace(/&quot;/gi,'"').replace(/&#(\d+);/g,(_,n)=>String.fromCharCode(Number(n))).replace(/<[^>]*>/g,' ').replace(/\s+/g,' ').trim();
function scoreCandidate(query,item){
 const terms=words(query),location=new URL(item.url).pathname.toLowerCase(),text=`${item.title??''} ${item.excerpt??''}`.toLowerCase();
 let score=0;
 for(const t of terms){if(location.includes(t))score+=10;if(text.includes(t))score+=3;}
 if(/\/(?:index\.html)?$/.test(location))score-=15;
 // Source tier is derived from a verified first-party index, not from a
 // hostname alone (pkg.go.dev hosts both standard and third-party code).
 if(item.source==='PYTHON_SPHINX_OFFICIAL' && terms[0] && (text.startsWith(terms[0]+'.')||text.startsWith(terms[0]+' ')))score+=75;
 if(item.source==='GO_STD_OFFICIAL_INDEX')score+=85;
 if(item.source==='GO_OFFICIAL_DOC_INDEX'||item.source==='GO_OFFICIAL_DOC_LINKED_BY_GO_TEAM')score+=55;
 if(item.source==='GO_OFFICIAL_MODULE_INDEX')score+=80;
 return score + (new URL(item.url).hostname==='doc.rust-lang.org'?rustSymbolRerank(query,item):0);
}
/** Parse only real anchor URLs in an official page; never invent links or invoke JavaScript. */
export function extractOfficialAnchors(html,base,{max=15000}={}){
 if(typeof html!=='string'||html.length>4000000)throw Error('INVALID_INDEX_PAGE');
 const result=[],seen=new Set();let m;
 const regex=/<a\b([^>]*\bhref\s*=\s*(?:"[^"]*"|'[^']*')[^>]*)>([\s\S]*?)<\/a\s*>/gi;
 while((m=regex.exec(html))&&result.length<max){
  const href=(m[1].match(/\bhref\s*=\s*(["'])(.*?)\1/i)??[])[2];
  if(!href||href.startsWith('#')||href.startsWith('javascript:'))continue;
  let url;try{url=new URL(unhtml(href),base).href;officialUrl(url);}catch{continue;}
  const u=new URL(url);u.hash='';url=u.href;
  if(!seen.has(url)){seen.add(url);result.push({url,title:unhtml(m[2]).slice(0,250),source:'OFFICIAL_HTML_INDEX'});}
 }
 return result;
}
export function extractSitemapUrls(xml,host,{max=20000}={}){
 if(typeof xml!=='string'||xml.length>4000000)throw Error('INVALID_SITEMAP');
 const rows=[];const seen=new Set();
 for(const m of xml.matchAll(/<loc>\s*([^<]{5,1500})\s*<\/loc>/gi)){
  const candidate=unhtml(m[1]);let u;try{u=officialUrl(candidate);}catch{continue;}
  if(u.hostname!==host)continue;
  const url=u.href;if(seen.has(url))continue;seen.add(url);rows.push({url,title:u.pathname,source:'OFFICIAL_SITEMAP'});
  if(rows.length>=max)break;
 }
 return rows;
}
const fixWords=q=>q.replace(/\b(site:\S+|official|documentation|api|reference)\b/gi,' ').replace(/\s+/g,' ').trim();
/** Bounded, time-limited cache of immutable official discovery indices only.
 * Page Fetch still goes to the source on every verification and is never
 * silently substituted with search-index metadata. */
export function createOfficialIndexCache({get=officialGet,ttlMs=300000,maxEntries=12,clock=Date.now}={}){
 if(typeof get!=='function'||typeof clock!=='function'||!Number.isSafeInteger(ttlMs)||ttlMs<0||ttlMs>900000||
    !Number.isSafeInteger(maxEntries)||maxEntries<1||maxEntries>64)throw Error('INVALID_INDEX_CACHE_POLICY');
 const indexPaths=new Set(['/3/objects.inv','/std/all.html','/error_codes/error-index.html',
  '/sitemap.xml','/docs/','/std','/doc/security/','/doc/tutorial/',
  '/golang.org/x/vuln']);
 const cache=new Map();
 return async function cachedOfficialIndex(url,options={}){
  const u=officialUrl(url),key=u.href;
  const eligible=indexPaths.has(u.pathname)&&!u.search&&!u.hash&&ttlMs>0;
  const t=clock(),old=eligible?cache.get(key):null;
  if(old && t>=old.at && t-old.at<ttlMs){
   cache.delete(key);cache.set(key,old);
   return {...old.response,bytes:Buffer.from(old.response.bytes)};
  }
  if(old)cache.delete(key);
  const response=await get(url,options);
  if(eligible&&Buffer.isBuffer(response?.bytes)&&response.bytes.length<=4*1024*1024){
   cache.set(key,{at:clock(),response:{...response,bytes:Buffer.from(response.bytes)}});
   while(cache.size>maxEntries)cache.delete(cache.keys().next().value);
  }
  return response;
 };
}

/** Returns candidates backed by a discovered official index, never a guessed endpoint. */
export async function discoverOfficial(language,query,{get=officialGet,trace=()=>{}}={}){
 if(typeof query!=='string'||!query.trim()||query.length>180||/[\x00-\x1f\x7f]/.test(query)||SENSITIVE_QUERY.test(query))throw Error('INVALID_OFFICIAL_QUERY');
 const safeGet=async(u,options)=>{trace({stage:'REQUEST',host:new URL(u).hostname,path:new URL(u).pathname});try{const result=await get(u,options);trace({stage:'HTTP_OK',host:new URL(u).hostname,bytes:result.bytes?.length??0});return result;}catch(e){trace({stage:'HTTP_FAIL',host:new URL(u).hostname,code:String(e.code??e.message??'UNKNOWN').slice(0,60)});throw e}};
 let rows=[];
 if(language==='Python'){
  const data=await safeGet('https://docs.python.org/3/objects.inv',{limit:1024*1024});
  const entries=parseSphinxInventory(data.bytes);
  rows=findSphinxEntries(entries,fixWords(query),{maxResults:35}).map(x=>({url:x.url,title:x.name,excerpt:x.kind,source:'PYTHON_SPHINX_OFFICIAL'}));
 } else if(language==='Rust'){
  const urls=/\bE\d{4}\b/i.test(query)?['https://doc.rust-lang.org/error_codes/error-index.html']:['https://doc.rust-lang.org/std/all.html'];
  for(const src of urls){const {bytes}=await safeGet(src,{limit:4*1024*1024});rows.push(...extractOfficialAnchors(bytes.toString('utf8'),src));}
 } else if(language==='TypeScript'){
  // Prefer first-party sitemap; if it is absent, use first-party docs index.
  try{
   const {bytes}=await safeGet('https://www.typescriptlang.org/sitemap.xml',{limit:4*1024*1024});
   rows=extractSitemapUrls(bytes.toString('utf8'),'www.typescriptlang.org');
  }catch(e){trace({stage:'SITEMAP_UNAVAILABLE',code:String(e.code??e.message)});}
  if(!rows.length){
   const src='https://www.typescriptlang.org/docs/';
   const {bytes}=await safeGet(src,{limit:4*1024*1024});
   rows=extractOfficialAnchors(bytes.toString('utf8'),src);
  }
 } else if(language==='Go'){
  const terms=words(query);
  const securityIntent=/govulncheck|vulnerab|security|cve|osv/i.test(query);
  // Go's maintained golang.org/x/vuln module is an independently hosted,
  // first-party index. pkg.go.dev's *global search* is NOT a first-party index:
  // only links actually observed on this trusted Go-team module page qualify.
  // This avoids repeated go.dev timeouts while retaining source provenance.
  if(securityIntent){
   const moduleRoot='https://pkg.go.dev/golang.org/x/vuln';
   try{
    const {bytes}=await safeGet(moduleRoot,{limit:4*1024*1024,timeout:5000});
    const anchors=extractOfficialAnchors(bytes.toString('utf8'),moduleRoot,{max:3000});
    for(const x of anchors){
     const u=new URL(x.url),path=u.pathname;
     const relevant=terms.some(t=>path.toLowerCase().includes(t)||x.title.toLowerCase().includes(t));
     if(!relevant||u.search||u.hash)continue;
     if(u.hostname==='pkg.go.dev'&&
        /^\/golang\.org\/x\/vuln(?:(?:@|%40)v\d+\.\d+\.\d+(?:-[a-z0-9.-]+)?(?:\+incompatible)?)?(?:\/|$)/i.test(path))
      rows.push({...x,source:'GO_OFFICIAL_MODULE_INDEX'});
     else if(u.hostname==='go.dev'&&path.startsWith('/doc/'))
      rows.push({...x,source:'GO_OFFICIAL_DOC_LINKED_BY_GO_TEAM'});
    }
    trace({stage:'GO_MODULE_INDEX',host:'pkg.go.dev',candidates:rows.length});
   }catch(e){trace({stage:'GO_MODULE_INDEX_UNAVAILABLE',host:'pkg.go.dev',code:String(e.code??e.message).slice(0,60)});}
  }
  // Go's own doc indices remain an independent failover, but a successful
  // qualified first-party module hit should not incur two more slow requests.
  const indexes=securityIntent
   ? (rows.length?[]:['https://go.dev/doc/tutorial/','https://go.dev/doc/security/'])
   : ['https://pkg.go.dev/std'];
  let indexSuccess=rows.length>0;
  for(const src of indexes){
   try{
    const {bytes}=await safeGet(src,{limit:4*1024*1024,timeout:securityIntent?4000:12000});
    const anchors=extractOfficialAnchors(bytes.toString('utf8'),src,{max:20000});
    const chosen=src.endsWith('/std')
      ?anchors.filter(x=>{
         const u=new URL(x.url);
         const canonicalPath=u.pathname.replace(/@go\d[\da-z.-]*$/i,'');
         return u.hostname==='pkg.go.dev'&&!u.search&&!u.hash&&
           /^\/[a-z][a-z0-9_]*(?:\/[a-z][a-z0-9_]*)*$/.test(canonicalPath)&&
           terms.some(t=>canonicalPath.toLowerCase()==='/'+t||canonicalPath.toLowerCase().endsWith('/'+t));
        }).map(x=>({...x,source:'GO_STD_OFFICIAL_INDEX'}))
      :anchors.filter(x=>{
         const u=new URL(x.url);
         return u.hostname==='go.dev'&&u.pathname.startsWith('/doc/')&&
           terms.some(t=>u.pathname.toLowerCase().includes(t)||x.title.toLowerCase().includes(t));
        }).map(x=>({...x,source:'GO_OFFICIAL_DOC_INDEX'}));
    rows.push(...chosen);indexSuccess=true;
    if(securityIntent&&rows.length>0)break; // bounded, evidence-first
   }catch(e){trace({stage:'GO_OFFICIAL_INDEX_UNAVAILABLE',host:new URL(src).hostname,code:String(e.code??e.message).slice(0,60)});}
  }
  // Bounded fallback: the result is marked as package-search data rather than
  // a standard-library authority, and is ranked below the official indexes.
  if(!rows.length){
   try{
    const filter=fixWords(query).replace(/\bGo\b/gi,' ').trim();
    const api=goSearchApiUrl(filter.slice(0,100),{maxResults:30});
    const {bytes}=await safeGet(api,{limit:1024*1024});
    rows.push(...parseGoPkgsiteSearch(bytes.toString('utf8'),{maxResults:30}));
   }catch(e){trace({stage:'GO_PACKAGE_FALLBACK_FAILED',code:String(e.code??e.message).slice(0,60)});if(!indexSuccess)throw e;}
  }
 } else throw Error('UNSUPPORTED_LANGUAGE');
 const uniq=new Map();
 for(const row of rows){try{const u=officialUrl(row.url);if(!uniq.has(u.href))uniq.set(u.href,row);}catch{}}
 return [...uniq.values()].map(row=>({...row,score:scoreCandidate(query,row)})).filter(row=>row.score>0)
    .sort((a,b)=>b.score-a.score||a.url.localeCompare(b.url)).slice(0,12);
}
/** Validate the *content* of a discovered source, not just a hostname or search snippet. */
export async function verifySource(candidate,focus,{get=officialGet}={}){
 const u=officialUrl(candidate.url);
 const raw=await get(u.href,{limit:4*1024*1024});
 const text=unhtml(raw.bytes.toString('utf8'));
 const term=String(focus).toLowerCase().trim();
 const needle=term.length>=2&&term.length<=120&&/^[\p{L}\p{N}\s._:-]+$/u.test(term)?term:null;
 const hit=needle?text.toLowerCase().indexOf(needle):-1;
 return {url:u.href,content_sha256:sha256(raw.bytes),bytes:raw.bytes.length,focus_found:hit>=0,
  excerpt:hit>=0?text.slice(Math.max(0,hit-150),hit+450):'',authority:'CONTEXT_ONLY',trust:candidate.source==='PKGSITE_OFFICIAL_API'?'UNTRUSTED_THIRD_PARTY_PACKAGE':'UNTRUSTED_OFFICIAL_SOURCE'};
}
