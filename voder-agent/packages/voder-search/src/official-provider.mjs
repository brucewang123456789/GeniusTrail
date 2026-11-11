/** v3.10 first-party official-document discovery, wired to Voder's existing host authorization. */
import {createHash} from 'node:crypto';
import {discoverOfficial,createOfficialIndexCache} from './official-v310/discover.mjs';
import {officialGet,officialUrl} from './official-v310/official-http.mjs';
const code=(c)=>Object.assign(new Error(c),{code:c});
const supported=new Set(['Python','TypeScript','Rust','Go']);
const SENSITIVE=/(?:\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|password|secret)\s*[:=]\s*\S{8,}|\bsk-[A-Za-z0-9_-]{16,}|\bgh[opurs]_[A-Za-z0-9]{16,})/i;
export function officialLanguage(query,language){
 if(language!==undefined){if(!supported.has(language))throw code('UNSUPPORTED_OFFICIAL_LANGUAGE');return language;}
 const q=String(query);
 const matched=new Set();
 if(/\bpython\b|\bpathlib\b|\basyncio\b|\bsubprocess\b|\bpip\b/i.test(q)||/site:docs\.python\.org\b/i.test(q))matched.add('Python');
 if(/\btypescript\b|\btsc\b|\bmapped\s+types?\b|\bdecorators?\b/i.test(q)||/site:www\.typescriptlang\.org\b/i.test(q))matched.add('TypeScript');
 if(/\brust\b|\brustdoc\b|\bE\d{4}\b|\bstd::/i.test(q)||/site:doc\.rust-lang\.org\b/i.test(q))matched.add('Rust');
 if(/\bgolang\b|\bgo\b|\bgovulncheck\b|\bencoding\/json\b|\bWithCancel\b|\bcontext\.WithCancel\b/i.test(q)||/site:(?:pkg\.go\.dev|go\.dev)\b/i.test(q))matched.add('Go');
 if(matched.size!==1)throw code(matched.size?'AMBIGUOUS_OFFICIAL_LANGUAGE':'OFFICIAL_LANGUAGE_REQUIRED');
 return [...matched][0];
}
function docText(input){
 return input.replace(/<script\b[^>]*>[\s\S]*?<\/script\s*>/gi,' ')
  .replace(/<style\b[^>]*>[\s\S]*?<\/style\s*>/gi,' ')
  .replace(/<(?:svg|noscript|template|head)\b[^>]*>[\s\S]*?<\/(?:svg|noscript|template|head)\s*>/gi,' ')
  .replace(/<[^>]*>/g,' ').replace(/&#(\d+);/g,(_,n)=>String.fromCodePoint(Math.min(0x10ffff,Number(n))))
  .replace(/&(?:nbsp|#160);/gi,' ').replace(/&amp;/gi,'&').replace(/&lt;/gi,'<').replace(/&gt;/gi,'>')
  .replace(/&quot;/gi,'"').replace(/\s+/g,' ').trim();
}
export class OfficialMultisourceProvider {
 constructor({get=officialGet,discover=discoverOfficial,cacheTtlMs=300000}={}){
  if(typeof get!=='function'||typeof discover!=='function')throw code('INVALID_OFFICIAL_BACKEND');
  this.get=get;this.discover=discover;
  this.getIndex=createOfficialIndexCache({get,ttlMs:cacheTtlMs});
 }
 async search(query,{signal,language}={}){
  if(typeof query!=='string'||!query.trim()||query.length>180||/[\x00-\x1f\x7f]/.test(query)||SENSITIVE.test(query))throw code('INVALID_OFFICIAL_QUERY');
  if(signal?.aborted)throw code('CANCELLED');
  const lang=officialLanguage(query,language);
  const hits=await this.discover(lang,query,{get:(url,opt={})=>this.getIndex(url,{...opt,signal})});
  if(signal?.aborted)throw code('CANCELLED');
  if(!Array.isArray(hits))throw code('WEB_PROVIDER_INVALID');
  return hits.filter(x=>x?.source!=='PKGSITE_OFFICIAL_API').slice(0,8).map(x=>{
   const url=officialUrl(x.url).href;
   return {title:String(x.title||'').slice(0,160),url,excerpt:String(x.excerpt||x.source||'').slice(0,1000),official_source:x.source};
  });
 }
 async extract(url,{focus='',signal}={}){
  const u=officialUrl(url);
  if(typeof focus!=='string'||focus.length>140||/[\x00-\x1f\x7f]/.test(focus)||SENSITIVE.test(focus))throw code('INVALID_OFFICIAL_QUERY');
  if(signal?.aborted)throw code('CANCELLED');
  // Always fetch the actual official source body, never silently return cached search index metadata.
  const page=await this.get(u.href,{signal,limit:4*1024*1024});
  if(!Buffer.isBuffer(page?.bytes)||page.bytes.length===0)throw code('WEB_EXTRACTION_FAILED');
  const raw_sha256=createHash('sha256').update(page.bytes).digest('hex');
  const all=docText(page.bytes.toString('utf8'));
  if(!all)throw code('WEB_EXTRACTION_FAILED');
  const index=focus?all.toLowerCase().indexOf(focus.toLowerCase()):-1;
  // Strict bounded excerpt to preserve Voder's 1 MiB request safety limit.
  const start=index<0?0:Math.max(0,index-1200);
  const text=all.slice(start,start+Math.min(26000,1024*1024));
  return {url:u.href,text,raw_sha256};
 }
}
