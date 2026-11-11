/** Experimental, offline-testable FIRST-PARTY source adapters. NOT deployable production code. */
import {inflateSync} from 'node:zlib';

const MAX_INV_COMPRESSED = 512*1024;
const MAX_INV_EXPANDED = 8*1024*1024;
const MAX_ENTRIES = 60000;
const safeUrl=(raw,host)=>{
 let u;try{u=new URL(raw);}catch{return null;}
 if(u.protocol!=='https:'||u.hostname!==host||u.port||u.username||u.password)return null;
 return u.href;
};
const tokens=s=>[...new Set((s.toLowerCase().match(/[a-z][a-z\d._-]*/g)||[]).filter(x=>!new Set(['python','rust','go','typescript','official','documentation','docs','api','guide','site','html','the','how','reference']).has(x)))].slice(0,10);
/** Parses Sphinx v2 inventories as published by the Python documentation team. */
export function parseSphinxInventory(data,{base='https://docs.python.org/3/'}={}){
 const buf=Buffer.from(data);
 if(buf.length>MAX_INV_COMPRESSED)throw Error('INVENTORY_TOO_LARGE');
 const sep=[];for(let i=0;i<buf.length&&sep.length<4;i++)if(buf[i]===10)sep.push(i+1);
 if(sep.length<4||buf.subarray(0,sep[0]).toString('utf8').trim()!=='# Sphinx inventory version 2'||
    !buf.subarray(sep[2],sep[3]).toString('utf8').includes('zlib'))throw Error('INVALID_INVENTORY_HEADER');
 let out;try{out=inflateSync(buf.subarray(sep[3]),{maxOutputLength:MAX_INV_EXPANDED}).toString('utf8');}
 catch{throw Error('INVALID_INVENTORY_BODY');}
 const root=new URL(base);if(root.protocol!=='https:'||root.hostname!=='docs.python.org')throw Error('UNTRUSTED_INDEX_ORIGIN');
 const entries=[];
 for(const line of out.split('\n')){
  if(entries.length>=MAX_ENTRIES)break;
  const m=/^(\S+)\s+(\w+:\w+)\s+(-?\d+)\s+(\S+)\s+(.+)$/.exec(line);
  if(!m)continue;
  const [,name,kind,priority,uriRaw,caption]=m;
  if(Number(priority)<0||!/^py:|^std:/.test(kind)||uriRaw.includes('..'))continue;
  const relative=uriRaw.endsWith('$')?uriRaw.slice(0,-1)+name:uriRaw;
  const url=safeUrl(new URL(relative,root).href,'docs.python.org');
  if(!url)continue;
  entries.push({name,kind,url,title:caption==='-'?name:caption});
 }
 return entries;
}

export function findSphinxEntries(entries,query,{maxResults=8}={}){
 const words=tokens(query);if(!words.length)return [];
 // A Python module explicitly named in a task is more informative than an
 // incidental token in a different module (subprocess != asyncio.subprocess).
 // Infer it from the *actual inventory*; no hard-coded symbol or result URL.
 const primary=words[0];
 const hasRoot=entries.some(e=>e.name.toLowerCase()===primary||e.name.toLowerCase().startsWith(primary+'.'));
 return entries.map(e=>{
  const n=e.name.toLowerCase(),t=e.title.toLowerCase(),p=new URL(e.url).pathname.toLowerCase();
  let score=0;
  for(const w of words){if(n===w||n.endsWith('.'+w))score+=25;else if(n.includes(w))score+=12;if(t.includes(w))score+=4;if(p.includes(w))score+=3;}
  if(hasRoot && (n===primary||n.startsWith(primary+'.'))){
   score+=65;
   if(p.includes('/library/'+primary+'.html')||p.includes('/library/'+primary+'-'))score+=25;
  }
  return {...e,score};
 }).filter(x=>x.score>0).sort((a,b)=>b.score-a.score||a.url.localeCompare(b.url)).slice(0,maxResults);
}

/** Consume a documented pkg.go.dev /v1/search JSON response. No guessed URLs. */
export function parseGoPkgsiteSearch(json,{maxResults=12}={}){
 const obj=typeof json==='string'?JSON.parse(json):json;
 if(!obj||!Array.isArray(obj.items)||obj.items.length>200)throw Error('INVALID_GO_API_RESPONSE');
 const out=[];
 for(const row of obj.items){
  if(typeof row?.packagePath!=='string'||!/^([\da-z][a-z\d._-]*\/)*[a-z\d][\w.-]*$/i.test(row.packagePath)||row.packagePath.length>180)continue;
  const url=safeUrl(`https://pkg.go.dev/${row.packagePath}`,'pkg.go.dev');
  if(!url)continue;
  out.push({url,title:row.packagePath,excerpt:String(row.synopsis??'').slice(0,1000),version:String(row.version??'').slice(0,50),source:'PKGSITE_OFFICIAL_API'});
  if(out.length>=maxResults)break;
 }
 return out;
}
/** Safe construction from a trusted first-party endpoint, never from model URL. */
export function goSearchApiUrl(query,{symbol,maxResults=15}={}){
 if(typeof query!=='string'||!query.trim()||query.length>120||/[\x00-\x1f\x7f]/.test(query))throw Error('INVALID_QUERY');
 if(symbol!==undefined&&(typeof symbol!=='string'||!/^[A-Za-z]\w{0,80}$/.test(symbol)))throw Error('INVALID_SYMBOL');
 if(!Number.isInteger(maxResults)||maxResults<1||maxResults>30)throw Error('INVALID_RESULTS_LIMIT');
 const u=new URL('https://pkg.go.dev/v1/search');u.searchParams.set('q',query);
 if(symbol)u.searchParams.set('symbol',symbol);u.searchParams.set('limit',String(maxResults));
 return u.href;
}
