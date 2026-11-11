/**
 * Conservative evidence candidate planning for Pi's existing read-only search tool.
 * This is NOT an independent agent, a verifier, or a claim that a search hit is true.
 * No networking, host permissions, or side effects are implemented in this module.
 */
const CATALOG = Object.freeze([
  {id:'python', pattern:/(?:\bpython\b|\bpathlib\b|\basyncio\b|\bsubprocess\b)/i, domains:['docs.python.org']},
  {id:'typescript',pattern:/(?:\btypescript\b|\btsconfig\b|\bdecorators?\b)/i,domains:['typescriptlang.org']},
  {id:'rust',pattern:/(?:\brust\b|\bcargo\b|\bstd::\b|\bE0\d{3}\b|\bArc\b)/i,domains:['doc.rust-lang.org']},
  {id:'go',pattern:/(?:\bgolang\b|\bgo\s+(?:lang|context|module|security)\b|\bgovulncheck\b|\bWithCancel\b)/i,domains:['pkg.go.dev','go.dev']}
]);
const STOP = new Set(['site','html','https','http','www','docs','doc','documentation','official','api','reference','guide','example','examples','for','the','and','how','with','using','in','on','of','to','a','is','an','class','language','programming','python','typescript','rust','golang','path','std','library','com','org','dev','cn']);
const safeTokens = value => [...new Set((value.toLowerCase().match(/[\p{L}\p{N}_]+/gu)??[]).filter(t=>t.length>=3&&!STOP.has(t)))].slice(0,14);
const allTokens = value => [...new Set((value.toLowerCase().match(/[\p{L}\p{N}_]+/gu)??[]).filter(t=>t.length>=2))];
export function siteScope(query){
 const scopes=[...String(query).matchAll(/(?:^|\s)site:([^\s]+)/gi)].map(m=>{
  const raw=m[1],host=raw.split('/')[0].toLowerCase();
  if(!/^[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?$/.test(host)||!host.includes('.')||host.includes('..')||host.split('.').some(x=>x.startsWith('-')||x.endsWith('-')))
   throw Object.assign(new Error('INVALID_SEARCH_QUERY'),{code:'INVALID_SEARCH_QUERY'});
  return {host,path:raw.includes('/')?'/'+raw.split('/').slice(1).join('/'):''};
 });
 return scopes;
}
export function matchesSiteScope(url,scopes){
 if(!scopes.length)return true;
 let u;try{u=new URL(url);}catch{return false;}
 return scopes.some(s=>(u.hostname===s.host||u.hostname.endsWith('.'+s.host))&&(!s.path||u.pathname===s.path||u.pathname.startsWith(s.path.endsWith('/')?s.path:s.path+'/')));
}
function catalogFor(query){
 // Only a known ecosystem selects an official-domain hint. A user's site: term
 // does not authorize an arbitrary domain and cannot change host allow/deny rules.
 for(const record of CATALOG)if(record.pattern.test(query))return record;
 const site=(query.match(/\bsite:([a-z0-9.-]+)(?:\/\S*)?/i)??[])[1]?.toLowerCase();
 return CATALOG.find(record=>record.domains.some(d=>site===d||site===`www.${d}`))??null;
}
const hostIs=(host,domain)=>host===domain||host===`www.${domain}`;
const normalizedBase=query=>query.replace(/\bsite:\S+/gi,' ').replace(/\s+/g,' ').trim();
export function planResearchQueries(query,{focus='',maxQueries=2}={}){
 if(!Number.isSafeInteger(maxQueries)||maxQueries<1||maxQueries>3)throw new TypeError('INVALID_RESEARCH_BUDGET');
 const scopes=siteScope(query);
 const ecosystem=catalogFor(query+' '+focus),domain=scopes[0]?.host??ecosystem?.domains[0];
 const base=normalizedBase(query);
 const queryTerms=[...new Set([...safeTokens(base),...safeTokens(focus)])].slice(0,9).join(' ');
 const variants=[query];
 if(domain&&queryTerms){
  const scoped=`site:${domain}${scopes[0]?.path??''} ${queryTerms}`;
  if(scoped.length<=180&&!variants.includes(scoped))variants.push(scoped);
 }
 if(queryTerms){
  const bare=scopes.length?`site:${scopes[0].host}${scopes[0].path} ${queryTerms}`:`${queryTerms} ${ecosystem?'official documentation':'official source'}`;
  if(bare.length<=180&&!variants.includes(bare))variants.push(bare);
 }
 return Object.freeze({ecosystem:ecosystem?.id??null,official_domains:ecosystem?.domains??[],queries:Object.freeze(variants.slice(0,maxQueries))});
}
export function rankResearchCandidates(query,rows,{focus='',maxResults=5}={}){
 if(!Number.isSafeInteger(maxResults)||maxResults<1||maxResults>8)throw new TypeError('INVALID_RESEARCH_BUDGET');
 const scopes=siteScope(query);
 const ecosystem=catalogFor(query+' '+focus);
 const queryTokens=safeTokens(normalizedBase(query)+' '+focus);
 const ranked=[];const seen=new Set();
 for(const row of rows){
  let u;try{u=new URL(row.url);}catch{continue;}
  if(!matchesSiteScope(u.href,scopes))continue;
  if(u.protocol!=='https:'||u.username||u.password||u.port||seen.has(u.href))continue;
  seen.add(u.href);
  const official=!!ecosystem?.domains.some(d=>hostIs(u.hostname.toLowerCase(),d));
  const path=decodeURIComponentSafe(u.pathname).toLowerCase();
  const title=String(row.title??'').toLowerCase();
  const excerpt=String(row.excerpt??'').toLowerCase();
  const isHomepage = path==='/'||/^\/(?:downloads?|index(?:\.html)?)\/?$/.test(path);
  const matching=queryTokens.filter(token=>path.includes(token)||title.includes(token)||excerpt.includes(token));
  // At least one discriminative task term, and a substantive path (not homepage).
  const lexical=queryTokens.filter(t=>/\p{L}/u.test(t));
  if(isHomepage||(!matching.length&&queryTokens.length)||(lexical.length&&!matching.some(t=>/\p{L}/u.test(t))))continue;
  const distinctMatches=matching.length;
  const score=(official?30:0)+distinctMatches*7+
    matching.filter(t=>path.includes(t)).length*6+
    matching.filter(t=>title.includes(t)).length*3-
    (path.split('/').length<3?4:0);
  // Unknown ecosystems can provide leads, but are never mislabelled official.
  // This gate also rejects generic brand/domain homepages without evidence terms.
  if(score<=0)continue;
  ranked.push({...row,requested_domain_match:scopes.length>0,official_domain_match:official,relevance_score:score,
    matched_terms:Object.freeze(matching.slice(0,8)),candidate_status:'UNVERIFIED_WEB_CANDIDATE'});
 }
 ranked.sort((a,b)=>b.relevance_score-a.relevance_score||a.url.localeCompare(b.url));
 return ranked.slice(0,maxResults);
}
function decodeURIComponentSafe(s){try{return decodeURIComponent(s);}catch{return s;}}
