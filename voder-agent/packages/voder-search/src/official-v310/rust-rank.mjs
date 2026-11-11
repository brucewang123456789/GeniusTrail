/** Rustdoc symbol-aware reranking. Operates ONLY on links observed in Rust's
 * first-party index; does not construct URLs, fetch unlisted pages, or promote
 * aliases returned by an untrusted search engine.
 *
 * The main failure in v3.9 was a substring tie: VecDeque beat Vec, OsString
 * beat String, nonpoison::Mutex beat sync::Mutex and aliases beat AtomicBool.
 */
const RUST_ITEM = /^\/std\/((?:[a-z0-9_]+\/)*)((?:struct|enum|type|trait|union|primitive|constant|fn|macro))\.([a-z_][a-z0-9_]*)\.html$/i;
const NONIDENTIFIER = /[^a-z0-9_]/i;
const IGNORE = new Set(['rust','std','api','docs','doc','documentation','reference','official','the','a','an','and','or','how','use','using','for','in','of','with','to','type','class','struct','enum','trait','function','method','symbol','language','programming','code','library']);
export function rustSymbolRerank(query,item){
  if (item?.source!=='OFFICIAL_HTML_INDEX') return 0;
  let url;
  try { url=new URL(item.url); } catch { return 0; }
  if(url.hostname!=='doc.rust-lang.org'||url.protocol!=='https:'||url.search||url.hash) return 0;
  const match=RUST_ITEM.exec(url.pathname);
  if(!match) return 0;
  const moduleParts=match[1].split('/').filter(Boolean).map(x=>x.toLowerCase());
  const itemSymbol=match[3].toLowerCase();
  // Whole identifiers: substring similarity alone must NEVER get an exact hit.
  const identifiers=(String(query).match(/[A-Za-z_][A-Za-z0-9_]*/g)??[])
    .filter(x=>!NONIDENTIFIER.test(x)).map(x=>x.toLowerCase()).filter(x=>!IGNORE.has(x));
  if(!identifiers.length) return 0;
  const terms=new Set(identifiers);
  if(!terms.has(itemSymbol)) return 0;
  const moduleHits=moduleParts.filter(x=>terms.has(x)).length;
  const missingModules=moduleParts.length-moduleHits;
  // Exact symbol is the most discriminative feature. Module path agreement
  // breaks ties between reexports (std::sync vs std::sync::nonpoison).
  // A stable shallow-path preference makes canonical paths win ties.
  return 150+moduleHits*32-missingModules*12-moduleParts.length*2;
}
