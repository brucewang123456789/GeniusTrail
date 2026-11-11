/** Route general facts to public search; preserve specialized official documentation. */
import {OfficialMultisourceProvider,officialLanguage} from './official-provider.mjs';
import {RustWebProvider} from './rust-web.mjs';
import {siteScope} from './research.mjs';
import {officialUrl} from './official-v310/official-http.mjs';
export class RoutedPublicProvider {
 constructor({official=new OfficialMultisourceProvider(),web=new RustWebProvider()}={}){this.official=official;this.web=web;}
 cacheIdentity(){return this.web.cacheIdentity?.()??'';}
 async search(query,options={}){
  const scopes=siteScope(query);
  // A requested site takes precedence over accidental language keywords.
  const docs=['docs.python.org','www.typescriptlang.org','typescriptlang.org','doc.rust-lang.org','pkg.go.dev','go.dev'];
  if(scopes.length&&scopes.some(s=>!docs.includes(s.host)))return this.web.search(query,options);
  try{officialLanguage(query,options.language);}catch(e){
   if(!['OFFICIAL_LANGUAGE_REQUIRED','AMBIGUOUS_OFFICIAL_LANGUAGE'].includes(e?.code))throw e;
   return this.web.search(query,options);
  }
  return this.official.search(query,options);
 }
 async extract(url,options={}){
  let official=false;try{officialUrl(url);official=true;}catch{}
  return (official?this.official:this.web).extract(url,options);
 }
}
