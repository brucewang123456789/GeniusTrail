/** Optional host-installed adapter example. No networking or disk writes.
 * Replace signing/parsing here for an unusual API. Never load model-supplied code.
 * This example consumes a newline-delimited TSV response: URL<TAB>title<TAB>snippet.
 */
export function buildRequest({request}) { return request; }
export function parseResponse(payload) {
 if(typeof payload!=='string')throw Object.assign(new Error('WEB_PROVIDER_INVALID'),{code:'WEB_PROVIDER_INVALID'});
 return payload.trim()?payload.trim().split('\n').slice(0,32).map(line=>{
  const [url,title='',excerpt='']=line.replace(/\r$/,'').split('\t');return {url,title,excerpt};
 }):[];
}
