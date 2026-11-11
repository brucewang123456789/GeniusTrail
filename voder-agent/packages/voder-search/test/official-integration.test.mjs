import {test} from 'node:test';
import assert from 'node:assert/strict';
import {OfficialMultisourceProvider,officialLanguage} from '../src/official-provider.mjs';
import {EvidenceSearchService} from '../src/index.mjs';
const mockUrl='https://docs.python.org/3/library/pathlib.html';
const payload=Buffer.from('<html><head><script>INJECTED_SCRIPT</script></head><body><h1>Pathlib</h1><p>The Path object</p></body></html>');
test('default router retains v3.10 official backend and network disabled by default',async()=>{
 const x=new EvidenceSearchService({cwd:process.cwd()});
 assert.ok(x.provider.official instanceof OfficialMultisourceProvider);assert.equal(x.allowWeb,false);
 await assert.rejects(x.query(x.newTask('search the web'),{mode:'web',query:'Python pathlib'}),{code:'WEB_DISABLED'});
});
test('explicit language wins; ambiguous and missing language do not cause untrusted search egress',()=>{
 assert.equal(officialLanguage('Path','Python'),'Python');
 assert.equal(officialLanguage('Python pathlib'),'Python');
 assert.throws(()=>officialLanguage('generic array type'),{code:'OFFICIAL_LANGUAGE_REQUIRED'});
 assert.throws(()=>officialLanguage('Python and Rust'),{code:'AMBIGUOUS_OFFICIAL_LANGUAGE'});
});
test('discovered official index, task URL allowlist and true Fetch preserve evidence authorization',async()=>{
 let discoveries=0,fetches=0;
 const provider=new OfficialMultisourceProvider({discover:async(lang,q)=>{discoveries++;assert.equal(lang,'Python');return [{url:mockUrl,title:'Path',excerpt:'Official Python',score:99,source:'PYTHON_SPHINX_OFFICIAL'}];},get:async(url)=>{fetches++;assert.equal(url,mockUrl);return {bytes:payload,url}}});
 const x=new EvidenceSearchService({cwd:process.cwd(),allowWeb:true,provider});const t=x.newTask('search the web for Python docs');
 await assert.rejects(x.query(t,{mode:'fetch',query:'http://127.0.0.1/'}),{code:'UNSAFE_SOURCE_URL'});
 const found=await x.query(t,{mode:'web',query:'Python pathlib Path',language:'Python'});
 assert.equal(found.status,'EVIDENCE_FOUND');assert.equal(found.results[0].url,mockUrl);assert.equal(discoveries,1);
 const fetched=await x.query(t,{mode:'fetch',query:mockUrl,focus:'Path'});
 assert.equal(fetched.status,'EVIDENCE_FOUND');assert.ok(fetched.results[0].excerpt.includes('Pathlib'));
 assert.ok(!fetched.results[0].excerpt.includes('INJECTED_SCRIPT'));
 assert.match(fetched.results[0].raw_source_sha256,/^[a-f0-9]{64}$/);assert.equal(fetches,1);
 assert.equal(fetched.authority,'CONTEXT_ONLY');
});
test('provider blocks private URL and aborts before external fetch',async()=>{
 const provider=new OfficialMultisourceProvider({get:async()=>{throw Error('UNEXPECTED_NETWORK')}});
 await assert.rejects(provider.extract('http://127.0.0.1/private'),{code:'INVALID_OFFICIAL_URL'});
 const c=new AbortController();c.abort();await assert.rejects(provider.search('Python asyncio',{signal:c.signal}),{code:'CANCELLED'});
});
