import {test} from 'node:test';
import assert from 'node:assert/strict';
import {approvedBingRssRedirect,KeylessWebProvider} from '../src/keyless-web.mjs';
const source='https://www.bing.com/search?q=Python+pathlib+Path+official+documentation&format=rss';
const target='https://cn.bing.com/search?q=Python+pathlib+Path+official+documentation&format=rss';
test('allows only exactly matched Bing RSS regional redirect',()=>assert.equal(approvedBingRssRedirect(source,target),target));
test('denies unapproved hosts redirects and credentials',()=>{
 for(const s of ['https://evil.example/search?q=Python+pathlib+Path+official+documentation&format=rss','http://cn.bing.com/search?q=Python+pathlib+Path+official+documentation&format=rss','https://cn.bing.com@evil.example/search?q=Python+pathlib+Path+official+documentation&format=rss','https://cn.bing.com:8443/search?q=Python+pathlib+Path+official+documentation&format=rss'])assert.throws(()=>approvedBingRssRedirect(source,s),{code:'WEB_REDIRECT_DENIED'});
});
test('denies modified queries fragments or non-RSS source',()=>{
 for(const s of [target.replace('format=rss','format=html'),target+'#x',target.replace('Python','Other'),target.replace('/search','/other')])assert.throws(()=>approvedBingRssRedirect(source,s),{code:'WEB_REDIRECT_DENIED'});
 assert.throws(()=>approvedBingRssRedirect('https://www.bing.com/',target),{code:'WEB_REDIRECT_DENIED'});
});
test('Bing RSS enables narrow redirect only in request options',async()=>{
 let opts;const p=new KeylessWebProvider({getImpl:async(_url,options)=>{opts=options;return '<rss><channel></channel></rss>';}});
 assert.deepEqual(await p.search('nodejs'),[]);
 assert.equal(opts.bingRssRedirect,true);
});
test('source extraction cannot enable RSS redirect',async()=>{
 let opts;const p=new KeylessWebProvider({getImpl:async(_url,options)=>{opts=options;return '<html><body>valid</body></html>';}});
 await p.extract('https://docs.python.org/3/library/pathlib.html');
 assert.equal(opts.bingRssRedirect,undefined);
});
