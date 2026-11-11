'use strict';
const test=require('node:test');
const assert=require('node:assert/strict');
const {createAssistantPresentation}=require('../src/output-presentation.cjs');
class TextNode {constructor(text){this.nodeType=3;this.textContent=String(text);}}
class Element {
 constructor(tag='div'){this.nodeType=1;this.tagName=tag;this.children=[];this.className='';this.href='';}
 append(...nodes){this.children.push(...nodes);}
 replaceChildren(...nodes){this.children=[...nodes];}
 get childNodes(){return this.children;}
 get textContent(){return this.children.map(n=>n.textContent).join('');}
 set textContent(value){this.children=[new TextNode(value)];}
}
const doc={createElement:tag=>new Element(tag),createTextNode:value=>new TextNode(value)};
const presenter=createAssistantPresentation(doc);
function render(raw){const root=new Element();presenter.render(raw,root);return root;}
function tagCount(node,tag){return (node.tagName===tag?1:0)+(node.children||[]).reduce((n,c)=>n+tagCount(c,tag),0);}

test('strict JSON envelope unwrapped without presenting metadata or changing scope semantics',()=>{
 const body='## Analysis\n- File read\n- Read-only';
 const output=render(JSON.stringify({answer:body,scope:null}));
 assert.equal(tagCount(output,'h3'),1);
 assert.equal(tagCount(output,'li'),2);
 assert.equal(output.textContent.includes('"answer"'),false);
 assert.equal(output.textContent.includes('"scope"'),false);
 assert.equal(output.textContent.includes(body.slice(0,2)),false);
 assert.equal(output.textContent.includes('Analysis'),true);
});
test('handles double envelope and exact fenced json without inventing result',()=>{
 const nested=JSON.stringify({answer:JSON.stringify({answer:'The answer is **42**',scope:null}),scope:null});
 assert.equal(render(nested).textContent,'The answer is 42');
 assert.equal(tagCount(render('```json\n'+nested+'\n```'),'strong'),1);
});
test('rejects noncontract JSON and malformed JSON without suppressing content',()=>{
 for(const raw of ['{"answer":bad}',JSON.stringify({answer:'Hello',scope:null,unexpected:'do not erase'}),'{"arbitrary":true}']){
  const result=render(raw);assert.equal(result.textContent,raw);
 }
});
test('does not hide ordinary JSON used as code or evidence',()=>{
 const raw='```json\n{"id":1,"name":"sample"}\n```';
 assert.equal(render(raw).textContent.includes('"id":1'),true);
 assert.equal(tagCount(render(raw),'pre'),1);
});
test('markdown lists, headings, links, code, blockquotes, tables are DOM nodes',()=>{
 const src='# Title\n\n1. First\n2. Second\n\n> Caution\n\n| A | B |\n| --- | --- |\n| 1 | 2 |\n\n```python\nprint(2)\n```\n\n[Safe link](https://example.org)';
 const node=render(src);
 assert.equal(tagCount(node,'h3'),1);
 assert.equal(tagCount(node,'ol'),1);
 assert.equal(tagCount(node,'blockquote'),1);
 assert.equal(tagCount(node,'table'),1);
 assert.equal(tagCount(node,'pre'),1);
 assert.equal(tagCount(node,'a'),1);
});
test('HTML/script input has no executable markup and is preserved as evidence text',()=>{
 const node=render('<img src=x onerror=alert(1)>\n<script>alert("x")</script>');
 assert.equal(tagCount(node,'img'),0);assert.equal(tagCount(node,'script'),0);
 assert.ok(node.textContent.includes('<script>'));
});
test('high-confidence credentials and home paths are masked only in display',()=>{
 const value='Bearer abcdefghijklmnopqrstuvwxyz123456 and sk-proj-abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQ in /root/autodl-tmp/tests/code.py';
 const safe=presenter.redact(value);
 assert.ok(!safe.includes('sk-proj-'));assert.ok(!safe.includes('Bearer abcde'));assert.ok(!safe.includes('/root/'));
 assert.ok(safe.includes('tests/code.py'));
 assert.ok(safe.includes('[credential redacted]'));
 assert.equal(value.includes('sk-proj-'),true);
});
test('message is length-bounded and valid failures preserve original wording',()=>{
 const long='x'.repeat(200000);assert.equal(render(long).textContent.length,131072);
 assert.equal(render('Task ended, unverified').textContent,'Task ended, unverified');
});
test('renderer is actually linked into Webview and does not mutate native core',()=>{
 const {chatHtml}=require('../src/chat-view.cjs');
 const html=chatHtml({cspSource:'vscode-resource:'});
 assert.match(html,/progressiveOutput\.render\(m\.answer,answer\)/);
 assert.match(html,/progressiveOutput\.render\(m\.draft,d\)/);
 assert.match(html,/default-src 'none'/);
 assert.doesNotMatch(html,/\.innerHTML\s*=/);
});
