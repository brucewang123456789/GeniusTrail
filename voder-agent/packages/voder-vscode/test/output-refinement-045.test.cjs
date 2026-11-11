'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const {createAssistantPresentation} = require('../src/output-presentation.cjs');
const {inspectionGoal, inspectReport} = require('../src/entry-orchestrator.cjs');
class TextNode { constructor(value) { this.nodeType=3; this.textContent=String(value); } }
class Element {
  constructor(tag='div') { this.nodeType=1; this.tagName=tag; this.children=[]; this.value=null; this.start=1; }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children=[...nodes]; }
  get childNodes() { return this.children; }
  set textContent(value) { this.children=[new TextNode(value)]; }
  get textContent() { return this.children.map(x=>x.textContent).join(''); }
}
const doc={createElement:tag=>new Element(tag),createTextNode:content=>new TextNode(content)};
const render=raw=>{const root=new Element();createAssistantPresentation(doc).render(raw,root);return root;};
const childrenOf=(parent,tag)=>parent.children.filter(x=>x.tagName===tag);

test('ordered lists with blank lines and mixed marker styles retain one ordered list',()=>{
  const root=render('1. First\n\n2) Second\n\n3. Third');
  const lists=childrenOf(root,'ol');
  assert.equal(lists.length,1);
  assert.equal(lists[0].start,1);
  assert.deepEqual(lists[0].children.map(x=>x.value),[1,2,3]);
  assert.deepEqual(lists[0].children.map(x=>x.textContent),['First','Second','Third']);
});
test('ordered list starting above one preserves its original first ordinal',()=>{
  const root=render('4. Fourth\n\n5. Fifth');
  const lists=childrenOf(root,'ol');
  assert.equal(lists.length,1);
  assert.equal(lists[0].start,4);
  assert.deepEqual(lists[0].children.map(x=>x.value),[4,5]);
});
test('deliberate ordered list restart is not silently renumbered',()=>{
  const root=render('1. First\n\n1. Another first');
  const lists=childrenOf(root,'ol');
  assert.equal(lists.length,2);
  assert.deepEqual(lists.map(x=>x.children[0].value),[1,1]);
});
test('non-list content cannot be pulled into an adjacent list',()=>{
  const root=render('1. One\n\nThe actual evidence is limited.\n\n2. Two');
  assert.equal(childrenOf(root,'ol').length,2);
  assert.equal(childrenOf(root,'p')[0].textContent,'The actual evidence is limited.');
});
test('blockquote, evidence code, JSON unwrapping and redaction stay intact',()=>{
  const snippet='## Overview\n\n> Evidence only\n\n```python\ndef _text(value):\n    return str(value)\n```';
  const root=render(JSON.stringify({answer:snippet,scope:null}));
  assert.equal(childrenOf(root,'h3').length,1);
  assert.equal(childrenOf(root,'blockquote').length,1);
  assert.equal(childrenOf(root,'div').length,1);
  assert.ok(!root.textContent.includes('"scope"'));
  assert.ok(root.textContent.includes('return str(value)'));
  assert.ok(!createAssistantPresentation(doc).redact('Bearer abcdefghijklmnopqrstuvwxyz').includes('abcdefghi'));
});
test('inspection response instructions improve writing but keep read-only and JSON contract',()=>{
  const original='请分析选中函数，不运行代码';
  const prompt=inspectionGoal(original,{hasReferences:true});
  assert.ok(prompt.startsWith(original));
  assert.match(prompt,/reference snapshots as evidence/);
  assert.match(prompt,/Do not modify any file, run scripts\/tests/);
  assert.match(prompt,/Return JSON only:/);
  assert.match(prompt,/"answer":/);
  assert.match(prompt,/"scope":null/);
  assert.match(prompt,/descriptive headings/);
  assert.match(prompt,/Do not begin with decorative bracketed labels/);
  assert.match(prompt,/absence of side effects/);
  assert.match(prompt,/distinguish observations from inferences/);
});
test('inspectionReport continues to preserve original answer and scope decision',()=>{
  const original='【Reference evidence】\n1. The file has an _text function.';
  const draft=JSON.stringify({answer:original,scope:null});
  const report=inspectReport('/tmp/unrelated',{status:'DRAFT_UNVERIFIED',verified:false,draft});
  assert.equal(report.answer,original);
  assert.equal(report.scope,null);
  assert.equal(report.status,'INSPECTED');
});
