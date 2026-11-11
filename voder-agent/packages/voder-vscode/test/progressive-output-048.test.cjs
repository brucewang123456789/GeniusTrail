'use strict';
const test=require('node:test'),assert=require('node:assert/strict');
const {createProgressiveOutput}=require('../src/progressive-output.cjs');
const {createAssistantPresentation}=require('../src/output-presentation.cjs');
class Text{
 constructor(value){this.nodeType=3;this.textContent=String(value)}appendData(text){this.textContent+=text}
}
class Element{
 constructor(tag='div'){this.nodeType=1;this.tagName=tag;this.children=[];this.className='';this.hidden=false;this.attrs={}}
 append(...nodes){this.children.push(...nodes)}replaceChildren(...nodes){this.children=[...nodes]}setAttribute(k,v){this.attrs[k]=v}removeAttribute(k){delete this.attrs[k]}
 get childNodes(){return this.children}get textContent(){return this.children.map(n=>n.textContent).join('')}set textContent(s){this.children=[new Text(s)]}
}
function setup({reduce=false,frames=true}={}){
 let time=0,next=0,updates=0;const queue=new Map(),events={};const doc={hidden:false,createElement:t=>new Element(t),createTextNode:t=>new Text(t),addEventListener:(k,v)=>{events[k]=v}};
 const renderer=createAssistantPresentation(doc);let parses=0;
 const progressive=createProgressiveOutput(doc,{render:(raw,node)=>{parses++;return renderer.render(raw,node)},requestFrame:frames?f=>{queue.set(++next,f);return next}:undefined,cancelFrame:id=>queue.delete(id),now:()=>time,reducedMotion:()=>reduce,onUpdate:()=>updates++});
 return {doc,events,progressive,root:new Element(),get parses(){return parses},get updates(){return updates},get pending(){return queue.size},tick(ms){time+=ms;const tasks=[...queue.values()];queue.clear();for(const f of tasks)f(time)}};
}
const count=(node,tag)=>(node.tagName===tag?1:0)+(node.children||[]).reduce((n,c)=>n+count(c,tag),0);
test('received answer appears over multiple frames; markdown is parsed only once',()=>{
 const x=setup(),body='Hello progressive answer '.repeat(100);x.progressive.render(body,x.root);assert.equal(x.root.textContent,'');assert.equal(x.parses,1);
 x.tick(80);assert.ok(x.root.textContent.length>0&&x.root.textContent.length<body.length);x.tick(4000);assert.equal(x.root.textContent,body);assert.equal(x.pending,0);assert.equal(x.progressive.pending,0);assert.equal(x.root.attrs['aria-busy'],'false');assert.equal(x.parses,1);
});
test('completed text, safe links, lists, code, math fallback and table structure match the existing renderer',()=>{
 const body='## Heading\n\n- Item **one**\n- Item two\n\n| Name | Value |\n| --- | --- |\n| A | 1 |\n| B | 2 |\n\n```rust\nfn main() {}\n```\n\n[Docs](https://docs.rs/reqwest/) $x^2$';
 const x=setup(),expected=new Element();createAssistantPresentation(x.doc).render(body,expected);x.progressive.render(body,x.root);x.tick(100);x.tick(5000);assert.equal(x.root.textContent,expected.textContent);
 for(const tag of ['table','tr','li','pre','a'])assert.equal(count(x.root,tag),count(expected,tag));
});
test('grapheme-safe updates do not split emoji sequences or combining accents',()=>{
 const text='👩‍💻e\u0301你好'.repeat(100),x=setup();x.progressive.render(text,x.root);x.tick(30);
 const segments=Array.from(new Intl.Segmenter(undefined,{granularity:'grapheme'}).segment(text),s=>s.segment);const prefixes=new Set(['']);let p='';for(const s of segments){p+=s;prefixes.add(p)}assert.ok(prefixes.has(x.root.textContent));x.tick(5000);assert.equal(x.root.textContent,text);
});
test('reduced motion and missing animation API show complete content immediately',()=>{
 for(const opts of [{reduce:true},{frames:false}]){const x=setup(opts);x.progressive.render('Accessible answer',x.root);assert.equal(x.root.textContent,'Accessible answer');assert.equal(x.pending,0)}
});
test('hidden tab completes pending content and releases scheduled work',()=>{
 const x=setup();x.progressive.render('Long answer '.repeat(300),x.root);x.tick(50);x.doc.hidden=true;x.events.visibilitychange();assert.equal(x.pending,0);assert.equal(x.progressive.pending,0);assert.equal(x.root.textContent,'Long answer '.repeat(300));
});
test('clearing the conversation cancels only presentation work without callbacks on removed answers',()=>{
 const x=setup();x.progressive.render('First answer '.repeat(100),x.root);x.tick(20);const visible=x.root.textContent;x.progressive.cancelAll();x.tick(5000);assert.equal(x.root.textContent,visible);assert.equal(x.pending,0);assert.equal(x.progressive.pending,0);
});
test('overlapping responses retain separate content and completion ownership',()=>{
 const x=setup(),second=new Element();x.progressive.render('Old answer '.repeat(100),x.root);x.tick(50);x.progressive.render('New answer '.repeat(100),second);x.tick(5000);assert.equal(x.root.textContent,'Old answer '.repeat(100));assert.equal(second.textContent,'New answer '.repeat(100));assert.equal(x.pending,0);
});
