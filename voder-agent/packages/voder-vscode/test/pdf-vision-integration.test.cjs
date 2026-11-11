'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const {fromBytes,ReferenceStore}=require('../src/reference-files.cjs');
const {referenceImages}=require('../../voder-agent-core/src/reference-context.cjs');
const {renderPage,selectPages,visualReferences}=require('../src/pdf-vision.cjs');
const fixture=n=>fs.readFileSync(path.join(process.env.VODER_PDF_VISION_FIXTURES||path.resolve(__dirname,'../../../../fixtures'),n));
test('real 2-page PDF: physical page count and exact page attribution',async()=>{
 const pdf=fixture('two-pages.pdf'),item=await fromBytes({name:'two.pdf',bytes:pdf});
 assert.equal(item.kind,'text');assert.match(item.note,/PDF physical pages=2 \(pdfinfo verified\)/);
 assert.match(item.content,/\[PDF PHYSICAL PAGE 1\/2\][^]*First page text/);
 assert.match(item.content,/\[PDF PHYSICAL PAGE 2\/2\][^]*Second page text/);
 assert.equal((item.content.match(/\[PDF PHYSICAL PAGE /g)||[]).length,2);
});
test('ordinary PDF request sends both visual pages and preserves full text',async()=>{
 const pdf=fixture('two-pages.pdf'),item=await fromBytes({name:'two.pdf',bytes:pdf});
 const store=new ReferenceStore();store.add([item]);store.pdfSources.set(item.id,pdf);
 const before=Buffer.from(item.content);const result=await store.prepare('/tmp','Summarize all paragraphs');
 assert.equal(result.length,3);assert.equal(referenceImages(result).length,2);assert.deepEqual(Buffer.from(result[0].content),before);
 store.dispose();
});
test('visual request renders the correct real page and transmits verified image bytes',async()=>{
 const pdf=fixture('two-pages.pdf'),item=await fromBytes({name:'two.pdf',bytes:pdf});
 const store=new ReferenceStore();store.add([item]);store.pdfSources.set(item.id,pdf);
 const result=await store.prepare('/tmp','Analyze the chart on page 2');
 assert.equal(result.length,3);assert.equal(result[2].kind,'image');assert.match(result[2].name,/page 2\/2/);
 const images=referenceImages(result);assert.equal(images.length,2);assert.equal(images[0].mimeType,'image/png');
 const exact=await renderPage(pdf,2,{count:2});assert.deepEqual(Buffer.from(images[1].data,'base64'),exact);
 store.dispose();
});
test('selected whole small PDF sends all visual pages, not just cover',async()=>{
 const result=await visualReferences(fixture('two-pages.pdf'),{name:'two.pdf',source:'attachment',goal:'Read every diagram and chart'});
 assert.equal(result.length,2);assert.deepEqual(result.map(x=>x.name),['two.pdf [PDF page 1/2]','two.pdf [PDF page 2/2]']);
});
test('scanned-only PDF triggers real OCR (no imaginary extractable text)',async()=>{
 const scanned=await fromBytes({name:'scan.pdf',bytes:fixture('scanned-ocr.pdf')});
 assert.match(scanned.content,/SCAN OCR VODER 418/);assert.match(scanned.note,/OCR count=1/);assert.match(scanned.content,/verify transcription against image/);
});
test('no silent visual truncation; out-of-range pages never invented',async()=>{
 assert.throws(()=>selectPages('Review all diagrams',45),{code:'REFERENCE_VISUAL_SCOPE'});
 assert.deepEqual(selectPages('Look at figure on page 45',45),[45]);
 assert.deepEqual(selectPages('Check figure on 第2页 plus 最后一页',45),[2,45]);
 assert.deepEqual(selectPages('看第2-4页的图表',45),[2,3,4]);
 assert.deepEqual(selectPages('查看最后3页的图形',45),[43,44,45]);
});
test('removal and post-accept clear erase PDF bytes; failed batch does not append',async()=>{
 const store=new ReferenceStore(),item=await fromBytes({name:'two.pdf',bytes:fixture('two-pages.pdf')});
 store.add([item]);store.pdfSources.set(item.id,fixture('two-pages.pdf'));
 store.remove(item.id);assert.equal(store.pdfSources.size,0);
 store.add([item]);store.pdfSources.set(item.id,fixture('two-pages.pdf'));store.clear();assert.equal(store.pdfSources.size,0);store.dispose();
});
