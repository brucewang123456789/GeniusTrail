'use strict';
const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const {ReferenceController}=require('../src/reference-ui.cjs');
const {selectPages,indexedFigures}=require('../src/pdf-vision.cjs');
const pdf=process.env.VODER_VISION_E2E_PDF||path.join(__dirname,'fixtures','figure-index.pdf');
assert.ok(pdf&&fs.existsSync(pdf),'Requires actual readable PDF');
const name=path.basename(pdf);
const api={window:{},workspace:{textDocuments:[]}};
const folder={uri:{fsPath:path.dirname(pdf)}};
const uri={scheme:'file',fsPath:pdf};
test('actual Poppler text maps indexed Figure 7 to real physical page 3',async()=>{
 const ctl=new ReferenceController({vscode:api});const view={};
 await ctl.files(view,[uri]);
 const r=await ctl.prepare(view,folder.uri.fsPath,'分析PDF中Figure 7架构图');
 const imgs=r.filter(x=>x.kind==='image');
 assert.equal(imgs.length,4);assert.match(imgs[2].name,/\[PDF page 3\/4\]/);
 assert.equal(Buffer.from(imgs[2].data,'base64').subarray(0,8).toString('hex'),'89504e470d0a1a0a');
 assert.equal(r[0].kind,'text'); assert.match(r[0].content,/\[PDF PHYSICAL PAGE 4\/4\]/);
 ctl.dispose(view);
});
test('real VS Code chat reference uses actual PDF and renders indexed figure',async()=>{
 const ctl=new ReferenceController({vscode:api});
 const r=await ctl.requestContext(folder,'请解释Figure 7架构图',[{value:{uri}}]);
 assert.equal(r.filter(x=>x.kind==='image').length,4);
 assert.match(r.filter(x=>x.kind==='image')[2].name,/\[PDF page 3\/4\]/);
});
test('complete visual bundle contains the requested physical page exactly',async()=>{
 const ctl=new ReferenceController({vscode:api});const view={};
 await ctl.files(view,[uri]);
 const r=await ctl.prepare(view,folder.uri.fsPath,'解释第2页图表');
 assert.match(r.filter(x=>x.kind==='image')[1].name,/\[PDF page 2\/4\]/);
 ctl.dispose(view);
});
test('45-page all-visual request is never silently treated as an eight-page scan',()=>{
 assert.throws(()=>selectPages('分析全部图表',45,'[PDF PHYSICAL PAGE 1/45]\nfoo'),e=>e.code==='REFERENCE_VISUAL_SCOPE');
});
test('unresolvable caption is rejected without guessing',()=>{
 assert.throws(()=>selectPages('请看Figure 99图表',45,'[PDF PHYSICAL PAGE 1/45]\nFigure 7. A'),e=>e.code==='REFERENCE_VISUAL_SCOPE');
});
test('figure index never treats a body reference as a caption',()=>{
 const text='[PDF PHYSICAL PAGE 1/3]\nText discusses Figure 7 in body.\n[PDF PHYSICAL PAGE 2/3]\nFigure 7. A design\n[PDF PHYSICAL PAGE 3/3]\nend';
 assert.deepEqual(indexedFigures(text,3).get(7),[2]);
});
