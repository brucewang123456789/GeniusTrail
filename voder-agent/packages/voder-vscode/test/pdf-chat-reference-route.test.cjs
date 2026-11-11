'use strict';
// Exercises the real ReferenceController / ReferenceStore / Poppler render path.
// Only the VS Code URI/document facade is supplied; no parser, PDF, model or VGR mocks.
const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const os=require('node:os');
const path=require('node:path');
const {createHash}=require('node:crypto');
const {ReferenceController}=require('../src/reference-ui.cjs');

const fixture=process.env.VODER_PDF_REF_ROUTE_FIXTURE||path.join(__dirname,'fixtures','documents','two-pages.pdf');
assert.ok(fixture&&fs.existsSync(fixture),'Real PDF fixture is required');
const tmp=fs.mkdtempSync(path.join(os.tmpdir(),'voder-pdf-chat-ref-'));
const pdf=path.join(tmp,'two-pages.pdf');
const pdf2=path.join(tmp,'copy.pdf');
const txt=path.join(tmp,'notes.txt');
fs.copyFileSync(fixture,pdf);
fs.copyFileSync(fixture,pdf2);
fs.writeFileSync(txt,'notes\nhello\n','utf8');
test.after(()=>fs.rmSync(tmp,{recursive:true,force:true}));
const uri=(name,scheme='file')=>({scheme,fsPath:name});
const ref=(file)=>({value:{uri:uri(file)}});
const folder={uri:uri(tmp)};
const api={workspace:{textDocuments:[],openTextDocument:async ()=>({getText:()=> 'selected',lineCount:1})},window:{}};
const controller=()=>new ReferenceController({vscode:api});
const signature=createHash('sha256').update(fs.readFileSync(pdf)).digest('hex');

test('PDF through real chat requestContext: no undefined pdfPairs, physical pages and digest',async()=>{
 const result=await controller().requestContext(folder,'Please read this PDF',[ref(pdf)]);
 assert.equal(result.length,3);
 assert.equal(result[0].kind,'text');
 assert.match(result[0].note,/PDF physical pages=2/);
 assert.match(result[0].note,new RegExp(signature));
 assert.match(result[0].content,/\[PDF PHYSICAL PAGE 1\/2\]/);
 assert.match(result[0].content,/\[PDF PHYSICAL PAGE 2\/2\]/);
});

test('PDF visual request in chat retains exact source and renders all real page images',async()=>{
 const result=await controller().requestContext(folder,'请看PDF第一页的图表',[ref(pdf)]);
 assert.equal(result.filter(x=>x.kind==='text').length,1);
 const images=result.filter(x=>x.kind==='image');
 assert.equal(images.length,2);
 assert.equal(images[0].mimeType,'image/png');
 assert.match(images[0].name,/\[PDF page 1\/2\]/);
 assert.match(images[0].note,new RegExp(signature));
 assert.equal(Buffer.from(images[0].data,'base64').subarray(0,8).toString('hex'),'89504e470d0a1a0a');
});

test('mixed PDF + text chat references preserve both source snapshots',async()=>{
 const result=await controller().requestContext(folder,'Summarize attachments',[ref(pdf),ref(txt)]);
 assert.equal(result.length,4);
 assert.equal(result[0].kind,'text');
 assert.equal(result[1].content,'notes\nhello\n');
});

test('non-PDF chat reference never activates PDF visuals',async()=>{
 const result=await controller().requestContext(folder,'Show diagrams',[ref(txt)]);
 assert.equal(result.length,1);
 assert.equal(result[0].content,'notes\nhello\n');
});

test('multiple PDF chat references each retain their own rendering source',async()=>{
 const result=await controller().requestContext(folder,'请查看第一页图表',[ref(pdf),ref(pdf2)]);
 assert.equal(result.filter(x=>x.kind==='text').length,2);
 assert.equal(result.filter(x=>x.kind==='image').length,4);
});

test('invalid reference is rejected and does not poison next independent request',async()=>{
 const ctl=controller();
 await assert.rejects(ctl.requestContext(folder,'x',[{value:{uri:uri(pdf,'untitled')}}]),e=>e.code==='REFERENCE_URI');
 const result=await ctl.requestContext(folder,'x',[ref(txt)]);
 assert.equal(result.length,1);
});

test('existing Add files entry continues to render PDF pages',async()=>{
 const ctl=controller(),view={};
 await ctl.files(view,[uri(pdf)]);
 const result=await ctl.prepare(view,tmp,'请看PDF第一页图表');
 assert.equal(result.filter(x=>x.kind==='image').length,2);
 ctl.dispose(view);
});

test('existing drop entry continues to render PDF pages',async()=>{
 const ctl=controller(),view={};
 await ctl.dropped(view,{files:[{name:'two-pages.pdf',data:fs.readFileSync(pdf).toString('base64')}]});
 const result=await ctl.prepare(view,tmp,'请看PDF第一页图表');
 assert.equal(result.filter(x=>x.kind==='image').length,2);
 ctl.dispose(view);
});
