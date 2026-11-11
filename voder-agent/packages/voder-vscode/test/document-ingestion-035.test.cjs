'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const root=process.env.VODER_AGENT_TEST_ROOT||path.resolve(__dirname,'../../..');
const fixtures=process.env.VODER_DOCUMENT_FIXTURES||path.join(root,'fixtures');
const {ReferenceController}=require('../src/reference-ui.cjs');
const {fromBytes}=require('../src/reference-files.cjs');
const {validateReferences,digest}=require('../src/reference-contract.cjs');
const {saveVisionModel}=require('../src/vision-setup.cjs');
async function attached(name,goal='概括全文'){
 const ctl=new ReferenceController({vscode:{window:{},workspace:{textDocuments:[]}}});
 const file=path.join(fixtures,name),folder={uri:{fsPath:fixtures}};
 return ctl.requestContext(folder,goal,[{value:{uri:{scheme:'file',fsPath:file}}}]);
}
for(const goal of ['概括全文','分析公式','读取表格'])test('PDF includes both pages for ordinary wording: '+goal,async()=>{
 const refs=await attached('two-pages.pdf',goal);assert.equal(refs.filter(x=>x.kind==='text').length,1);assert.equal(refs.filter(x=>x.kind==='image').length,2);
 for(const x of refs.filter(x=>x.kind==='image'))assert.equal(digest(Buffer.from(x.data,'base64')),x.sha256);
});
test('45-page PDF has 45 actual image blocks including last page; no sixteen-item truncation',async()=>{
 const refs=await attached('forty-five.pdf');const images=refs.filter(x=>x.kind==='image');assert.equal(images.length,45);assert.match(images[44].name,/page 45\/45/);assert.match(refs[0].content,/PHYSICAL_PAGE_45/);
});
test('PDF signature works without an extension',async()=>{const bytes=fs.readFileSync(path.join(fixtures,'two-pages.pdf'));const r=await fromBytes({name:'no-extension',bytes});assert.equal(r.kind,'text');assert.match(r.note,/physical pages=2/);});
test('explicit @PDF page range sends exactly selected visual pages',async()=>{
 const ctl=new ReferenceController({vscode:{window:{},workspace:{textDocuments:[]}}}),v={};
 try{const refs=await ctl.prepare(v,fixtures,'读取 @two-pages.pdf#P2-P2');assert.equal(refs.filter(x=>x.kind==='image').length,1);assert.match(refs.at(-1).name,/page 2\/2/);}finally{ctl.dispose(v);}
});
for(const [name,needle,minImages] of [['sample.docx','DOCX_UNIQUE_FIRST',1],['sample.pptx','PPTX_SLIDE_TWO',2],['sample.xlsx','ROW_1040_LAST',2]])test('Office text, structure and actual rendered visuals: '+name,async()=>{
 const refs=await attached(name);assert.match(refs[0].content,new RegExp(needle));assert.ok(refs.filter(x=>x.kind==='image').length>=minImages);
 if(name.endsWith('xlsx')){assert.match(refs[0].content,/SUM\(1,2\)/);assert.match(refs[0].content,/\[SHEET Last\]/);}
});
for(const name of ['embedded.png','sample.jpeg','sample.webp','sample.bmp','sample.tiff','sample.gif'])test('image decoder accepts '+name,async()=>{const r=await fromBytes({name,bytes:fs.readFileSync(path.join(fixtures,name))});assert.equal(r.kind,'image');assert.equal(r.sha256,digest(Buffer.from(r.data,'base64')));});
for(const [name,needle,minImages] of [['sample.doc','DOCX_UNIQUE_FIRST',1],['sample.rtf','DOCX_UNIQUE_FIRST',1],['sample.odt','DOCX_UNIQUE_FIRST',1],['sample.ppt','PPTX_SLIDE_TWO',2],['sample.odp','PPTX_SLIDE_TWO',2],['sample.xls','LEGACY_LAST',2],['sample.ods','LEGACY_LAST',2]])test('legacy/ODF real conversion retains text and visual pages: '+name,async()=>{const refs=await attached(name);assert.match(refs[0].content,new RegExp(needle));assert.ok(refs.filter(x=>x.kind==='image').length>=minImages);});
test('code is data, UTF-16 with BOM is decoded without substitution',async()=>{assert.match((await fromBytes({name:'sample.py',bytes:fs.readFileSync(path.join(fixtures,'sample.py'))})).content,/READ_AS_DATA_NEVER_EXECUTE/);assert.equal((await fromBytes({name:'utf16.txt',bytes:fs.readFileSync(path.join(fixtures,'utf16.txt'))})).content,'UTF16_TEXT');});
test('CSV retains multiline records and final row verbatim',async()=>{assert.equal((await fromBytes({name:'sample.csv',bytes:fs.readFileSync(path.join(fixtures,'sample.csv'))})).content,fs.readFileSync(path.join(fixtures,'sample.csv'),'utf8'));});
test('corrupt Office package fails, binary is not interpreted as prose',async()=>{await assert.rejects(fromBytes({name:'bad.docx',bytes:Buffer.from('PKbad')}),e=>e.code==='REFERENCE_ARCHIVE_INVALID');await assert.rejects(fromBytes({name:'unknown.dat',bytes:Buffer.from([0,1,0])}),{code:'REFERENCE_ENCODING'});});
test('wire/digest validation refuses tampering across reference collections',()=>{const b=fs.readFileSync(path.join(fixtures,'embedded.png')),ref={id:'11111111',kind:'image',name:'x',source:'attachment',data:b.toString('base64'),mimeType:'image/png',sha256:'0'.repeat(64)};assert.throws(()=>validateReferences([ref]),{code:'REFERENCE_DIGEST'});assert.throws(()=>validateReferences(Array(513).fill(ref)),{code:'REFERENCE_DIGEST'});});
test('cancelled file decode never starts conversion',async()=>{const c=new AbortController();c.abort();await assert.rejects(fromBytes({name:'sample.docx',bytes:fs.readFileSync(path.join(fixtures,'sample.docx')),signal:c.signal}),{code:'REFERENCE_CANCELLED'});});
test('vision setup preserves other models and stores credentials privately',async()=>{
 const fsp=require('node:fs/promises'),base=await fsp.mkdtemp(path.join(require('node:os').tmpdir(),'voder-vision-'));
 try{await fsp.writeFile(path.join(base,'additional-models.json'),JSON.stringify({version:1,models:[{provider:'voder-secondary-existing',id:'existing'}]}),{mode:0o600});await saveVisionModel({id:'fixture-vision',api:'openai-completions',baseUrl:'https://example.invalid/v1',apiKey:'fixture-not-a-real-key',contextWindow:262144},{base});const data=JSON.parse(await fsp.readFile(path.join(base,'additional-models.json')));assert.equal(data.models[0].id,'existing');assert.deepEqual(data.models[1].input,['text','image']);assert.equal((await fsp.stat(path.join(base,'additional-models.json'))).mode&0o077,0);}finally{await fsp.rm(base,{recursive:true,force:true});}
});
