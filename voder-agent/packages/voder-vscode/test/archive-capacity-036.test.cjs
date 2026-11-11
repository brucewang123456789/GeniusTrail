'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),fsp=require('node:fs/promises'),path=require('node:path'),os=require('node:os');
const fixtures=process.env.VODER_ARCHIVE_FIXTURES||path.resolve(__dirname,'../../../../fixtures036');
const oldFixtures=process.env.VODER_DOCUMENT_FIXTURES||path.resolve(__dirname,'../../../../public/Voder_File_Multimodal_0.1.35_Patch/fixtures');
const {decodeArchive,loadReferences,extractArchiveFile}=require('../src/archive-input.cjs');
const {ReferenceController}=require('../src/reference-ui.cjs');
const {fromBytes,ReferenceStore}=require('../src/reference-files.cjs');
const {digest,validateReferences}=require('../src/reference-contract.cjs');
const {LIMITS}=require('../src/input-capacity.cjs');
const read=n=>fs.readFileSync(path.join(fixtures,n));
for(const name of ['sample.zip','sample.7z','sample.tar','sample.tar.gz','sample.tar.bz2','sample.tar.xz','sample.tar.zst','sample.tar.lz4','sample.tar.lz','sample.tar.Z','sample.cpio','sample.ar','sample.iso','sample.xar','regular-rar4.rar','test_read_format_rar5_stored.rar','test_read_format_cab_1.cab'])test('real archive decoder: '+name,async()=>{
 const r=await decodeArchive(name,read(name));assert.ok(r.entries.some(x=>x.kind==='file'));
 for(const e of r.entries.filter(x=>x.kind==='file'))assert.equal(digest(Buffer.from(e.data,'base64')),e.sha256);
});
for(const name of ['plain.txt.gz','plain.txt.bz2','plain.txt.xz'])test('standalone compression preserves complete stream: '+name,async()=>{
 const r=await loadReferences({name,bytes:read(name)});assert.equal(r.items.at(-1).content,'RAW_STREAM_036');
});
for(const name of ['traversal.zip','absolute.zip','windows.zip','symlink.zip','hardlink.tar','duplicate.zip','conflict.zip'])test('unsafe archive rejected before extracting or submitting: '+name,async()=>{
 await assert.rejects(decodeArchive(name,read(name)),e=>/^REFERENCE_ARCHIVE_(?:PATH|LINK|DUPLICATE)/.test(e.code));
});
test('nested archives recursively retain final member',async()=>{
 const r=await loadReferences({name:'nested.zip',bytes:read('nested.zip')});assert.ok(r.items.some(x=>x.content==='LAST_MEMBER_036'));assert.equal(r.items.at(-1).content,'OUTER_LAST_036');
});
test('archive manifest explicitly identifies binary assets',async()=>{
 const r=await loadReferences({name:'sample.zip',bytes:read('sample.zip')});const m=JSON.parse(r.items[0].content);assert.equal(m.member_count,3);assert.equal(m.members.find(x=>x.path==='binary.dat').status,'METADATA_ONLY');assert.ok(r.items.some(x=>x.name.endsWith('code.py')&&x.content.includes('READ_ONLY_036')));
});
test('600 text members exceed former count limits without truncation',async()=>{
 const r=await loadReferences({name:'many.zip',bytes:read('many.zip')});assert.equal(r.items.length,601);assert.equal(r.items.at(-1).content,'599');validateReferences(r.items);
});
test('image of exactly the reported 12387x7498 dimensions is accepted and canvas resized',async()=>{
 const r=await fromBytes({name:'benchmark.png',bytes:read('large-12387x7498.png')});assert.equal(r.kind,'image');const b=Buffer.from(r.data,'base64');assert.equal(b.readUInt32BE(16),8192);assert.equal(b.readUInt32BE(20),4959);assert.match(r.note,/12387x7498/);assert.equal(r.sha256,digest(b));
});
test('multi-image preparation respects provider 4096-side capacity without dropping images',async()=>{
 const store=new ReferenceStore();for(let n=0;n<15;n++)store.add([await fromBytes({name:'image'+n+'.png',bytes:read('wide.png')})]);
 const r=await store.prepare(fixtures,'看图');assert.equal(r.length,15);for(const x of r){const b=Buffer.from(x.data,'base64');assert.ok(b.readUInt32BE(16)<=4096);assert.match(x.note,/resized/);}
});
test('unchanged PNG original digest is preserved',async()=>{
 const b=fs.readFileSync(path.join(oldFixtures,'embedded.png'));assert.equal((await fromBytes({name:'x.png',bytes:b})).sha256,digest(b));
});
test('all archive attachment routes use the same recursive loader',async()=>{
 const ctl=new ReferenceController({vscode:{window:{},workspace:{textDocuments:[]}}}),v={};
 try{
  await ctl.files(v,[{scheme:'file',fsPath:path.join(fixtures,'nested.zip')}]);let r=await ctl.prepare(v,fixtures,'读附件');assert.ok(r.some(x=>x.content==='LAST_MEMBER_036'));
  ctl.state(v).store.clear();await ctl.dropped(v,{files:[{name:'nested.zip',data:read('nested.zip').toString('base64')}]});r=await ctl.prepare(v,fixtures,'读附件');assert.ok(r.some(x=>x.content==='LAST_MEMBER_036'));
  const chat=await ctl.requestContext({uri:{fsPath:fixtures}},'读附件',[{value:{scheme:'file',fsPath:path.join(fixtures,'nested.zip')}}]);assert.ok(chat.some(x=>x.content==='OUTER_LAST_036'));
  const mentioned=await new ReferenceStore().prepare(fixtures,'读 @nested.zip');assert.ok(mentioned.some(x=>x.content==='OUTER_LAST_036'));
 }finally{ctl.dispose(v);}
});
test('archive-contained PDF and image reach prepared visual context',async()=>{
 const ctl=new ReferenceController({vscode:{window:{},workspace:{textDocuments:[]}}}),v={};
 try{await ctl.files(v,[{scheme:'file',fsPath:path.join(fixtures,'pdf-image.zip')}]);const r=await ctl.prepare(v,fixtures,'概括');assert.equal(r.filter(x=>x.kind==='image').length,3);assert.ok(r.some(x=>x.content?.includes('PDF PHYSICAL PAGE 2/2')));}finally{ctl.dispose(v);}
});
test('Office documents in an archive retain spreadsheet cells and all rendered pages',async()=>{
 const ctl=new ReferenceController({vscode:{window:{},workspace:{textDocuments:[]}}}),v={};
 try{await ctl.files(v,[{scheme:'file',fsPath:path.join(fixtures,'mixed.zip')}]);const r=await ctl.prepare(v,fixtures,'读图文');assert.ok(r.some(x=>x.content?.includes('DOCX_UNIQUE_FIRST')));assert.ok(r.some(x=>x.content?.includes('PPTX_SLIDE_TWO')));assert.ok(r.filter(x=>x.kind==='image').length>=6);}finally{ctl.dispose(v);}
});
test('cancelled archive read produces no partial reference store',async()=>{
 const ctl=new ReferenceController({vscode:{window:{},workspace:{textDocuments:[]}}}),v={};ctl.state(v).abort.abort();await assert.rejects(ctl.files(v,[{scheme:'file',fsPath:path.join(fixtures,'sample.zip')}]),{code:'REFERENCE_CANCELLED'});assert.deepEqual(ctl.state(v).store.items,[]);ctl.dispose(v);
});
test('corrupt archive explicitly fails instead of producing a partial list',async()=>{await assert.rejects(decodeArchive('bad.zip',Buffer.from('PKbad')),e=>e.code.startsWith('REFERENCE_ARCHIVE'));});
test('nested expansion consumes one shared resource budget',async()=>{await assert.rejects(loadReferences({name:'nested.zip',bytes:read('nested.zip')},{expanded:LIMITS.expandedBytes,metadata:0}),{code:'REFERENCE_ARCHIVE_RESOURCE'});});
test('archive extraction writes exact bytes into a new directory and never executes code',async()=>{
 const parent=await fsp.mkdtemp(path.join(os.tmpdir(),'voder-archive-test-'));
 try{await fsp.writeFile(path.join(parent,'existing.txt'),'KEEP');const r=await extractArchiveFile(path.join(fixtures,'nested.zip'),parent);assert.equal(r.status,'EXTRACTED');assert.equal(await fsp.readFile(path.join(parent,'existing.txt'),'utf8'),'KEEP');assert.deepEqual(await fsp.readFile(path.join(r.directory,'inner.tar.gz')),read('sample.tar.gz'));assert.equal(await fsp.readFile(path.join(r.directory,'last.txt'),'utf8'),'OUTER_LAST_036');assert.equal((await fsp.stat(path.join(r.directory,'last.txt'))).mode&0o111,0);}finally{await fsp.rm(parent,{recursive:true,force:true});}
});
test('unsafe archive cannot create a destination folder',async()=>{
 const parent=await fsp.mkdtemp(path.join(os.tmpdir(),'voder-archive-test-'));
 try{await assert.rejects(extractArchiveFile(path.join(fixtures,'traversal.zip'),parent));assert.deepEqual(await fsp.readdir(parent),[]);}finally{await fsp.rm(parent,{recursive:true,force:true});}
});
test('large source code is parsed without executing it',async()=>{const content='# static source\n'+('print("DO_NOT_EXECUTE")\n'.repeat(50000));const r=await fromBytes({name:'code.py',bytes:Buffer.from(content)});assert.equal(r.content,content);});
test('pickle is statically inspected, never unpickled',async()=>{const r=await fromBytes({name:'danger.pkl',bytes:read('danger.pkl')});assert.match(r.content,/REDUCE/);assert.match(r.note,/Static pickle/);assert.ok(!fs.existsSync(path.join(fixtures,'PICKLE_MUST_NOT_EXECUTE')));});
test('resource capacities exceed historical fixed quotas on this host',()=>{assert.ok(LIMITS.items>16);assert.ok(LIMITS.blocks>512);assert.ok(LIMITS.wireBytes>40*1024*1024);assert.ok(LIMITS.imageBytes>20*1024*1024);});
test('archive extraction command is contributed to the installed extension',()=>{const pkg=require('../package.json');assert.ok(pkg.contributes.commands.some(x=>x.command==='voder.extractArchive'));});
test('public text/code extension list and common modern code remain complete data',async()=>{
 const extensions='.asm .bat .c .cc .conf .cpp .css .cxx .def .dic .eml .h .hh .htm .html .ics .ifb .in .js .json .ksh .list .log .markdown .md .mht .mhtml .mime .mjs .nws .pl .py .rst .s .sql .srt .text .txt .vcf .vtt .xml .ts .tsx .jsx .rs .go .java .cs .php .rb .sh .tex .yaml .toml .ipynb .csv .tsv .iif'.split(' ');
 for(const ext of extensions){const body='全文数据_036\nLAST_LINE\n';assert.equal((await fromBytes({name:'file'+ext,bytes:Buffer.from(body)})).content,body);}
});
test('201-page document passes the former 200-page boundary with every image',async()=>{
 const p=require('../src/pdf-vision.cjs');const refs=await p.visualReferences(read('two-hundred-one.pdf'),{name:'201.pdf',source:'attachment',goal:'概括全部'});assert.equal(refs.length,201);assert.match(refs.at(-1).name,/page 201\/201/);
});
test('disk extraction streams a 300MiB member beyond inline expansion capacity',async()=>{
 const parent=await fsp.mkdtemp(path.join(os.tmpdir(),'voder-archive-large-'));
 try{
  await assert.rejects(decodeArchive('large.zip',read('large-expanded.zip')),{code:'REFERENCE_ARCHIVE_RESOURCE'});
  const r=await extractArchiveFile(path.join(fixtures,'large-expanded.zip'),parent);
  assert.equal((await fsp.stat(path.join(r.directory,'large.bin'))).size,300*1024*1024);
  assert.equal(await fsp.readFile(path.join(r.directory,'last.txt'),'utf8'),'LARGE_ARCHIVE_LAST_MEMBER');
 }finally{await fsp.rm(parent,{recursive:true,force:true});}
});
