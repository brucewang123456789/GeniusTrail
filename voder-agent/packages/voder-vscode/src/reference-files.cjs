'use strict';
/** Explicit file snapshots only. No project writes, directory recursion, shell or cloud storage. */
const fs=require('node:fs/promises'),path=require('node:path'),{constants}=require('node:fs');
const {randomUUID}=require('node:crypto'),{spawn}=require('node:child_process');
const {LIMITS,digest,imageMime,decodeBase64,validateReferences}=require('./reference-contract.cjs');
const documents=require('./document-input.cjs');
const pdfVision=require('./pdf-vision.cjs');
const fail=(code,message)=>{throw Object.assign(Error(message),{code});};
// Document parsers receive no model credentials, sandbox tokens or proxy secrets.
const parserEnv=()=>({...Object.fromEntries(['PATH','SystemRoot','WINDIR','TEMP','TMP'].filter(k=>typeof process.env[k]==='string').map(k=>[k,process.env[k]])),LANG:'C.UTF-8',LC_ALL:'C'});
const inside=(root,file)=>{const r=path.relative(root,file);return r===''||r!=='..'&&!r.startsWith('..'+path.sep)&&!path.isAbsolute(r);};
function rangeLines(text,range){
 if(!range)return text;
 if(range.unit!=='line'||!Number.isSafeInteger(range.start)||!Number.isSafeInteger(range.end)||range.start<1||range.end<range.start)fail('REFERENCE_RANGE','Use #Lstart-Lend for an explicit line range');
 const lines=text.match(/[^\n]*\n|[^\n]+$/g)||[''];
 if(range.end>lines.length)fail('REFERENCE_RANGE','Requested line range exceeds this file; nothing was truncated');
 return lines.slice(range.start-1,range.end).join('');
}
async function safeRead(file,{root,signal}={}){
 if(signal?.aborted)fail('REFERENCE_CANCELLED','Reference loading cancelled');
 const full=path.resolve(file);
 if(root&&!inside(path.resolve(root),full))fail('REFERENCE_SCOPE','@ references must stay inside the selected workspace; use Add files for an external document');
 let current=path.parse(full).root;
 for(const part of full.slice(current.length).split(path.sep).filter(Boolean)){
  current=path.join(current,part);const st=await fs.lstat(current);
  if(st.isSymbolicLink())fail('REFERENCE_SYMLINK','Symbolic links are not accepted as reference files');
 }
 const before=await fs.lstat(full);
 if(!before.isFile()||before.size>LIMITS.fileBytes)fail('REFERENCE_FILE_LIMIT','Select a regular file within available host memory capacity');
 const fd=await fs.open(full,constants.O_RDONLY|(constants.O_NOFOLLOW||0)|(constants.O_NONBLOCK||0));
 try{
  const opened=await fd.stat();
  if(!opened.isFile()||opened.dev!==before.dev||opened.ino!==before.ino||opened.size>LIMITS.fileBytes)fail('REFERENCE_CHANGED','Reference changed while opening');
  const canonical=await fs.realpath(full);
  if(root&&!inside(await fs.realpath(root),canonical))fail('REFERENCE_SCOPE','Reference escaped workspace');
  // Read a bounded descriptor even if another process grows the file concurrently.
  const buffer=Buffer.alloc(opened.size+1);let offset=0;
  while(offset<buffer.length){if(signal?.aborted)fail('REFERENCE_CANCELLED','Reference loading cancelled');const {bytesRead}=await fd.read(buffer,offset,buffer.length-offset,null);if(!bytesRead)break;offset+=bytesRead;}
  const after=await fd.stat(),atPath=await fs.lstat(full);
  if(offset!==opened.size||after.size!==opened.size||after.mtimeMs!==opened.mtimeMs||after.ctimeMs!==opened.ctimeMs||atPath.isSymbolicLink()||atPath.ino!==opened.ino||atPath.dev!==opened.dev)fail('REFERENCE_CHANGED','Reference changed while reading; select it again');
  return buffer.subarray(0,offset);
 }finally{await fd.close();}
}
async function pdfPageCount(bytes,{signal,spawnFn=spawn}={}){
 return new Promise((resolve,reject)=>{
  let child,done=false,total=0;const chunks=[];
  const finish=(error,value)=>{if(done)return;done=true;clearTimeout(timer);signal?.removeEventListener('abort',abort);if(error){child?.kill();reject(error);}else resolve(value);};
  const error=(code,message)=>Object.assign(Error(message),{code});
  const abort=()=>finish(error('REFERENCE_CANCELLED','PDF inspection cancelled'));
  const timer=setTimeout(()=>finish(error('REFERENCE_PDF_TIMEOUT','PDF page inspection timed out')),30000);
  if(signal?.aborted){abort();return;}
  try{child=spawnFn('pdfinfo',['-'],{stdio:['pipe','pipe','pipe'],shell:false,windowsHide:true,env:parserEnv()});}catch{finish(error('REFERENCE_PDF_UNAVAILABLE','PDF page inspection is unavailable'));return;}
  signal?.addEventListener('abort',abort,{once:true});
  child.on('error',()=>finish(error('REFERENCE_PDF_UNAVAILABLE','PDF page inspection is unavailable on this host')));
  child.stdout.on('data',chunk=>{total+=chunk.length;if(total>16384)finish(error('REFERENCE_PDF_ERROR','PDF metadata exceeds bounded inspection'));else chunks.push(chunk);});
  child.stderr.on('data',()=>{});child.stdin.on('error',()=>{});
  child.on('close',code=>{if(done)return;const match=Buffer.concat(chunks).toString('utf8').match(/^Pages:\s+(\d+)\s*$/m);if(code!==0||!match)finish(error('REFERENCE_PDF_ERROR','Cannot determine PDF page count'));else finish(null,Number(match[1]));});
  child.stdin.end(bytes);
 });
}
async function extractPdf(bytes,{range,signal,spawnFn=spawn}={}){
 if(bytes.subarray(0,5).toString()!=='%PDF-')fail('REFERENCE_FORMAT','Invalid PDF signature');
 if(range&&(range.unit!=='page'||!Number.isSafeInteger(range.start)||!Number.isSafeInteger(range.end)||range.start<1||range.end<range.start||range.end>10000000))fail('REFERENCE_RANGE','Invalid PDF page range');
 if(range&&range.end>await pdfPageCount(bytes,{signal,spawnFn}))fail('REFERENCE_RANGE','Requested page range exceeds this PDF; nothing was truncated');
 return new Promise((resolve,reject)=>{
  if(signal?.aborted){reject(Object.assign(Error('PDF extraction cancelled'),{code:'REFERENCE_CANCELLED'}));return;}
  const args=['-enc','UTF-8','-layout',...(range?['-f',String(range.start),'-l',String(range.end)]:[]),'-','-'];
  let child,done=false,total=0;const chunks=[];
  const finish=(error,text)=>{if(done)return;done=true;clearTimeout(timer);signal?.removeEventListener('abort',abort);if(error){child?.kill();reject(error);}else resolve(text);};
  const error=(code,message)=>Object.assign(Error(message),{code});
  const abort=()=>finish(error('REFERENCE_CANCELLED','PDF extraction cancelled'));
  const timer=setTimeout(()=>finish(error('REFERENCE_PDF_TIMEOUT','PDF text extraction exceeded 60 seconds; nothing was truncated or sent')),60000);
  try{child=spawnFn('pdftotext',args,{stdio:['pipe','pipe','pipe'],shell:false,windowsHide:true,env:parserEnv()});}catch(e){finish(error('REFERENCE_PDF_UNAVAILABLE','PDF text extraction is unavailable on the extension host'));return;}
  signal?.addEventListener('abort',abort,{once:true});
  child.on('error',e=>finish(error(e.code==='ENOENT'?'REFERENCE_PDF_UNAVAILABLE':'REFERENCE_PDF_ERROR',e.code==='ENOENT'?'PDF text extraction is unavailable on this host; attach exported UTF-8 text instead':'Unable to start PDF extraction')));
  child.stdout.on('data',chunk=>{total+=chunk.length;if(total>LIMITS.textBytes)finish(error('REFERENCE_LIMIT','PDF text exceeds this model-request capacity; no content was truncated or sent. Large-document indexing is required for larger files'));else chunks.push(chunk);});
  child.stderr.on('data',()=>{});child.stdin.on('error',()=>{});
  child.on('close',code=>{
   if(done)return;
   if(code!==0){finish(error('REFERENCE_PDF_ERROR','PDF extraction failed; encrypted, damaged or invalid page selection'));return;}
   let text;try{text=new TextDecoder('utf-8',{fatal:true}).decode(Buffer.concat(chunks));}catch{finish(error('REFERENCE_ENCODING','PDF extraction produced invalid UTF-8'));return;}
   if(!text.trim()){finish(error('REFERENCE_PDF_NO_TEXT','This PDF has no extractable text; OCR and visual PDF analysis are not enabled'));return;}
   finish(null,text);
  });
  child.stdin.end(bytes);
 });
}
async function fromBytes({name,bytes,source='attachment',range,signal,pdf=extractPdf}){
 if(signal?.aborted)fail('REFERENCE_CANCELLED','Reference loading cancelled');
 if(typeof name!=='string'||!Buffer.isBuffer(bytes)||bytes.length>LIMITS.fileBytes)fail('REFERENCE_FILE_LIMIT','Invalid or oversized reference file');
 const common={id:randomUUID(),name,source};const mime=imageMime(bytes);let item;
 if(mime||documents.EXTRA_IMAGES.has(path.extname(name).toLowerCase())){
  if(range)fail('REFERENCE_RANGE','Line and page ranges do not apply to images');
  const converted=await documents.runDecoder('image',name,bytes,{signal});
  const image=converted.unchanged?bytes:Buffer.from(converted.image||'','base64');
  item={...common,kind:'image',mimeType:converted.unchanged?mime:'image/png',data:image.toString('base64'),sha256:digest(image),
   ...(converted.unchanged?{}:{note:'Complete canvas decoded; original dimensions '+converted.originalSize.join('x')+'; sent '+converted.outputSize.join('x')+'; original SHA256 '+digest(bytes)})};
 }else if(/\.(?:pkl|pickle)$/i.test(name)){
  const decoded=await documents.runDecoder('pickle',name,bytes,{signal});
  item={...common,kind:'text',content:decoded.text,sha256:digest(Buffer.from(decoded.text)),note:'Static pickle inspection only; source SHA256 '+digest(bytes)};
 }else if(documents.OFFICE.has(path.extname(name).toLowerCase())){
  if(/\.(?:docx|docm|dotx|pptx|pptm|potx|ppsx|xlsx|xlsm|xltx|odt|odp|ods)$/i.test(name)&&bytes.subarray(0,2).toString()!=='PK')fail('REFERENCE_FORMAT','Office extension does not match its container format');
  if(range&&range.unit!=='page')fail('REFERENCE_RANGE','Office document ranges use physical rendered pages');
  const converted=await documents.documentPdf(name,bytes,{signal});
  const parsed=await fromBytes({name:name+'.pdf',bytes:converted.pdf,source,range,signal});
  const content=(converted.structuredText?converted.structuredText+'\n\n':'')+parsed.content;
  item={...parsed,...common,content,sha256:digest(Buffer.from(content)),note:'Office decoded; all rendered pages accompany this reference; original SHA256 '+digest(bytes)};
 }else if(path.extname(name).toLowerCase()==='.pdf'||bytes.subarray(0,5).toString()==='%PDF-'){
  const pages=await pdfVision.pageCount(bytes,{signal});
  let content,ocr=false,ocrLanguage='',ocrPages=[];
  try{content=await pdf(bytes,{range,signal});}
  catch(e){
   if(e.code!=='REFERENCE_PDF_NO_TEXT')throw e;
   const selected=range?Array.from({length:range.end-range.start+1},(_,i)=>range.start+i):undefined;
   try{const result=await pdfVision.ocrDocument(bytes,{count:pages,signal,selectedPages:selected});
   content=result.text;ocr=true;ocrLanguage=result.language;ocrPages=selected||Array.from({length:pages},(_,i)=>i+1);}catch(ocrError){if(['REFERENCE_CANCELLED','REFERENCE_LIMIT'].includes(ocrError.code))throw ocrError;content=Array.from({length:range?range.end-range.start+1:pages},(_,i)=>'[PDF PHYSICAL PAGE '+((range?range.start:1)+i)+'/'+pages+']\n[No text layer; page image is authoritative; OCR unavailable]\n').join('');}
  }
  if(signal?.aborted)fail('REFERENCE_CANCELLED','PDF reference cancelled');
  // Poppler emits a form-feed at each physical page boundary. Preserve text and prove page mapping.
  if(!ocr&&pdf===extractPdf){
   const segments=content.split('\f');
   if(segments.length>1&&segments[segments.length-1]==='')segments.pop();
   const expected=range?range.end-range.start+1:pages;
   if(segments.length===expected){
    const originals=[];
    for(let i=0;i<segments.length;i++){
     const n=(range?range.start:1)+i;
     let pageText=segments[i];
     if(!pageText.trim()){
      try{const scan=await pdfVision.ocrDocument(bytes,{count:pages,signal,selectedPages:[n]});
      pageText=scan.blocks[0];ocr=true;ocrLanguage=scan.language;ocrPages.push(n);}catch(ocrError){if(['REFERENCE_CANCELLED','REFERENCE_LIMIT'].includes(ocrError.code))throw ocrError;pageText='[No text layer; page image is authoritative; OCR unavailable]';}
     }
     originals.push('[PDF PHYSICAL PAGE '+n+'/'+pages+']\n'+pageText);
    }
    content=originals.join('\n');
   }
  }
  item={...common,kind:'text',content,sha256:digest(Buffer.from(content)),...(range?{range}:{}),note:'PDF physical pages='+pages+' (pdfinfo verified); '+(ocr?'OCR count='+ocrPages.length+' (language='+ocrLanguage+', transcription may err)':'text extracted; figures need visual analysis')+'; Source SHA256: '+digest(bytes)};
 }else{
  if(/\.(?:gdoc|gslides|gsheet)$/i.test(name))fail('REFERENCE_CLOUD_POINTER','This file is a cloud-document pointer; export the actual document as DOCX, PPTX, XLSX or PDF');
  if(/\.(?:zip|exe|dll|so|bin|png|jpe?g|webp)$/i.test(name))fail('REFERENCE_FORMAT','Unsupported or mismatched file format; use UTF-8 text, PNG, JPEG, WebP or a text PDF');
  let encoding='utf-8',input=bytes;
  if(bytes[0]===255&&bytes[1]===254){encoding='utf-16le';input=bytes.subarray(2);}
  else if(bytes[0]===254&&bytes[1]===255){encoding='utf-16be';input=bytes.subarray(2);}
  else if(bytes.includes(0))fail('REFERENCE_ENCODING','Binary file without a supported decoder');
  let content;try{content=new TextDecoder(encoding,{fatal:true,ignoreBOM:encoding==='utf-8'}).decode(input);}catch{fail('REFERENCE_ENCODING','File encoding is invalid; nothing was substituted');}
  if(content.includes('\0'))fail('REFERENCE_ENCODING','Binary content cannot be read as text');
  content=rangeLines(content,range);
  item={...common,kind:'text',content,sha256:digest(Buffer.from(content)),...(range?{range}:{})};
 }
 return validateReferences([item])[0];
}
function mentions(goal){
 const out=[];
 // Delimited grammar avoids email addresses; quoted paths support spaces and Chinese names.
 const re=/(?:^|[\s(])@(?:"([^"\r\n]+)"(#[LP]\d+(?:-[LP]?\d+)?)?|([^\s<>"(),;，；]+))/gi;
 for(const m of goal.matchAll(re)){
  const raw=m[1]?m[1]+(m[2]||''):m[3],match=raw.match(/^(.*?)(?:#(L|P)(\d+)(?:-\2?(\d+))?)?$/i),file=match[1];
  if(file==='voder'||(!m[1]&&!match[2]&&!/[./]/.test(file)&&!/^(?:README|LICENSE|Makefile|Dockerfile|AGENTS|CHANGELOG)$/.test(file)))continue;
  if(!file||path.isAbsolute(file)||file.includes('\\')||file.split('/').some(p=>!p||p==='..'||p==='.'||p.startsWith('.'))||file.length>1024)fail('REFERENCE_SCOPE','Use a non-hidden workspace-relative @path, or select an external file with Add files');
  const range=match[2]?{unit:match[2].toUpperCase()==='L'?'line':'page',start:Number(match[3]),end:Number(match[4]||match[3])}:undefined;
  if(range&&(range.start<1||range.end<range.start||range.end>10000000))fail('REFERENCE_RANGE','Invalid reference range');
  if(!out.some(x=>x.file===file&&JSON.stringify(x.range)===JSON.stringify(range)))out.push({file,range});
  if(out.length>LIMITS.items)fail('REFERENCE_LIMIT','Too many file references');
 }
 return out;
}
async function normalizeImageBatch(items,signal){
 const images=items.filter(x=>x.kind==='image');
 if(images.length<15)return items;
 const normalized=[];
 for(const item of items){
  if(item.kind!=='image'){normalized.push(item);continue;}
  const bytes=Buffer.from(item.data,'base64');
  if(item.mimeType==='image/png'&&bytes.length>=24&&bytes.readUInt32BE(16)<=4096&&bytes.readUInt32BE(20)<=4096){normalized.push(item);continue;}
  const converted=await documents.runDecoder('image4096',item.name,bytes,{signal});
  if(converted.unchanged){normalized.push(item);continue;}
  const output=Buffer.from(converted.image,'base64');
  normalized.push({...item,mimeType:'image/png',data:converted.image,sha256:digest(output),note:'Full canvas resized for multi-image provider capacity; source image SHA256 '+item.sha256+'; original source '+(item.note?.match(/original SHA256 ([a-f0-9]{64})/)?.[1]||item.sha256)});
 }
 return normalized;
}
class ReferenceStore{
 constructor(){this.items=[];this.pdfSources=new Map();this.version=0;this.disposed=false;}
 add(items){if(this.disposed)fail('REFERENCE_CANCELLED','Reference view closed');const checked=validateReferences([...this.items,...items]);this.items=[...checked];this.version++;return this.list();}
 remove(id){this.items=this.items.filter(x=>x.id!==id);this.pdfSources.delete(id);this.version++;return this.list();}
 clear(){this.items=[];this.pdfSources.clear();this.version++;}
 dispose(){this.clear();this.disposed=true;}
 list(){return this.items.map(({content,data,...metadata})=>({...metadata,bytes:content!==undefined?Buffer.byteLength(content):Buffer.from(data,'base64').length}));}
 snapshot(){return validateReferences(this.items);}
 async prepare(root,goal,{signal,read=safeRead,pdf}={}){
  const initial=this.snapshot(),version=this.version,extra=[],visualSources=[];
  for(const {file,range} of mentions(goal)){
   const bytes=await read(path.resolve(root,file),{root,signal});
   const loaded=await require('./archive-input.cjs').loadReferences({name:file,source:'workspace',bytes,range,signal,...(pdf?{pdf}:{})});
   extra.push(...loaded.items);validateReferences([...initial,...extra]);visualSources.push(...loaded.visualSources);
  }
  const images=[];
  if(true){ // Every attached document includes its visual plane, independent of wording.
   for(const item of initial){const bytes=this.pdfSources.get(item.id);if(bytes)visualSources.push({item,bytes});}
   for(const {item,bytes} of visualSources){const converted=await documents.documentPdf(item.name,bytes,{signal});images.push(...await pdfVision.visualReferences(converted.pdf,{name:item.name,source:item.source,goal,signal,documentText:item.content,range:item.range,originalDigest:digest(bytes)}));validateReferences([...initial,...extra,...images]);}
  }
  if(this.disposed||version!==this.version)fail('REFERENCE_CHANGED','References changed during preparation; submit again');
  return validateReferences(await normalizeImageBatch([...initial,...extra,...images],signal));
 }
}
module.exports={safeRead,rangeLines,extractPdf,fromBytes,mentions,ReferenceStore,decodeBase64,LIMITS};
