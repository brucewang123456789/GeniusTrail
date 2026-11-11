'use strict';
const path=require('node:path'),fs=require('node:fs/promises'),{randomUUID}=require('node:crypto');
const {runDecoder,OFFICE,needsVisual}=require('./document-input.cjs');
const {LIMITS,digest,decodeBase64,validateReferences}=require('./reference-contract.cjs');
const EXT=/\.(?:zip|zipx|7z|rar|tar|tgz|tbz|tbz2|txz|tlz|tzst|gz|bz2|xz|zst|zstd|lz4|lzma|lz|lzo|z|br|lha|lzh|cab|cpio|ar|a|deb|rpm|iso|xar|war|jar|ear|apk|epub|cbz|cbr)$/i;
const fail=(code,message)=>{throw Object.assign(Error(message),{code});};
function isArchive(name,bytes){
 if(OFFICE.has(path.extname(name).toLowerCase()))return false;
 return EXT.test(name)||bytes?.subarray(0,4).equals(Buffer.from('PK\x03\x04','binary'))||bytes?.subarray(0,6).equals(Buffer.from([0x37,0x7a,0xbc,0xaf,0x27,0x1c]))||bytes?.subarray(0,7).toString('binary')==='Rar!\x1a\x07\x00'||bytes?.subarray(257,262).toString()==='ustar'||bytes?.subarray(0,2).equals(Buffer.from([31,139]))||bytes?.subarray(0,3).toString()==='BZh'||bytes?.subarray(0,6).equals(Buffer.from([253,55,122,88,90,0]));
}
async function decodeArchive(name,bytes,{signal}={}){
 let decoderName=name;
 if(!EXT.test(name)){if(bytes.subarray(0,2).equals(Buffer.from([31,139])))decoderName='content.gz';else if(bytes.subarray(0,3).toString()==='BZh')decoderName='content.bz2';else if(bytes.subarray(0,6).equals(Buffer.from([253,55,122,88,90,0])))decoderName='content.xz';}
 const result=await runDecoder('archive',decoderName,bytes,{signal});
 if(result.format==='raw'){const entry=result.entries?.find(x=>x.kind==='file');if(entry)entry.path=path.basename(name).replace(/\.(?:gz|bz2|xz|zst|zstd|lz4|lzma|lz|lzo|z|br)$/i,'')||'content';}
 if(!Array.isArray(result.entries)||!Number.isSafeInteger(result.expandedBytes))fail('REFERENCE_ARCHIVE_DECODE','Archive decoder returned invalid inventory');
 const seen=new Set();let total=0;
 for(const e of result.entries){
  if(typeof e.path!=='string'||!e.path||e.path.length>1024||/[\x00-\x1f\x7f\\]/.test(e.path)||path.posix.isAbsolute(e.path)||/^[a-z]:/i.test(e.path)||e.path.split('/').some(x=>!x||x==='..'||x==='.')||seen.has(e.path)||!['file','directory'].includes(e.kind))fail('REFERENCE_ARCHIVE_PATH','Unsafe or duplicate archive member');
  seen.add(e.path);
  if(e.kind==='file'){
   const body=decodeBase64(e.data,LIMITS.expandedBytes);
   if(body.length!==e.bytes||digest(body)!==e.sha256)fail('REFERENCE_ARCHIVE_DIGEST','Archive member changed during decoding');
   total+=body.length;
  }
 }
 if(total!==result.expandedBytes||total>LIMITS.expandedBytes)fail('REFERENCE_ARCHIVE_RESOURCE','Archive expansion exceeds available host memory');
 return result;
}
async function loadReferences({name,bytes,source='attachment',range,signal,pdf},state){
 const fromBytes=require('./reference-files.cjs').fromBytes;
 if(!isArchive(name,bytes)){
  const item=await fromBytes({name,bytes,source,range,signal,...(pdf?{pdf}:{})});
  return {items:[item],visualSources:needsVisual(name,bytes)?[{item,bytes}]:[]};
 }
 if(range)fail('REFERENCE_RANGE','Archive ranges require selecting an extracted member first');
 state??={expanded:0,metadata:0};
 const decoded=await decodeArchive(name,bytes,{signal});state.expanded+=decoded.expandedBytes;
 if(state.expanded>LIMITS.expandedBytes)fail('REFERENCE_ARCHIVE_RESOURCE','Nested archive expansion exceeds available host memory; nothing was sent');
 const items=[],visualSources=[],inventory=[];
 for(const entry of decoded.entries){
  if(signal?.aborted)fail('REFERENCE_CANCELLED','Archive preparation cancelled');
  const {data,...meta}=entry;
  state.metadata+=Buffer.byteLength(JSON.stringify(meta));
  if(state.metadata>LIMITS.textBytes)fail('REFERENCE_ARCHIVE_RESOURCE','Archive inventory exceeds host memory');
  if(entry.kind==='directory'){inventory.push({...meta,status:'DIRECTORY'});continue;}
  const body=Buffer.from(data,'base64');
  const childName=name+'!/'+entry.path;
  if(Buffer.byteLength(childName)>1024)fail('REFERENCE_ARCHIVE_PATH_TOO_LONG','Nested archive member identity exceeds protocol path capacity');
  try{
   const child=await loadReferences({name:childName,bytes:body,source,signal,pdf},state);
   items.push(...child.items);validateReferences(items);visualSources.push(...child.visualSources);
   inventory.push({...meta,status:'PARSED',reference_ids:child.items.map(x=>x.id)});
  }catch(e){
   // Binary assets remain extractable and appear explicitly in the inventory.
   // Corrupt supported formats, encrypted archives and capacity failures abort.
   if(!['REFERENCE_ENCODING','REFERENCE_FORMAT','REFERENCE_CLOUD_POINTER'].includes(e.code))throw e;
   inventory.push({...meta,status:'METADATA_ONLY',reason:e.code});
  }
 }
 const content=JSON.stringify({archive:name,format:decoded.format,source_sha256:digest(bytes),expanded_bytes:decoded.expandedBytes,member_count:decoded.entries.length,policy:'All regular members are decoded when supported; metadata-only members are listed explicitly. No member code was executed. Nested archives are recursively expanded.',members:inventory});
 const manifest={id:randomUUID(),name,source,kind:'text',content,sha256:digest(Buffer.from(content)),note:'Archive expanded; inventory includes every entry and parsing status; original SHA256 '+digest(bytes)};
 return {items:validateReferences([manifest,...items]),visualSources};
}
async function extractArchiveFile(file,parent,{signal}={}){
 // Explicit command creates a new unique directory, never overwrites project files.
 const {safeRead}=require('./reference-files.cjs');
 const bytes=await safeRead(file,{signal});
 let current=path.parse(path.resolve(parent)).root;
 for(const part of path.resolve(parent).slice(current.length).split(path.sep).filter(Boolean)){
  current=path.join(current,part);const st=await fs.lstat(current);if(st.isSymbolicLink()||!st.isDirectory())fail('REFERENCE_ARCHIVE_DESTINATION','Choose an existing real destination directory');
 }
 const free=await fs.statfs(parent),expandedBytes=Math.min(Number.MAX_SAFE_INTEGER,Number(free.bavail)*Number(free.bsize));
 // Scan verifies paths, CRCs and size without retaining decompressed member bodies.
 // Extraction then streams to disk, so expanded size is bounded by disk, not RAM.
 const inventory=await runDecoder('archive-scan',path.basename(file),bytes,{signal,expandedBytes});
 if(!Array.isArray(inventory.entries))fail('REFERENCE_ARCHIVE_DECODE','Archive inventory is unavailable');
 const directory=await fs.mkdtemp(path.join(parent,'voder-extracted-'));await fs.chmod(directory,0o700);
 try{
  const written=await runDecoder('archive-extract',path.basename(file),bytes,{signal,expandedBytes,destination:directory});
  if(JSON.stringify(written)!==JSON.stringify(inventory))fail('REFERENCE_ARCHIVE_CHANGED','Archive extraction differs from the verified inventory');
  return {status:'EXTRACTED',directory,files:written.entries.filter(e=>e.kind==='file').length,expandedBytes:written.expandedBytes,archive_sha256:digest(bytes)};
 }catch(e){fail(e.code||'REFERENCE_ARCHIVE_EXTRACTION_INCOMPLETE',e.message+'; partial extraction retained only in '+directory);}
}

async function extractArchiveCommand(vscode){
 const selected=await vscode.window.showOpenDialog({title:'Voder · Select archive to extract',canSelectFiles:true,canSelectFolders:false,canSelectMany:false});if(!selected?.length)return;
 const parent=await vscode.window.showOpenDialog({title:'Voder · Choose destination folder (creates a new subfolder)',canSelectFiles:false,canSelectFolders:true,canSelectMany:false});if(!parent?.length)return;
 const result=await extractArchiveFile(selected[0].fsPath,parent[0].fsPath);
 await vscode.window.showInformationMessage('Extracted '+result.files+' files to '+result.directory);return result;
}
module.exports={isArchive,decodeArchive,loadReferences,extractArchiveFile,extractArchiveCommand};
